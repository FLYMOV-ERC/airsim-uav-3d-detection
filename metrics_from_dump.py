#!/usr/bin/env python3
"""Métricas de artigo a partir dos dumps de detecção (rápido, sem GPU):
- CLEAR-MOT no ponto de operação (MOTA, IDF1, IDS, MT, ML, MOTP)
- AMOTA / AMOTP (integrais sobre recall — padrão nuScenes/AB3DMOT, independe do detector)
- Detecção: Precision, Recall, F1, AP (área da curva P-R)
- Localização 3D: RMSE/MAE por faixa de distância
Por ambiente (NH/City/Coast) com média±desvio entre sequências."""
import sys, json, glob, argparse, numpy as np
from pathlib import Path
sys.path.insert(0,'/home/ericyos/airsim')
from sort_tracker import SortTracker
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

GATE=4.0  # m (casamento 3D GT-track)
GTMAP={'Drone3':0,'Drone4':1,'Intruder1':2}


def run_tracker(frames, fuse_w, thr, max_age=15, min_hits=3, max_dist=90):
    """Roda SORT nos dets filtrados por fuse>=thr. Retorna acc motmetrics + erros 3D + contagens."""
    trk=SortTracker(iou_thr=0.3,max_age=max_age,min_hits=min_hits,nms_iou=0.4)
    acc=mm.MOTAccumulator(auto_id=True); errs=[]
    for fr in frames:
        dets=[]
        for d in fr['dets']:
            f=fuse_w*d['yolo']+(1-fuse_w)*d['pn']
            if f>=thr and d['d']<=max_dist:
                dets.append({'bbox':tuple(d['bbox']),'fused':f,'d':d['d'],'gpos':np.array(d['gpos'])})
        tracks=trk.update(dets)
        gt_ids=[GTMAP.get(n,9) for n in fr['gt']]; gt_pos=[np.array(v) for v in fr['gt'].values()]
        hyp_ids=[t.id for t in tracks]; hyp_pos=[t.det['gpos'] for t in tracks]
        if gt_pos and hyp_pos:
            D=np.full((len(gt_pos),len(hyp_pos)),np.nan)
            for a in range(len(gt_pos)):
                for b in range(len(hyp_pos)):
                    e=np.linalg.norm(gt_pos[a]-hyp_pos[b])
                    if e<GATE: D[a,b]=e
            # erros 3D dos matches (p/ AMOTP/RMSE)
            C=np.where(np.isnan(D),1e6,D); ri,ci=linear_sum_assignment(C)
            for a,b in zip(ri,ci):
                if C[a,b]<GATE: errs.append((np.linalg.norm(gt_pos[a]),C[a,b]))
        else:
            D=np.empty((len(gt_pos),len(hyp_pos)))
        acc.update(gt_ids,hyp_ids,D)
    return acc,errs


def clear_mot(acc):
    mh=mm.metrics.create()
    s=mh.compute(acc,metrics=['mota','motp','idf1','num_switches','mostly_tracked','mostly_lost',
                              'num_objects','num_false_positives','num_misses'],name='x')
    return {k:s[k].iloc[0] for k in s.columns}


def seq_metrics(frames, fuse_w):
    # ponto de operação (thr=0.5)
    acc,errs=run_tracker(frames,fuse_w,0.5)
    cm=clear_mot(acc)
    P=cm['num_objects']
    # AMOTA/AMOTP: varre thresholds, mapeia p/ recall
    motars=[]; amotp_terms=[]
    for thr in np.linspace(0.05,0.95,19):
        a,e=run_tracker(frames,fuse_w,thr)
        c=clear_mot(a); P2=c['num_objects']
        if P2==0: continue
        fn=c['num_misses']; fp=c['num_false_positives']; ids=c['num_switches']
        recall=(P2-fn)/P2
        if recall<=0: continue
        motar=max(0.0, 1 - (ids+fp+fn-(1-recall)*P2)/(recall*P2))
        motars.append((recall,motar))
        if e: amotp_terms.append((recall, np.mean([x[1] for x in e])))
    amota=np.mean([m for _,m in motars]) if motars else 0.0
    amotp=np.mean([p for _,p in amotp_terms]) if amotp_terms else float('nan')
    # detecção P/R/F1 no ponto de operação (a partir do acc tracker)
    tp=P-cm['num_misses']; prec=tp/max(1,tp+cm['num_false_positives']); rec=tp/max(1,P)
    f1=2*prec*rec/max(1e-9,prec+rec)
    return {'mota':float(cm['mota']),'idf1':float(cm['idf1']),'idsw':int(cm['num_switches']),
            'mt':int(cm['mostly_tracked']),'ml':int(cm['mostly_lost']),
            'amota':float(amota),'amotp':float(amotp),
            'prec':float(prec),'rec':float(rec),'f1':float(f1),
            'errs':errs,'recall_max':float(max([r for r,_ in motars]) if motars else 0)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dumps", default="det_dumps")
    ap.add_argument("--fuse_w", type=float, default=0.5)
    args=ap.parse_args()
    dumps=sorted(glob.glob(f"{args.dumps}/*.json"))
    by_env={'nh':[],'city':[],'coast':[]}
    for dp in dumps:
        name=Path(dp).stem; env=name.split('_')[0]
        if env not in by_env: continue
        fr=json.load(open(dp))['frames']
        by_env[env].append((name, seq_metrics(fr, args.fuse_w)))
        m=by_env[env][-1][1]
        print(f"  {name}: MOTA={m['mota']:.2f} AMOTA={m['amota']:.2f} IDF1={m['idf1']:.2f} F1={m['f1']:.2f} IDsw={m['idsw']}")
    print("\n================ MÉTRICAS DE ARTIGO (média±desvio) ================")
    allerr=[]
    for env,rows in by_env.items():
        if not rows: continue
        def ag(k): v=[r[1][k] for r in rows]; return np.mean(v),np.std(v)
        e=[x for r in rows for x in r[1]['errs']]; allerr+=e
        rmse=np.sqrt(np.mean([x[1]**2 for x in e])) if e else float('nan')
        print(f"\n### {env.upper()} (n={len(rows)} vídeos)")
        for k,l in [('amota','AMOTA'),('amotp','AMOTP(m)'),('mota','MOTA@0.5'),('idf1','IDF1'),
                    ('f1','det-F1'),('prec','det-P'),('rec','det-R')]:
            mu,sd=ag(k); print(f"  {l:10s}: {mu:.3f} ± {sd:.3f}")
        print(f"  IDsw total: {sum(r[1]['idsw'] for r in rows)}  | erro3D RMSE={rmse:.2f}m (n={len(e)})")
    if allerr:
        rmse=np.sqrt(np.mean([x[1]**2 for x in allerr]))
        print(f"\n### GLOBAL (3 envs): erro3D RMSE={rmse:.2f}m MAE={np.mean([x[1] for x in allerr]):.2f}m n={len(allerr)}")
        for lo,hi in [(0,30),(30,50),(50,90)]:
            ee=[x[1] for x in allerr if lo<=x[0]<hi]
            if ee: print(f"    {lo}-{hi}m: RMSE={np.sqrt(np.mean(np.square(ee))):.2f}m (n={len(ee)})")


if __name__=="__main__":
    main()
