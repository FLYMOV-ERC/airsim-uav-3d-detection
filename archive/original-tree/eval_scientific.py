#!/usr/bin/env python3
"""Avaliação CIENTÍFICA do pipeline em múltiplas sequências com ground truth.
Métricas: detecção (P/R/F1), localização 3D (RMSE/MAE por faixa de distância),
e MOT (MOTA, MOTP, IDF1, ID-switches, MT/ML) via motmetrics.
Agrega média±desvio entre sequências. Compara configs (baseline vs best)."""
import sys, json, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
from ultralytics import YOLO
from detector_frustum_pn2 import FrustumPN2Detector
from sort_tracker import SortTracker
import inference_pipeline as P
from ekf_3d_global import rotation_matrix
import motmetrics as mm

MATCH_GATE_M = 4.0   # par GT-track casa se erro 3D < 4m


def process_seq(sess, yolo, det, cfg):
    meta = json.load(open(sess/"meta.json")); frames = meta["frames"]
    trk = SortTracker(iou_thr=0.3, max_age=cfg['max_age'], min_hits=cfg['min_hits'], nms_iou=0.4)
    acc = mm.MOTAccumulator(auto_id=True)
    pos_errs = []            # (gt_dist, err3d) p/ RMSE por faixa
    n_tp=n_fp=n_fn=0
    for fr in frames:
        i=fr["frame"]; xp=np.array(fr["x_plat"]); gt=fr["gt_global"]
        bgr=cv2.imread(str(sess/"rgb"/f"{i:05d}.jpg")); depth=np.load(sess/"depth"/f"{i:05d}.npy").astype(np.float32)
        pc=P.depth_to_pointcloud_cv(depth)
        r=yolo.predict(bgr,conf=cfg['conf'],imgsz=1280,verbose=False); dets=[]
        if r and r[0].boxes is not None:
            for k in range(len(r[0].boxes.xyxy)):
                bb=tuple(map(float,r[0].boxes.xyxy.cpu().numpy()[k])); c=float(r[0].boxes.conf.cpu().numpy()[k])
                pr=det.predict(pc,depth,bb,c)
                if pr is None: continue
                f=cfg['fuse_w']*c+(1-cfg['fuse_w'])*pr['is_drone_prob']
                d3=float(np.linalg.norm(P.cv_to_frd(pr['center_cv'])))
                if f<cfg['fuse_thr'] or d3>cfg['max_dist']: continue
                frd=P.cv_to_frd(np.array(pr['center_cv']))
                g=xp[:3]+rotation_matrix(xp[6],xp[7],xp[8])@frd
                dets.append({'bbox':bb,'fused':f,'d':d3,'gpos':g})
        tracks=trk.update(dets)
        # GT visíveis (projetam no quadro)
        Rm=rotation_matrix(xp[6],xp[7],xp[8]); gt_ids=[]; gt_pos=[]
        gtmap={'Drone3':0,'Drone4':1,'Intruder1':2}
        for n,gg in gt.items():
            pf=Rm.T@(np.array(gg)-xp[:3])
            if pf[0]<=0.1: continue
            u=pf[1]*P.FX/pf[0]+P.CX; v=pf[2]*P.FY/pf[0]+P.CY
            if 0<=u<P.IMAGE_W and 0<=v<P.IMAGE_H:
                gt_ids.append(gtmap.get(n,99)); gt_pos.append(np.array(gg))
        hyp_ids=[t.id for t in tracks]; hyp_pos=[t.det['gpos'] for t in tracks]
        # matriz de distância 3D (gate)
        if gt_pos and hyp_pos:
            D=np.zeros((len(gt_pos),len(hyp_pos)))
            for a in range(len(gt_pos)):
                for b in range(len(hyp_pos)):
                    e=np.linalg.norm(gt_pos[a]-hyp_pos[b]); D[a,b]=e if e<MATCH_GATE_M else np.nan
        else:
            D=np.empty((len(gt_pos),len(hyp_pos)))
        acc.update(gt_ids, hyp_ids, D)
        # detecção frame-level + erros 3D (matching guloso p/ contagem)
        from scipy.optimize import linear_sum_assignment
        if gt_pos and hyp_pos:
            C=np.where(np.isnan(D),1e6,D); ri,ci=linear_sum_assignment(C); matched=0
            usedg=set();usedh=set()
            for a,b in zip(ri,ci):
                if C[a,b]<MATCH_GATE_M:
                    matched+=1; usedg.add(a);usedh.add(b)
                    pos_errs.append((np.linalg.norm(gt_pos[a]-xp[:3]), C[a,b]))
            n_tp+=matched; n_fp+=len(hyp_pos)-len(usedh); n_fn+=len(gt_pos)-len(usedg)
        else:
            n_fp+=len(hyp_pos); n_fn+=len(gt_pos)
    mh=mm.metrics.create()
    summ=mh.compute(acc, metrics=['mota','motp','idf1','num_switches','mostly_tracked','mostly_lost','num_objects'], name='s')
    prec=n_tp/max(1,n_tp+n_fp); rec=n_tp/max(1,n_tp+n_fn); f1=2*prec*rec/max(1e-9,prec+rec)
    return {'mota':float(summ['mota'].iloc[0]),'motp_m':float(summ['motp'].iloc[0]),
            'idf1':float(summ['idf1'].iloc[0]),'idsw':int(summ['num_switches'].iloc[0]),
            'mt':int(summ['mostly_tracked'].iloc[0]),'ml':int(summ['mostly_lost'].iloc[0]),
            'prec':prec,'rec':rec,'f1':f1,'pos_errs':pos_errs}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sessions", default="sessions/nh_seq*")
    ap.add_argument("--tag", default="cfg")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v4/best.pt")
    ap.add_argument("--det_band", type=float, default=0.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fuse_w", type=float, default=0.5); ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0)
    ap.add_argument("--max_age", type=int, default=8); ap.add_argument("--min_hits", type=int, default=3)
    args=ap.parse_args()
    cfg=dict(conf=args.conf,fuse_w=args.fuse_w,fuse_thr=args.fuse_thr,max_dist=args.max_dist,
             max_age=args.max_age,min_hits=args.min_hits)
    import glob
    sessions=sorted(glob.glob(args.sessions))
    yolo=YOLO("runs/drone/drone_synth_v1/weights/best.pt")
    det=FrustumPN2Detector(checkpoint_path=args.pointnet,cls_threshold=0.0,band_m=args.det_band)
    cache_dir=Path("eval_cache"); cache_dir.mkdir(exist_ok=True)
    rows=[]; all_errs=[]
    for s in sessions:
        try:
            json.load(open(Path(s)/"meta.json"))
        except Exception:
            print(f"  SKIP {Path(s).name} (meta corrompido)"); continue
        cpath=cache_dir/f"{args.tag}__{Path(s).name}.json"
        if cpath.exists():
            m=json.load(open(cpath)); print(f"  CACHE {Path(s).name}")
        else:
            m=process_seq(Path(s),yolo,det,cfg)
            json.dump(m, open(cpath,"w"))
        rows.append((Path(s).name,m)); all_errs+=m['pos_errs']
        print(f"  {Path(s).name}: MOTA={m['mota']:.2f} IDF1={m['idf1']:.2f} "
              f"P/R/F1={m['prec']:.2f}/{m['rec']:.2f}/{m['f1']:.2f} IDsw={m['idsw']}")
    def agg(k): v=[r[1][k] for r in rows]; return np.mean(v),np.std(v)
    print(f"\n===== {args.tag}  (n={len(rows)} sequências) =====")
    for k,lbl in [('mota','MOTA'),('idf1','IDF1'),('motp_m','MOTP(m)'),('f1','det F1'),('prec','det P'),('rec','det R')]:
        mu,sd=agg(k); print(f"  {lbl:10s}: {mu:.3f} ± {sd:.3f}")
    tot_idsw=sum(r[1]['idsw'] for r in rows); print(f"  ID-switches total: {tot_idsw}")
    # erro 3D por faixa de distância
    e=np.array(all_errs)
    if len(e):
        print(f"  Erro 3D global: RMSE={np.sqrt((e[:,1]**2).mean()):.2f}m  MAE={e[:,1].mean():.2f}m  (n={len(e)})")
        for lo,hi in [(0,30),(30,50),(50,90)]:
            mk=(e[:,0]>=lo)&(e[:,0]<hi)
            if mk.sum(): print(f"    {lo}-{hi}m: RMSE={np.sqrt((e[mk,1]**2).mean()):.2f}m (n={mk.sum()})")


if __name__=="__main__":
    main()
