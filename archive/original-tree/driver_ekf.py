#!/usr/bin/env python3
"""Resumo das métricas SORT-assoc + EKF 3D por dump (média entre os 3 ambientes + RMSE global)."""
import sys, json, glob, numpy as np
from pathlib import Path
import metrics_ekf_from_dump as M

def summarize(dumpdir, fuse_w=0.5):
    by_env={'nh':[],'city':[],'coast':[]}; allerr=[]
    for dp in sorted(glob.glob(f"{dumpdir}/*.json")):
        name=Path(dp).stem; env=name.split('_')[0]
        if env not in by_env: continue
        fr=json.load(open(dp))['frames']
        mp=Path("sessions")/name/"meta.json"
        xpof={f["frame"]:np.array(f["x_plat"]) for f in json.load(open(mp))["frames"]} if mp.exists() else {}
        m=M.seq_metrics(fr,xpof,fuse_w); by_env[env].append(m); allerr+=m['errs']
    envm={}
    for env,rows in by_env.items():
        if rows: envm[env]={k:np.mean([r[k] for r in rows]) for k in ['amota','mota','idf1','prec','rec','f1']}
        if rows: envm[env]['idsw']=sum(r['idsw'] for r in rows)
    mean=lambda k: np.mean([envm[e][k] for e in envm])
    rmse=np.sqrt(np.mean([x[1]**2 for x in allerr])) if allerr else float('nan')
    bands={}
    for lo,hi in [(0,30),(30,50),(50,90)]:
        ee=[x[1] for x in allerr if lo<=x[0]<hi]
        bands[f"{lo}-{hi}"]=np.sqrt(np.mean(np.square(ee))) if ee else float('nan')
    return envm, dict(amota=mean('amota'),mota=mean('mota'),idf1=mean('idf1'),prec=mean('prec'),
                      rec=mean('rec'),f1=mean('f1'),rmse=rmse,idsw=sum(envm[e]['idsw'] for e in envm),
                      bands=bands)

if __name__=="__main__":
    dumps=sys.argv[1:] or ["det_dumps_base","det_dumps_p95","det_dumps_best","det_dumps_convnet",
        "det_dumps_segnet","det_dumps_pp","det_dumps_ppfine","det_dumps_pptemp","det_dumps_vox3d",
        "det_dumps_pfn","det_dumps_fusion_pfn"]
    print(f"{'method':22s} {'AMOTA':>6} {'MOTA':>6} {'IDF1':>6} {'detP':>6} {'detR':>6} {'detF1':>6} {'RMSE':>6} {'IDsw':>6}")
    for d in dumps:
        if not glob.glob(f"{d}/*.json"): print(f"{d:22s}  (sem dump)"); continue
        envm,g=summarize(d)
        print(f"{d.replace('det_dumps_',''):22s} {g['amota']:6.2f} {g['mota']:6.2f} {g['idf1']:6.2f} "
              f"{g['prec']:6.2f} {g['rec']:6.2f} {g['f1']:6.2f} {g['rmse']:6.2f} {g['idsw']:6d}", flush=True)
        if d in ("det_dumps_best","det_dumps_pfn"):
            for e in ('nh','city','coast'):
                v=envm.get(e,{}); print(f"    {e:5s} AMOTA={v.get('amota',0):.2f} MOTA={v.get('mota',0):.2f} "
                      f"IDF1={v.get('idf1',0):.2f} P={v.get('prec',0):.2f} R={v.get('rec',0):.2f} F1={v.get('f1',0):.2f}")
            print(f"    bands: {g['bands']}")
