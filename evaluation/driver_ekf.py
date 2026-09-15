#!/usr/bin/env python3
"""Summarise the SORT-association + 3D-EKF metrics per dump (mean over the three environments + global RMSE).

Usage:
    python evaluation/driver_ekf.py [DUMPDIR ...] [--sessions DIR] [--fuse-weight W]

With no DUMPDIR the eleven dump directories of the Chapter 7 comparison are used;
directories that hold no ``*.json`` are reported as "(no dump)" and skipped.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root

import argparse, json, glob, numpy as np
from pathlib import Path
import evaluation.metrics_ekf as M

DEFAULT_DUMPS = ["det_dumps_base","det_dumps_p95","det_dumps_best","det_dumps_convnet",
    "det_dumps_segnet","det_dumps_pp","det_dumps_ppfine","det_dumps_pptemp","det_dumps_vox3d",
    "det_dumps_pfn","det_dumps_fusion_pfn"]

def summarize(dumpdir, fuse_w=0.5, sessions_root="sessions"):
    by_env={'nh':[],'city':[],'coast':[]}; allerr=[]
    for dp in sorted(glob.glob(f"{dumpdir}/*.json")):
        name=Path(dp).stem; env=name.split('_')[0]
        if env not in by_env: continue
        fr=json.load(open(dp))['frames']
        mp=Path(sessions_root)/name/"meta.json"
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

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-environment aggregation of the SORT + 3D-EKF metrics, one row per dump directory.")
    ap.add_argument("dumps", nargs="*", default=None, metavar="DUMPDIR",
                    help="dump directories to score (default: the eleven Chapter 7 dumps)")
    ap.add_argument("--sessions", default="sessions", metavar="DIR",
                    help="root holding <sequence>/meta.json for the platform poses (default: sessions)")
    ap.add_argument("--fuse-weight", type=float, default=0.5, metavar="W",
                    help="detection/EKF fusion weight passed to metrics_ekf.seq_metrics (default: 0.5)")
    return ap.parse_args(argv)

if __name__=="__main__":
    args = parse_args()
    dumps = args.dumps or DEFAULT_DUMPS
    print(f"{'method':22s} {'AMOTA':>6} {'MOTA':>6} {'IDF1':>6} {'detP':>6} {'detR':>6} {'detF1':>6} {'RMSE':>6} {'IDsw':>6}")
    for d in dumps:
        if not glob.glob(f"{d}/*.json"): print(f"{d:22s}  (no dump)"); continue
        envm,g=summarize(d, args.fuse_weight, args.sessions)
        print(f"{d.replace('det_dumps_',''):22s} {g['amota']:6.2f} {g['mota']:6.2f} {g['idf1']:6.2f} "
              f"{g['prec']:6.2f} {g['rec']:6.2f} {g['f1']:6.2f} {g['rmse']:6.2f} {g['idsw']:6d}", flush=True)
        if d in ("det_dumps_best","det_dumps_pfn"):
            for e in ('nh','city','coast'):
                v=envm.get(e,{}); print(f"    {e:5s} AMOTA={v.get('amota',0):.2f} MOTA={v.get('mota',0):.2f} "
                      f"IDF1={v.get('idf1',0):.2f} P={v.get('prec',0):.2f} R={v.get('rec',0):.2f} F1={v.get('f1',0):.2f}")
            print(f"    bands: {g['bands']}")
