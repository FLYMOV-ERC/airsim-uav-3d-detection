#!/usr/bin/env python3
"""Cache de nuvens de pontos CV (in-range, subamostradas) por frame → acelera o voxel temporal.
Para cada frame: depth→PC(CV)→filtra ao range BEV→subamostra p/ N pontos→salva pccv/<frame>.npy.
Resumível (pula existentes). Roda em paralelo."""
import sys, json, glob, numpy as np
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0,'/home/ericyos/airsim')
import inference_pipeline as P

XMIN,XMAX,ZMIN,ZMAX = -40.0,40.0,2.0,90.0
N=12000

def do_frame(args):
    s,i=args
    s=Path(s); od=s/"pccv"; od.mkdir(exist_ok=True); op=od/f"{i:05d}.npy"
    if op.exists(): return 0
    dfp=s/"depth"/f"{i:05d}.npy"
    if not dfp.exists(): np.save(op,np.zeros((1,3),np.float32)); return 0
    pc=P.depth_to_pointcloud_cv(np.load(dfp).astype(np.float32))
    x=pc[:,0]; z=pc[:,2]; m=(x>=XMIN)&(x<XMAX)&(z>=ZMIN)&(z<ZMAX)
    pc=pc[m]
    if len(pc)>N: pc=pc[np.random.choice(len(pc),N,replace=False)]
    np.save(op,pc.astype(np.float32)); return 1

def main():
    jobs=[]
    for s in sorted(glob.glob("sessions/*_seq*")):
        meta=json.load(open(Path(s)/"meta.json"))
        for fr in meta["frames"]: jobs.append((s,fr["frame"]))
    print(f"{len(jobs)} frames p/ cachear")
    done=0
    with Pool(10) as pool:
        for k,r in enumerate(pool.imap_unordered(do_frame,jobs,chunksize=20)):
            done+=r
            if k%500==0: print(f"  {k}/{len(jobs)}",flush=True)
    print(f"cache pronto ({done} novos)")

if __name__=="__main__": main()
