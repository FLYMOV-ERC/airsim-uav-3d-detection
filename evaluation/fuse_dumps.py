#!/usr/bin/env python3
"""Late (decision-level) fusion of two detection dumps, e.g. T-Net (+) PointPillars.
Union of the detections, matched by 3D proximity (agreement), with the score adjusted by agreement:
  - ambos concordam (<gate 3D): score alto (confirmado pelos dois paradigmas)
  - frustum only (precise): keeps its score
  - voxel only (higher recall, noisier): score with a discount
3D NMS. The output is in dump format, so evaluation/metrics.py computes CLEAR-MOT/AMOTA from it."""
import sys, json, glob, argparse, numpy as np
from pathlib import Path

GATE3D = 3.0      # m, below which two detections are taken to be the same object
NMS3D = 2.0       # m p/ deduplicar
PP_DISCOUNT = 0.65
AGREE_BONUS = 0.25


def score_of(d):  # base score of a detection (fused yolo x pn, as everywhere else)
    return 0.5*d['yolo'] + 0.5*d['pn']


def nms3d(dets, r=NMS3D):
    dets = sorted(dets, key=lambda d: d['score'], reverse=True); keep=[]
    for d in dets:
        if all(np.linalg.norm(np.array(d['gpos'])-np.array(k['gpos']))>r for k in keep):
            keep.append(d)
    return keep


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a", default="det_dumps_segnet")   # frustum (preciso)
    ap.add_argument("--b", default="det_dumps_pp")        # voxel (recall)
    ap.add_argument("--out", default="det_dumps_fusion")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(exist_ok=True)
    for ap_ in sorted(glob.glob(f"{args.a}/*.json")):
        name=Path(ap_).stem; bp=Path(args.b)/f"{name}.json"
        if not bp.exists(): continue
        A=json.load(open(ap_))['frames']; B={f['frame']:f for f in json.load(open(bp))['frames']}
        frames_out=[]
        for fa in A:
            i=fa['frame']; fb=B.get(i,{'dets':[]})
            seg=[{**d,'score':score_of(d)} for d in fa['dets']]
            pp=[{**d,'score':d['yolo']} for d in fb['dets']]   # pp: yolo=pn=heatmap
            fused=[]
            # frustum: kept; if a voxel detection agrees, it gets a bonus
            pp_used=set()
            for d in seg:
                g=np.array(d['gpos']); agree=False
                for k,e in enumerate(pp):
                    if np.linalg.norm(g-np.array(e['gpos']))<GATE3D:
                        agree=True; pp_used.add(k)
                sc=min(1.0, d['score']+AGREE_BONUS) if agree else d['score']
                fused.append({'bbox':d['bbox'],'gpos':d['gpos'],'d':d['d'],'score':sc,'src':'both' if agree else 'frustum'})
            # voxel alone (unmatched): extra recall, at a discount
            for k,e in enumerate(pp):
                if k not in pp_used:
                    fused.append({'bbox':e['bbox'],'gpos':e['gpos'],'d':e['d'],'score':e['score']*PP_DISCOUNT,'src':'voxel'})
            fused=nms3d(fused)
            # dump format: yolo = pn = score (the metrics use fused = 0.5y + 0.5p = score)
            dets=[{'bbox':d['bbox'],'yolo':d['score'],'pn':d['score'],'gpos':d['gpos'],'d':d['d']} for d in fused]
            frames_out.append({'frame':i,'ts':fa['ts'],'dets':dets,'gt':fa['gt']})
        json.dump({'name':name,'frames':frames_out},open(out/f"{name}.json",'w'))
    print(f"fusion written to {out} ({len(glob.glob(f'{out}/*.json'))} sessions)")


if __name__=="__main__":
    main()
