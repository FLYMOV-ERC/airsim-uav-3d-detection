#!/usr/bin/env python3
"""Dump the RAW detections (YOLO + frustum) per frame, per sequence -- ONE heavy pass.
Saves, per frame, a list of detections [bbox, yolo_conf, pn_prob, gpos (global 3D)] plus the ground truth.
evaluation/metrics.py then recomputes CLEAR-MOT/AMOTA/AP cheaply while sweeping the threshold."""
import sys, json, argparse, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from ultralytics import YOLO
from detection.frustum.detector import FrustumPN2Detector
from archive.early_detectors.detector_painted_v4 import PaintedPointNetDetector
import common.pipeline as P
from tracking.ekf_3d import rotation_matrix


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sessions", default="sessions/*_seq*")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v5band/best.pt")
    ap.add_argument("--det_band", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.05)  # low, so the threshold can be swept afterwards
    ap.add_argument("--detector", default="frustum", choices=["frustum","painted","convnet","segnet"])
    ap.add_argument("--out", default="det_dumps")
    ap.add_argument("--yolo_cache", default="yolo_box_cache")
    args=ap.parse_args()
    import glob
    sessions=sorted(glob.glob(args.sessions))
    out=Path(args.out); out.mkdir(exist_ok=True)
    yolo=YOLO("runs/drone/drone_synth_v1/weights/best.pt")
    if args.detector=="painted":
        det=PaintedPointNetDetector(checkpoint_path=args.pointnet,cls_threshold=0.0)
    else:
        det=FrustumPN2Detector(checkpoint_path=args.pointnet,cls_threshold=0.0,band_m=args.det_band,arch=(args.detector if args.detector in ("convnet","segnet") else "pointnet"))
    ycdir=Path(args.yolo_cache); ycdir.mkdir(exist_ok=True)
    for s in sessions:
        s=Path(s); dump_p=out/f"{s.name}.json"
        ycp=ycdir/f"{s.name}.json"; ycache=json.load(open(ycp)) if ycp.exists() else {}
        if dump_p.exists():
            print(f"CACHE {s.name}"); continue
        try: meta=json.load(open(s/"meta.json"))
        except Exception: print(f"SKIP {s.name} (meta ruim)"); continue
        frames_out=[]
        for fr in meta["frames"]:
            i=fr["frame"]; xp=np.array(fr["x_plat"]); gt=fr["gt_global"]
            img=s/"rgb"/f"{i:05d}.jpg"; dp=s/"depth"/f"{i:05d}.npy"
            if not img.exists() or not dp.exists(): continue
            bgr=cv2.imread(str(img));
            if bgr is None: continue
            depth=np.load(dp).astype(np.float32); pc=P.depth_to_pointcloud_cv(depth)
            # YOLO boxes: use the cache if present, otherwise run YOLO and save it
            ckey=ycache.get(f"{s.name}/{i}")
            if ckey is None:
                r=yolo.predict(bgr,conf=args.conf,imgsz=1280,verbose=False); boxes=[]
                if r and r[0].boxes is not None:
                    for k in range(len(r[0].boxes.xyxy)):
                        boxes.append([list(map(float,r[0].boxes.xyxy.cpu().numpy()[k])),float(r[0].boxes.conf.cpu().numpy()[k])])
                ycache[f"{s.name}/{i}"]=boxes
            else:
                boxes=ckey
            dets=[]
            for bb,c in boxes:
                    pr=det.predict(pc,depth,tuple(bb),c)
                    if pr is None: continue
                    frd=P.cv_to_frd(np.array(pr['center_cv']))
                    g=(xp[:3]+rotation_matrix(xp[6],xp[7],xp[8])@frd)
                    d3=float(np.linalg.norm(P.cv_to_frd(pr['center_cv'])))
                    dets.append({'bbox':bb,'yolo':c,'pn':float(pr['is_drone_prob']),
                                 'gpos':[float(x) for x in g],'d':d3})
            # visible (in-frame) ground truth
            Rm=rotation_matrix(xp[6],xp[7],xp[8]); gtv={}
            for n,gg in gt.items():
                pf=Rm.T@(np.array(gg)-xp[:3])
                if pf[0]<=0.1: continue
                u=pf[1]*P.FX/pf[0]+P.CX; v=pf[2]*P.FY/pf[0]+P.CY
                if 0<=u<P.IMAGE_W and 0<=v<P.IMAGE_H: gtv[n]=[float(x) for x in gg]
            frames_out.append({'frame':i,'ts':fr['ts'],'dets':dets,'gt':gtv})
        json.dump({'name':s.name,'frames':frames_out}, open(dump_p,'w'))
        json.dump(ycache, open(ycp,'w'))
        npos=sum(len(f['gt']) for f in frames_out); ndet=sum(len(f['dets']) for f in frames_out)
        print(f"DUMP {s.name}: {len(frames_out)}f, {ndet} dets, {npos} GT-vis")


if __name__=="__main__":
    main()
