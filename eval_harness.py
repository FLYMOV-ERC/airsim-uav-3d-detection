#!/usr/bin/env python3
"""Harness de avaliação do pipeline contra GROUND TRUTH (session gravada).
Mede, por drone: recall YOLO (2D), recall fusão (após 3D), e track-on-GT %.
Parametrizável p/ comparar hipóteses (conf, fusão, SAHI, max_age, depth-band)."""
import json, argparse, numpy as np, sys, cv2
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/home/ericyos/airsim')
from ultralytics import YOLO
from detector_frustum_pn2 import FrustumPN2Detector
from sort_tracker import SortTracker
import inference_pipeline as P
from ekf_3d_global import rotation_matrix

GT_PX = 60  # raio p/ casar GT a uma detecção/track


def sahi_detect(yolo, bgr, conf, imgsz, nx, ny, overlap, full=True):
    """YOLO com tiling. Retorna lista de (bbox_global, conf). NMS global simples."""
    H, W = bgr.shape[:2]
    boxes = []
    if full:
        r = yolo.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)
        if r and r[0].boxes is not None:
            for k in range(len(r[0].boxes.xyxy)):
                boxes.append((list(map(float, r[0].boxes.xyxy.cpu().numpy()[k])),
                              float(r[0].boxes.conf.cpu().numpy()[k])))
    tw, th = int(W/nx), int(H/ny)
    ox, oy = int(tw*overlap), int(th*overlap)
    for iy in range(ny):
        for ix in range(nx):
            x0 = max(0, ix*tw - ox); y0 = max(0, iy*th - oy)
            x1 = min(W, (ix+1)*tw + ox); y1 = min(H, (iy+1)*th + oy)
            tile = bgr[y0:y1, x0:x1]
            r = yolo.predict(tile, conf=conf, imgsz=imgsz, verbose=False)
            if r and r[0].boxes is not None:
                for k in range(len(r[0].boxes.xyxy)):
                    b = r[0].boxes.xyxy.cpu().numpy()[k]
                    boxes.append(([float(b[0])+x0, float(b[1])+y0, float(b[2])+x0, float(b[3])+y0],
                                  float(r[0].boxes.conf.cpu().numpy()[k])))
    # NMS global
    def iou(a, b):
        ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
        iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
        ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua>0 else 0
    boxes.sort(key=lambda x: x[1], reverse=True)
    keep=[]
    for b in boxes:
        if all(iou(b[0], k[0]) < 0.5 for k in keep): keep.append(b)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_rec2")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--yolo", default="runs/drone/drone_synth_v1/weights/best.pt")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v4/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--fuse_w", type=float, default=0.5)
    ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0)
    ap.add_argument("--max_age", type=int, default=8)
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--nx", type=int, default=2); ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--det_band", type=float, default=0.0, help="depth-band no detector (casar com treino)")
    args = ap.parse_args()

    sess = Path(args.session); meta = json.load(open(sess/"meta.json")); frames = meta["frames"]
    yolo = YOLO(args.yolo)
    det = FrustumPN2Detector(checkpoint_path=args.pointnet, cls_threshold=0.0, band_m=args.det_band)
    trk = SortTracker(iou_thr=0.3, max_age=args.max_age, min_hits=args.min_hits, nms_iou=0.4)

    vis=defaultdict(int); yhit=defaultdict(int); fhit=defaultdict(int)
    gt_tot=0; gt_trk=0; perrs=[]
    for fr in frames:
        i=fr["frame"]; xp=np.array(fr["x_plat"]); gt=fr["gt_global"]
        bgr=cv2.imread(str(sess/"rgb"/f"{i:05d}.jpg")); depth=np.load(sess/"depth"/f"{i:05d}.npy").astype(np.float32)
        pc=P.depth_to_pointcloud_cv(depth)
        if args.sahi:
            yb = sahi_detect(yolo, bgr, args.conf, args.imgsz, args.nx, args.ny, 0.2)
        else:
            r = yolo.predict(bgr, conf=args.conf, imgsz=args.imgsz, verbose=False)
            yb = []
            if r and r[0].boxes is not None:
                for k in range(len(r[0].boxes.xyxy)):
                    yb.append((list(map(float, r[0].boxes.xyxy.cpu().numpy()[k])),
                               float(r[0].boxes.conf.cpu().numpy()[k])))
        # frustum + fusão
        boxes=[]; dets=[]
        for bb, c in yb:
            pr = det.predict(pc, depth, tuple(bb), c)
            f = args.fuse_w*c + (1-args.fuse_w)*pr['is_drone_prob'] if pr else 0
            d3 = float(np.linalg.norm(P.cv_to_frd(pr['center_cv']))) if pr else 999
            boxes.append((bb, c, f))
            if pr and f >= args.fuse_thr and d3 <= args.max_dist:
                dets.append({'bbox': tuple(bb), 'fused': f, 'd': d3})
        tracks = trk.update(dets)
        Rm = rotation_matrix(xp[6], xp[7], xp[8])
        for n, g in gt.items():
            pf = Rm.T @ (np.array(g) - xp[:3])
            if pf[0] <= 0.1: continue
            u = pf[1]*P.FX/pf[0]+P.CX; v = pf[2]*P.FY/pf[0]+P.CY
            if not (0<=u<P.IMAGE_W and 0<=v<P.IMAGE_H): continue
            vis[n]+=1; gt_tot+=1
            near=[(bb,c,f) for bb,c,f in boxes if np.hypot((bb[0]+bb[2])/2-u,(bb[1]+bb[3])/2-v)<GT_PX]
            if near: yhit[n]+=1
            if any(f>=args.fuse_thr for _,_,f in near): fhit[n]+=1
            best=1e9
            for t in tracks:
                e=np.hypot((t.bbox[0]+t.bbox[2])/2-u,(t.bbox[1]+t.bbox[3])/2-v)
                best=min(best,e)
            if best<GT_PX: gt_trk+=1; perrs.append(best)

    print(f"\n===== {args.tag} =====")
    print(f"params: conf={args.conf} imgsz={args.imgsz} fuse_w={args.fuse_w} thr={args.fuse_thr} "
          f"max_age={args.max_age} sahi={args.sahi}({args.nx}x{args.ny})")
    for n in sorted(vis):
        print(f"  {n}: YOLO={100*yhit[n]/vis[n]:.0f}%  fusao={100*fhit[n]/vis[n]:.0f}%")
    yall = 100*sum(yhit.values())/max(1,sum(vis.values()))
    fall = 100*sum(fhit.values())/max(1,sum(vis.values()))
    print(f"  GLOBAL: YOLO={yall:.0f}%  fusao={fall:.0f}%  track-on-GT={100*gt_trk/max(1,gt_tot):.0f}%"
          f"  pix_err={np.median(perrs) if perrs else 0:.0f}px")


if __name__ == "__main__":
    main()
