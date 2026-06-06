#!/usr/bin/env python3
"""Processa sessão gravada com YOLO@1280 → frustum PointNet → fusão → NMS → SORT (IoU 2D).
Tracks vêm SÓ de detecções confirmadas (fused>=thr); NMS dedup; SORT associa por IoU.
"""
import sys, json, argparse
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
from ultralytics import YOLO
from detector_frustum_pn2 import FrustumPN2Detector
from sort_tracker import SortTracker
import inference_pipeline as P

PALETTE = [(0,255,0),(255,128,0),(0,128,255),(255,0,255),(0,255,255),(128,0,255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_rec")
    ap.add_argument("--yolo", default="runs/drone/drone_synth_v1/weights/best.pt")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v4/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--fuse_w", type=float, default=0.5)
    ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=90.0, help="rejeita medida 3D além disso (ruído de fundo)")
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--max_age", type=int, default=8)
    ap.add_argument("--det_band", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sess = Path(args.session)
    meta = json.load(open(sess/"meta.json")); frames = meta["frames"]; rec_fps = meta["fps"]
    print(f"[SORT-offline] {len(frames)} frames @ {rec_fps:.1f} fps")

    yolo = YOLO(args.yolo)
    det = FrustumPN2Detector(checkpoint_path=args.pointnet, cls_threshold=0.0, band_m=args.det_band)
    tracker = SortTracker(iou_thr=0.3, max_age=args.max_age, min_hits=args.min_hits, nms_iou=0.4)

    out_path = args.out or str(sess/"offline_sort.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             max(1.0, rec_fps), (P.IMAGE_W, P.IMAGE_H))
    from ekf_3d_global import rotation_matrix
    n_yolo=0; n_conf=0; dists=[]
    for fr in frames:
        i = fr["frame"]; x_plat = np.array(fr["x_plat"]); gt = fr.get("gt_global", {})
        bgr = cv2.imread(str(sess/"rgb"/f"{i:05d}.jpg"))
        depth = np.load(sess/"depth"/f"{i:05d}.npy").astype(np.float32)
        pc = P.depth_to_pointcloud_cv(depth)
        res = yolo.predict(bgr, conf=args.yolo_conf, imgsz=args.imgsz, verbose=False)
        dets = []
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            xyxy = res[0].boxes.xyxy.cpu().numpy(); confs = res[0].boxes.conf.cpu().numpy()
            for k in range(len(xyxy)):
                bb = tuple(map(float, xyxy[k])); conf = float(confs[k]); n_yolo += 1
                pred = det.predict(pc, depth, bb, conf)
                if pred is None: continue
                pn = pred['is_drone_prob']; fused = args.fuse_w*conf + (1-args.fuse_w)*pn
                d3 = float(np.linalg.norm(P.cv_to_frd(pred['center_cv'])))
                if fused < args.fuse_thr: continue          # só confirmados pela fusão PN
                if d3 > args.max_dist: continue              # rejeita 3D absurdo (fundo)
                dets.append({'bbox': bb, 'fused': fused, 'd': d3,
                             'yolo_conf': conf, 'pn_prob': pn})
        tracks = tracker.update(dets)
        # overlay
        vis = bgr.copy()
        for d in dets:  # caixas confirmadas (antes do track) em cinza fino
            x1,y1,x2,y2=[int(v) for v in d['bbox']]
            cv2.rectangle(vis,(x1,y1),(x2,y2),(150,150,150),1)
        for t in tracks:
            n_conf += 1; dists.append(t.det['d'])
            col = PALETTE[t.id % len(PALETTE)]
            x1,y1,x2,y2=[int(v) for v in t.bbox]
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            cv2.putText(vis,f"ID{t.id} {t.det['d']:.0f}m F{t.det['fused']:.2f}",(x1,max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
            pts=[(int(a),int(b)) for a,b in t.history]
            for a in range(1,len(pts)): cv2.line(vis,pts[a-1],pts[a],col,2)
        # GT (verdade) projetado — cruz amarela: prova que o track bate com o drone real
        Rm = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
        for name, g in gt.items():
            pf = Rm.T @ (np.array(g) - x_plat[:3])
            if pf[0] <= 0.1: continue
            gu = pf[1]*P.FX/pf[0] + P.CX; gv = pf[2]*P.FY/pf[0] + P.CY
            if 0<=gu<P.IMAGE_W and 0<=gv<P.IMAGE_H:
                cv2.drawMarker(vis,(int(gu),int(gv)),(0,255,255),cv2.MARKER_CROSS,28,2)
        st=tracker.stats()
        cv2.putText(vis,f"f{i} yolo_dets={len(dets)} tracks={st['confirmed']} (amarelo=GT)",(10,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
        cv2.putText(vis,f"f{i} yolo_dets={len(dets)} tracks={st['confirmed']}",(10,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        writer.write(vis)
        if i % 20 == 0: print(f"  f{i}: dets={len(dets)} tracks_conf={st['confirmed']} active={st['active']}")
    writer.release()
    print(f"\n[Done] {out_path}")
    print(f"  YOLO={n_yolo}  track-frames confirmados={n_conf}")
    if dists: print(f"  dist 3D: media={np.mean(dists):.1f}m range=[{min(dists):.0f},{max(dists):.0f}]")


if __name__ == "__main__":
    main()
