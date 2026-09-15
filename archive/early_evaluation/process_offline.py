#!/usr/bin/env python3
"""ARCHIVED. PHASE 2 -- process a recorded session with the FULL pipeline.

YOLO at 1280 -> frustum PointNet -> YOLO x PointNet fusion -> multi-target EKF tracker.

Being offline there is no time pressure, so it runs at full resolution and writes
the video at the real recording frame rate, with stable tracks (IDs, trails,
inter-frame prediction).

Uses the multi-target EKF association that predates SORT; superseded by
scripts/run_offline_sort.py once 2D association proved more stable.
"""
import sys, json, argparse
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from ultralytics import YOLO
from detection.frustum.detector import FrustumPN2Detector
from tracking.multi_ekf import MultiEKFTracker, MultiTrackerParams
from tracking.ekf_3d import EKFParams
import common.pipeline as P# depth_to_pointcloud_cv, cv_to_frd, frd_to_spherical, draw_overlay, project_global_to_pixel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session_rec")
    ap.add_argument("--yolo", default="runs/drone/drone_synth_v1/weights/best.pt")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v4/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--fuse_w", type=float, default=0.5)
    ap.add_argument("--fuse_thr", type=float, default=0.5)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--max_age", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sess = Path(args.session)
    meta = json.load(open(sess/"meta.json"))
    frames = meta["frames"]; rec_fps = meta["fps"]
    print(f"[Offline] {len(frames)} frames @ {rec_fps:.1f} fps gravados")

    yolo = YOLO(args.yolo)
    det = FrustumPN2Detector(checkpoint_path=args.pointnet, cls_threshold=0.0)  # thresholding happens in the fusion
    tracker = MultiEKFTracker(MultiTrackerParams(
        chi2_gate=13.28, max_age=args.max_age, min_hits=args.min_hits, ekf_params=EKFParams()))

    out_path = args.out or str(sess/"offline_tracked.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             max(1.0, rec_fps), (P.IMAGE_W, P.IMAGE_H))
    prev_ts = None
    n_yolo_tot = 0; n_det_tot = 0; dists = []
    for fr in frames:
        i = fr["frame"]; ts = fr["ts"]; x_plat = np.array(fr["x_plat"], dtype=np.float64)
        bgr = cv2.imread(str(sess/"rgb"/f"{i:05d}.jpg"))
        depth = np.load(sess/"depth"/f"{i:05d}.npy").astype(np.float32)
        pc = P.depth_to_pointcloud_cv(depth)

        # YOLO at 1280 (the training resolution)
        res = yolo.predict(bgr, conf=args.yolo_conf, imgsz=args.imgsz, verbose=False)
        measurements = []
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            xyxy = res[0].boxes.xyxy.cpu().numpy(); confs = res[0].boxes.conf.cpu().numpy()
            for k in range(len(xyxy)):
                bb = tuple(map(float, xyxy[k])); conf = float(confs[k])
                n_yolo_tot += 1
                pred = det.predict(pc, depth, bb, conf)
                if pred is None: continue
                pn = pred['is_drone_prob']
                fused = args.fuse_w*conf + (1-args.fuse_w)*pn
                if fused < args.fuse_thr:
                    continue
                p_frd = P.cv_to_frd(pred['center_cv'])
                d, phi, theta = P.frd_to_spherical(p_frd)
                r = float(np.clip(max(pred['size_cv'])/2.0, 0.1, 10.0))
                measurements.append({
                    'd': d, 'phi': phi, 'theta': theta, 'r': r,
                    'bbox': list(bb), 'yolo_conf': conf, 'pn_prob': pn,
                    'center_cv': pred['center_cv'].tolist(), 'size_cv': pred['size_cv'].tolist(),
                })
                n_det_tot += 1; dists.append(d)

        dt = float(np.clip((ts - prev_ts) if prev_ts else 0.1, 0.01, 1.0)); prev_ts = ts
        meas = [(m['d'], m['phi'], m['theta'], m['r'], m['yolo_conf']) for m in measurements]
        tracks = tracker.update(meas, x_plat, dt)
        histories = {t.id: [p.tolist() for p in t.history[-60:]] for t in tracks}
        # associate the box with the track (same logic as the pipeline)
        track_bboxes = {}
        from tracking.ekf_3d import rotation_matrix
        for t in tracks:
            best_dist, best_bbox = float('inf'), None
            for m in measurements:
                mp = np.array(m['center_cv'])
                p_cv_pred = np.array([(rotation_matrix(x_plat[6],x_plat[7],x_plat[8])
                    @ (np.array(t.position_global())-x_plat[:3]))[j] for j in (1,2,0)])
                dd = float(np.linalg.norm(p_cv_pred - mp))
                if dd < best_dist: best_dist, best_bbox = dd, m['bbox']
            if best_dist < 8.0 and best_bbox is not None:
                track_bboxes[t.id] = best_bbox

        pkt = {'bgr': bgr, 'frame': i, 'x_plat': x_plat, 'measurements': measurements,
               'tracks': [t.to_dict() for t in tracks], 'histories': histories,
               'track_bboxes': track_bboxes, 'stats': tracker.statistics()}
        vis = P.draw_overlay(pkt)
        writer.write(vis)
        if i % 10 == 0:
            print(f"  f{i}: yolo+pn meas={len(measurements)} tracks={tracker.statistics()['active']} conf={tracker.statistics()['confirmed']}")
    writer.release()
    print(f"\n[Done] video: {out_path}")
    print(f"  YOLO total={n_yolo_tot}  det3D(aceitos)={n_det_tot}")
    if dists:
        print(f"  dist 3D: media={np.mean(dists):.1f}m range=[{min(dists):.0f},{max(dists):.0f}]")


if __name__ == "__main__":
    main()
