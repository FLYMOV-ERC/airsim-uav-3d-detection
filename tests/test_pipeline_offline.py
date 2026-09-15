#!/usr/bin/env python3
"""Smoke test of the end-to-end pipeline OFFLINE (no AirSim needed).

Loads one frame of the dataset and simulates the detection + filter threads:
  YOLO predictions (do disco) → PointNet++ → spherical → EKF
Renders the overlay. Validates that all the pieces integrate.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import torch

from ultralytics import YOLO

from tracking.ekf_3d import EKFParams
from tracking.multi_ekf import MultiEKFTracker, MultiTrackerParams
from archive.early_detectors.pointnet_inference import PointNetInference
from common.pipeline import (
    depth_to_pointcloud_cv, cv_to_frd, frd_to_spherical, draw_overlay,
    project_global_to_pixel, CAMERA_PITCH_OFFSET,
)


def airsim_to_cv(p):
    return np.array([p[1], p[2], p[0]], dtype=np.float32)


def main():
    dataset = "dataset_nh_v2"
    yolo_path = "runs/drone/drone_urban_v1/weights/best.pt"
    pn_path = "runs/pointnet2_painted/v4_balanced/best.pt"

    # Load one frame
    root = Path(dataset)
    frame = "frame_000000"
    img_path = None
    for split in ("train", "val"):
        p = root / "yolo" / "images" / split / f"{frame}.jpg"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        # take any frame
        for split in ("train", "val"):
            cand = sorted((root / "yolo" / "images" / split).glob("*.jpg"))
            if cand:
                img_path = cand[0]
                frame = img_path.stem
                break
    pc_path = root / "pointnet" / "point_clouds" / f"{frame}.npy"
    meta_path = root / "metadata" / f"{frame}.json"

    bgr = cv2.imread(str(img_path))
    pc = np.load(pc_path).astype(np.float32)
    meta = json.load(open(meta_path))
    ego_pos = np.array(meta["ego_xyz"], dtype=np.float64)
    yaw = float(meta["ego_yaw_rad"])
    print(f"Frame: {img_path.name}  PC={len(pc)} pts  ego pos={ego_pos}  yaw={np.degrees(yaw):.1f}°")

    # Rebuild x_plat (yaw only -- roll/pitch assumed zero while hovering)
    pitch_eff = 0.0 + CAMERA_PITCH_OFFSET
    x_plat = np.array([ego_pos[0], ego_pos[1], ego_pos[2],
                       0., 0., 0., 0.0, pitch_eff, yaw], dtype=np.float64)
    print(f"x_plat: pos=({x_plat[0]:.1f},{x_plat[1]:.1f},{x_plat[2]:.1f})  "
          f"rpy=({np.degrees(x_plat[6]):.1f}, {np.degrees(x_plat[7]):.1f}, {np.degrees(x_plat[8]):.1f})°")

    # Modelos
    print("Loading models...")
    yolo = YOLO(yolo_path)
    pn = PointNetInference(pn_path, cls_threshold=0.3)

    # YOLO inference
    results = yolo.predict(bgr, conf=0.25, verbose=False)
    bboxes = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            bboxes.append((tuple(map(float, xyxy[i])), float(confs[i])))
    print(f"YOLO: {len(bboxes)} bboxes")

    # PointNet per bbox + spherical
    measurements = []
    for bbox, conf in bboxes:
        pred = pn.predict(pc, bbox, conf)
        if pred is None:
            continue
        if not pred['is_drone']:
            print(f"  PN rejected: bbox={bbox}  prob={pred['is_drone_prob']:.2f}")
            continue
        p_frd = cv_to_frd(pred['center_cv'])
        d, phi, theta = frd_to_spherical(p_frd)
        r = float(np.clip(max(pred['size_cv']) / 2.0, 0.1, 10.0))
        measurements.append({
            'd': d, 'phi': phi, 'theta': theta, 'r': r,
            'bbox': list(bbox), 'yolo_conf': conf,
            'pn_prob': pred['is_drone_prob'],
            'center_cv': pred['center_cv'].tolist(),
            'size_cv': pred['size_cv'].tolist(),
        })
        print(f"  Meas: d={d:.1f}m  phi={np.degrees(phi):.1f}°  theta={np.degrees(theta):.1f}°  r={r:.2f}m")

    # Filter: simulate 5 frames with the same (static) measurement to confirm the tracks
    print("\n--- Filter (5 frames of identical measurements) ---")
    tracker = MultiEKFTracker(MultiTrackerParams(min_hits=2, max_age=5, chi2_gate=13.28))
    meas_tup = [(m['d'], m['phi'], m['theta'], m['r'], m['yolo_conf']) for m in measurements]
    for k in range(5):
        tracks = tracker.update(meas_tup, x_plat, dt=0.1)
    print(f"Confirmed tracks: {len(tracks)} / total_created={tracker.total_created}")
    for t in tracks:
        pos = t.position_global()
        print(f"  ID {t.id}: pos_global=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})m  "
              f"hits={t.hits}  age={t.age}  r={t.radius():.2f}m")

    # GT comparison
    lbl_path = root / "pointnet" / "labels_3d" / "frame_000000.json"
    if lbl_path.exists():
        gt = json.load(open(lbl_path))
        print(f"\n--- GT comparison ({len(gt)} GT) ---")
        for g in gt:
            gt_cv = airsim_to_cv(g["center"])
    # The ground truth is in the CV camera frame; convert to global using R_m
            from tracking.ekf_3d import rotation_matrix
            R_m = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
    # CV -> FRD (the same conversion as for the prediction)
            gt_frd = cv_to_frd(gt_cv)
            gt_global = x_plat[:3] + R_m.T @ gt_frd
            print(f"  GT '{g.get('name','?')}': camera_cv={gt_cv.round(1).tolist()}  global_NED={gt_global.round(1).tolist()}")
            best = min(tracks, key=lambda t: np.linalg.norm(t.position_global() - gt_global),
                       default=None) if tracks else None
            if best is not None:
                d = float(np.linalg.norm(best.position_global() - gt_global))
                print(f"    closest track ID {best.id}: err={d:.2f}m")

    # Render overlay
    pkt = {
        'frame': 0, 'bgr': bgr, 'x_plat': x_plat,
        'measurements': measurements,
        'tracks': [t.to_dict() for t in tracks],
        'histories': {t.id: [p.tolist() for p in t.history] for t in tracks},
        'track_bboxes': {},
        'stats': tracker.statistics(),
    }
    # Match a box to a track (simple projection-based heuristic)
    for t in tracks:
        u, v, _ = project_global_to_pixel(t.position_global(), x_plat)
        if u is None: continue
        best_iou, best_bbox = 0, None
        for m in measurements:
            bx1, by1, bx2, by2 = m['bbox']
            if bx1 <= u <= bx2 and by1 <= v <= by2:
                area = (bx2-bx1)*(by2-by1)
                if 1.0/max(area, 1) > best_iou:
                    best_iou = 1.0/max(area, 1)
                    best_bbox = m['bbox']
        if best_bbox:
            pkt['track_bboxes'][t.id] = best_bbox

    vis = draw_overlay(pkt)
    out_path = "/tmp/pipeline_offline_test.png"
    cv2.imwrite(out_path, vis)
    print(f"\nOverlay saved: {out_path}")
    print("=== END-TO-END OFFLINE TEST PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
