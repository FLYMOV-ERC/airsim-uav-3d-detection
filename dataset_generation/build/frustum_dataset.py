#!/usr/bin/env python3
"""Build the frustum dataset for PointNet++ from YOLO boxes at conf=0.10.

Per YOLO box:
  - crop the frustum (point-cloud points inside the projected 2D box)
  - resample to a fixed N=4096 (with replacement if there are too few)
  - normaliza XYZ (centroide + escala)
  - match against the 3D ground truth by 2D IoU >= 0.3
      match → positivo: center+size regression
      mismatch → false positive: cls=0, dummy regression

Random negatives are added (random crops outside every ground-truth box).

Input: dataset_*/yolo_predictions_c10/ + dataset_*/yolo/labels/ + pointnet/point_clouds + labels_3d
Output: dataset_frustum_pn2/{train,val}/{point_clouds,labels}
"""
import json
import argparse
import numpy as np
from pathlib import Path
import random


IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2
N_POINTS = 4096
MIN_FRUSTUM_PTS = 8
SEED = 42


def airsim_to_cv(p):
    return [p[1], p[2], p[0]]


# Camera offset + pitch do front_center (settings.json)
CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])    # body frame NED
CAM_PITCH_RAD = np.radians(-15.0)


def airsim_body_to_cv_camera(p_body):
    """body NED (forward, right, down) → CV camera (right, down, forward).
    Accounts for the camera lever arm and the -15 deg pitch applied by the Unreal camera render.

    p_body: AirSim relative_pose.position direto da detection ou desnormalizado da pose
    """
    # 1. Subtract the camera offset (the camera sits at CAM_OFFSET_BODY in the body frame)
    p = np.array(p_body, dtype=np.float32) - CAM_OFFSET_BODY
    # 2. Apply the body -> camera rotation (R_y(-pitch) = R_y(+15 deg))
    a = -CAM_PITCH_RAD  # = +15°
    c, s = np.cos(a), np.sin(a)
    # body NED FRD: x=fwd, y=right, z=down. The tilted camera has the same axes, rotated by pitch
    # R_y rotation: new_x = c*x + s*z; new_z = -s*x + c*z; new_y = y
    x_cam_fwd = c * p[0] + s * p[2]
    y_cam_right = p[1]
    z_cam_down = -s * p[0] + c * p[2]
    # 3. Reorder into CV (right, down, forward)
    return [float(y_cam_right), float(z_cam_down), float(x_cam_fwd)]


def project_to_pixel(pc):
    z = pc[:, 2]
    valid = z > 0.1
    z_safe = np.where(valid, z, 1.0)
    u = pc[:, 0] * FX / z_safe + CX
    v = pc[:, 1] * FY / z_safe + CY
    return u, v, valid


BAND_M = 0.0


def depth_band(frustum, band_m=10.0, anchor_pct=5.0):
    """Keep only the points near the CLOSEST object (the drone), discarding the far background.
    Anchors on the depth (z = forward in the CV frame) at percentile anchor_pct and keeps [z0, z0+band]."""
    if len(frustum) < 5 or band_m <= 0:
        return frustum
    z = frustum[:, 2]
    z0 = np.percentile(z, anchor_pct)
    return frustum[z <= z0 + band_m]


def crop_frustum(pc, bbox, expand_pct=0.0, band_m=0.0):
    """Crop the frustum. If expand_pct > 0, enlarge the box. If band_m > 0, cut the background in depth.
    """
    x1, y1, x2, y2 = bbox
    if expand_pct > 0:
        w = x2 - x1; h = y2 - y1
        dx = w * expand_pct; dy = h * expand_pct
        x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
        x2 = min(IMAGE_W, x2 + dx); y2 = min(IMAGE_H, y2 + dy)
    u, v, valid = project_to_pixel(pc)
    inside = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    frustum = pc[inside]
    if band_m > 0:
        frustum = depth_band(frustum, band_m)
    return frustum


def expand_bbox(bbox, expand_pct=0.25):
    x1, y1, x2, y2 = bbox
    w = x2 - x1; h = y2 - y1
    dx = w * expand_pct; dy = h * expand_pct
    return (max(0, x1 - dx), max(0, y1 - dy),
            min(IMAGE_W, x2 + dx), min(IMAGE_H, y2 + dy))


def sample_to_fixed(pts, n=N_POINTS, rng=None):
    if rng is None:
        rng = np.random
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
    else:
        idx = rng.choice(len(pts), n, replace=True)
    return pts[idx]


import os as _os
NORM_MODE = _os.environ.get("NORM_MODE", "max")   # max | p95 | fixed
FIXED_SCALE = 8.0
def normalize_pc(pc_xyz):
    centroid = pc_xyz.mean(axis=0)
    pc_centered = pc_xyz - centroid
    dists = np.linalg.norm(pc_centered, axis=1)
    if NORM_MODE == "p95":
        scale = float(np.percentile(dists, 95))      # robust to an isolated far point
    elif NORM_MODE == "fixed":
        scale = FIXED_SCALE                            # fixed metric scale (as in the voxel pipeline)
    else:
        scale = float(dists.max())
    scale = max(scale, 1e-6)
    return pc_centered / scale, centroid, scale


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def yolo_to_bbox_px(line, W=IMAGE_W, H=IMAGE_H):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    _, xc, yc, bw, bh = parts
    xc, yc, bw, bh = float(xc) * W, float(yc) * H, float(bw) * W, float(bh) * H
    return (xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2)


def process_dataset(src_root, src_tag, out_root, yolo_subdir, rng):
    src = Path(src_root)
    pc_dir = src / "pointnet" / "point_clouds"
    lbl_dir = src / "pointnet" / "labels_3d"
    yolo_pred_dir = src / yolo_subdir
    gt_yolo_dir = src / "yolo" / "labels"

    n_pos = 0
    n_fp = 0
    n_neg = 0
    n_skipped_empty = 0

    for split in ("train", "val"):
        out_split = out_root / split
        (out_split / "point_clouds").mkdir(parents=True, exist_ok=True)
        (out_split / "labels").mkdir(parents=True, exist_ok=True)

        for gt_yolo_p in sorted((gt_yolo_dir / split).glob("*.txt")):
            frame = gt_yolo_p.stem
            pc_p = pc_dir / f"{frame}.npy"
            yolo_pred_p = yolo_pred_dir / f"{frame}.json"
            gt3d_p = lbl_dir / f"{frame}.json"

            if not pc_p.exists():
                continue
            pc = np.load(pc_p).astype(np.float32)
            if len(pc) < 100:
                continue

            gt_lines = [l for l in open(gt_yolo_p) if l.strip()]
            gt_bboxes_px = [yolo_to_bbox_px(l) for l in gt_lines]
            gt_bboxes_px = [b for b in gt_bboxes_px if b]
            api_labels = []
            if gt3d_p.exists():
                try:
                    api_labels = json.load(open(gt3d_p))
                except Exception:
                    pass

            yolo_preds = []
            if yolo_pred_p.exists():
                try:
                    yolo_preds = json.load(open(yolo_pred_p))
                except Exception:
                    pass

            # === POSITIVES / FALSE POSITIVES: one sample per YOLO detection ===
            for k, pred in enumerate(yolo_preds):
                ybb = tuple(pred["bbox_2d"])
                conf = pred["confidence"]

                # Enlarge the box by 25% to capture more context and the rotors sticking out
                frustum = crop_frustum(pc, ybb, expand_pct=0.25, band_m=BAND_M)
                if len(frustum) < MIN_FRUSTUM_PTS:
                    n_skipped_empty += 1
                    continue

                # Match against the ground truth
                best_gt = None
                best_iou = 0.0
                for gt_bb, api_l in zip(gt_bboxes_px, api_labels):
                    iou = bbox_iou(ybb, gt_bb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = api_l

                # Sample fixo + normaliza
                sampled = sample_to_fixed(frustum, N_POINTS, rng)
                pc_xyz_norm, centroid, scale = normalize_pc(sampled)

                if best_iou >= 0.3 and best_gt:
                # labels_3d v6 are ALREADY in the CV camera frame (center and box3D in CV).
                # Do not re-transform -- use them directly.
                    center_cv = np.array(best_gt["center"], dtype=np.float64)
                    center_rel = ((center_cv - centroid) / scale).tolist()
                    if "box3D_min" in best_gt:
                        bmin = np.array(best_gt["box3D_min"], dtype=np.float64)
                        bmax = np.array(best_gt["box3D_max"], dtype=np.float64)
                        size = [abs(bmax[ii] - bmin[ii]) for ii in range(3)]
                    else:
                        size = [1.0, 0.3, 1.0]
                    size_norm = [s / scale for s in size]
                    is_drone = 1
                    n_pos += 1
                else:
                    center_rel = [0.0, 0.0, 0.0]
                    size_norm = [0.0, 0.0, 0.0]
                    is_drone = 0
                    n_fp += 1

                sample_id = f"{src_tag}_{frame}_det{k}"
                np.save(out_split / "point_clouds" / f"{sample_id}.npy",
                        pc_xyz_norm.astype(np.float32))
                with open(out_split / "labels" / f"{sample_id}.json", "w") as f:
                    json.dump({
                        "is_drone": is_drone,
                        "center_rel_normalized": list(map(float, center_rel)),
                        "size_normalized": list(map(float, size_norm)),
                        "yolo_bbox": list(ybb),
                        "yolo_confidence": float(conf),
                        "best_gt_iou": float(best_iou),
                        "scale": float(scale),
                        "centroid_cv": centroid.tolist(),
                        "n_frustum_pts": int(len(frustum)),
                    }, f)

            # === Random NEGATIVES (no drone) ===
            # Every ground-truth/YOLO box is enlarged by 50% and negatives with any overlap are rejected.
            # Ground-truth boxes and YOLO boxes together (all possible drone regions) form the exclusion zones
            exclusion_bboxes = list(gt_bboxes_px)
            for pred in yolo_preds:
                exclusion_bboxes.append(tuple(pred["bbox_2d"]))
                # Enlarge each exclusion zone by 50% for slack
            exclusion_expanded = [expand_bbox(b, 0.50) for b in exclusion_bboxes]

            n_negatives_to_add = max(2, len(yolo_preds))
            tries = 0
            added = 0
            while added < n_negatives_to_add and tries < n_negatives_to_add * 10:
                tries += 1
                if rng.uniform(0, 1) < 0.5:
                    w = rng.randint(10, 40)
                    h = rng.randint(5, 30)
                else:
                    w = rng.randint(40, 200)
                    h = rng.randint(20, 100)
                x1 = rng.randint(0, IMAGE_W - w)
                y1 = rng.randint(0, IMAGE_H - h)
                neg_bbox = (x1, y1, x1 + w, y1 + h)
                # Reject on any overlap with an enlarged exclusion zone
                overlap_any = False
                for eb in exclusion_expanded:
                    if bbox_iou(neg_bbox, eb) > 0.0:
                        overlap_any = True; break
                if overlap_any:
                    continue
                frustum = crop_frustum(pc, neg_bbox, band_m=BAND_M)
                if len(frustum) < MIN_FRUSTUM_PTS:
                    continue
                sampled = sample_to_fixed(frustum, N_POINTS, rng)
                pc_xyz_norm, centroid, scale = normalize_pc(sampled)
                sample_id = f"{src_tag}_{frame}_neg{added}"
                np.save(out_split / "point_clouds" / f"{sample_id}.npy",
                        pc_xyz_norm.astype(np.float32))
                with open(out_split / "labels" / f"{sample_id}.json", "w") as f:
                    json.dump({
                        "is_drone": 0,
                        "center_rel_normalized": [0.0, 0.0, 0.0],
                        "size_normalized": [0.0, 0.0, 0.0],
                        "yolo_bbox": list(neg_bbox),
                        "yolo_confidence": 0.0,
                        "best_gt_iou": 0.0,
                        "scale": float(scale),
                        "centroid_cv": centroid.tolist(),
                        "n_frustum_pts": int(len(frustum)),
                    }, f)
                n_neg += 1
                added += 1

    return n_pos, n_fp, n_neg, n_skipped_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="dataset_frustum_pn2")
    ap.add_argument("--yolo_subdir", default="yolo_predictions_c10")
    ap.add_argument("--sources", nargs="+", default=[
        "dataset_nh_v4:nh", "dataset_city_v4:city", "dataset_coast_v4:coast",
    ], help="<path>:<tag> pares")
    ap.add_argument("--band_m", type=float, default=0.0, help="depth band (m): cuts the far background out of the frustum")
    args = ap.parse_args()
    global BAND_M; BAND_M = args.band_m
    print(f"depth_band = {BAND_M}m")

    out_root = Path(args.output)
    sources = [tuple(s.split(":")) for s in args.sources]

    total = {'pos': 0, 'fp': 0, 'neg': 0, 'skip': 0}
    for src_path, tag in sources:
        if not Path(src_path).exists():
            print(f"SKIP: {src_path} does not exist")
            continue
        rng = np.random.RandomState(SEED + hash(tag) % 1000)
        random.seed(SEED + hash(tag) % 1000)
        p, fp, n, sk = process_dataset(src_path, tag, out_root, args.yolo_subdir, rng)
        print(f"{src_path}: POS={p}  FP={fp}  NEG={n}  SKIP_EMPTY={sk}")
        total['pos'] += p
        total['fp'] += fp
        total['neg'] += n
        total['skip'] += sk

    grand = sum([total['pos'], total['fp'], total['neg']])
    print(f"\n{'='*50}")
    print(f"TOTAL: {grand} samples (+ {total['skip']} skipped empty frustums)")
    print(f"  Positives (drone matched GT): {total['pos']} ({total['pos']/max(grand,1)*100:.1f}%)")
    print(f"  False Positives (YOLO err):   {total['fp']} ({total['fp']/max(grand,1)*100:.1f}%)")
    print(f"  Negatives (random crops):     {total['neg']} ({total['neg']/max(grand,1)*100:.1f}%)")
    print(f"Output: {out_root.absolute()}")


if __name__ == "__main__":
    main()
