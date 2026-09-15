#!/usr/bin/env python3
"""
Build the painted (PointPainting-style) dataset for PointNet:
  Per-bbox YOLO sample. Input: (N=4096, 4) = xyz + prob (YOLO conf inside bbox, 0 outside)
  Output label: GT 3D bbox (do label JSON do dataset original)

  - one YOLO detection -> one painted sample
  - match against the ground truth by 2D IoU >= 0.3 to assign the regression label
  - if YOLO fired without a ground-truth match: sample with cls=0 (false positive)
  - extra negatives: frames with no YOLO detection -> all zeros (cls=0)
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
SEED = 42


def airsim_to_cv(p):
    return [p[1], p[2], p[0]]


CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = np.radians(-15.0)


def airsim_body_to_cv_camera(p_body):
    """Body NED -> CV camera (right, down, forward), applying the pitch and the lever arm."""
    p = np.array(p_body, dtype=np.float32) - CAM_OFFSET_BODY
    a = -CAM_PITCH_RAD
    c, s = np.cos(a), np.sin(a)
    x_cam_fwd = c * p[0] + s * p[2]
    y_cam_right = p[1]
    z_cam_down = -s * p[0] + c * p[2]
    return [float(y_cam_right), float(z_cam_down), float(x_cam_fwd)]


def project_to_pixel(pc):
    z = pc[:, 2]
    valid = z > 0.1
    z_safe = np.where(valid, z, 1.0)
    u = pc[:, 0] * FX / z_safe + CX
    v = pc[:, 1] * FY / z_safe + CY
    return u, v, valid


def sample_to_fixed(pts, n=N_POINTS, rng=None):
    if rng is None:
        rng = np.random
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
    else:
        idx = rng.choice(len(pts), n, replace=True)
    return pts[idx]


def normalize_pc(pc_xyz):
    centroid = pc_xyz.mean(axis=0)
    pc_centered = pc_xyz - centroid
    scale = np.linalg.norm(pc_centered, axis=1).max()
    scale = max(scale, 1e-6)
    return pc_centered / scale, centroid, scale


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def yolo_to_bbox_px(line, W=IMAGE_W, H=IMAGE_H):
    parts = line.strip().split()
    if len(parts) != 5: return None
    _, xc, yc, bw, bh = parts
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return (xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2)


def paint_pc(pc, yolo_bbox, confidence):
    """
    For every point, project it to a pixel.
    Inside the box  -> prob = confidence
    Outside the box -> prob = 0
    Returns (N, 4) = xyz + prob.
    """
    x1, y1, x2, y2 = yolo_bbox
    u, v, valid_z = project_to_pixel(pc)
    inside = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    prob = np.where(inside, confidence, 0.0).astype(np.float32)
    painted = np.concatenate([pc, prob[:, None]], axis=1).astype(np.float32)
    return painted


def process_dataset(src_root, src_tag, out_root, rng, yolo_subdir="yolo_predictions"):
    src = Path(src_root)
    pc_dir = src / "pointnet" / "point_clouds"
    lbl_dir = src / "pointnet" / "labels_3d"
    yolo_pred_dir = src / yolo_subdir
    gt_yolo_dir = src / "yolo" / "labels"

    n_pos = 0
    n_neg = 0
    n_fp = 0  # YOLO fired without a ground-truth match

    for split in ("train", "val"):
        out_split = out_root / split
        (out_split / "point_clouds").mkdir(parents=True, exist_ok=True)
        (out_split / "labels").mkdir(parents=True, exist_ok=True)

        for gt_yolo_p in sorted((gt_yolo_dir / split).glob("*.txt")):
            frame = gt_yolo_p.stem
            pc_p = pc_dir / f"{frame}.npy"
            yolo_pred_p = yolo_pred_dir / f"{frame}.json"
            json_p = lbl_dir / f"{frame}.json"

            if not pc_p.exists():
                continue
            pc = np.load(pc_p).astype(np.float32)
            if len(pc) < 100:
                continue

            # ground-truth boxes + 3D labels
            gt_lines = [l for l in open(gt_yolo_p) if l.strip()]
            gt_bboxes_px = [yolo_to_bbox_px(l) for l in gt_lines]
            gt_bboxes_px = [b for b in gt_bboxes_px if b]
            api_labels = []
            if json_p.exists():
                try:
                    api_labels = json.load(open(json_p))
                except: pass

            # YOLO predictions
            yolo_preds = []
            if yolo_pred_p.exists():
                try:
                    yolo_preds = json.load(open(yolo_pred_p))
                except: pass

            # === POSITIVES: one sample per YOLO detection ===
            for k, pred in enumerate(yolo_preds):
                ybb = tuple(pred["bbox_2d"])
                conf = pred["confidence"]

                # Match against the ground truth by IoU
                best_gt = None
                best_iou = 0
                for gt_bb, api_l in zip(gt_bboxes_px, api_labels):
                    iou = bbox_iou(ybb, gt_bb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = api_l

                # Paint the full point cloud with the probability channel
                painted = paint_pc(pc, ybb, conf)

                # Sample fixo N=4096
                sampled = sample_to_fixed(painted, N_POINTS, rng)

                # Normalise XYZ only (leave prob untouched)
                pc_xyz_norm, centroid, scale = normalize_pc(sampled[:, :3])
                sampled[:, :3] = pc_xyz_norm  # replace xyz with the normalised version

                # Label: regress if matched, otherwise mark as a false positive (cls=0)
                if best_iou >= 0.3 and best_gt:
                    # labels_3d v6 are ALREADY in the CV camera frame. Use them directly.
                    center_cv = np.array(best_gt["center"], dtype=np.float64)
                    center_rel = ((center_cv - centroid) / scale).tolist()
                    if "box3D_min" in best_gt:
                        bmin = np.array(best_gt["box3D_min"], dtype=np.float64)
                        bmax = np.array(best_gt["box3D_max"], dtype=np.float64)
                        size = [abs(bmax[k]-bmin[k]) for k in range(3)]
                    else:
                        size = [1.0, 0.3, 1.0]
                    size_norm = [s/scale for s in size]
                    is_drone = 1
                    n_pos += 1
                else:
                    center_rel = [0.0, 0.0, 0.0]
                    size_norm = [0.0, 0.0, 0.0]
                    is_drone = 0
                    n_fp += 1

                sample_id = f"{src_tag}_{frame}_det{k}"
                np.save(out_split / "point_clouds" / f"{sample_id}.npy", sampled)
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
                    }, f)

            # === AGGRESSIVE NEGATIVES: random crops where there is no drone ===
            # Two random negatives per YOLO detection added,
            # forcing a roughly 33%/67% positive/negative balance
            n_negatives_to_add = max(3, 2 * len(yolo_preds))
            for neg_i in range(n_negatives_to_add):
                # Random bbox 2D
                w = rng.randint(60, 300)
                h = rng.randint(30, 150)
                x1 = rng.randint(0, IMAGE_W - w)
                y1 = rng.randint(0, IMAGE_H - h)
                neg_bbox = (x1, y1, x1 + w, y1 + h)
                # Ensure no overlap with a ground-truth box (i.e. it really is not a drone)
                max_iou_gt = max((bbox_iou(neg_bbox, gb) for gb in gt_bboxes_px), default=0)
                if max_iou_gt > 0.1:
                    continue
                # Random confidence (mimics a plausible false positive)
                fake_conf = rng.uniform(0.3, 0.9)
                painted = paint_pc(pc, neg_bbox, fake_conf)
                sampled = sample_to_fixed(painted, N_POINTS, rng)
                pc_xyz_norm, centroid, scale = normalize_pc(sampled[:, :3])
                sampled[:, :3] = pc_xyz_norm

                sample_id = f"{src_tag}_{frame}_neg{neg_i}"
                np.save(out_split / "point_clouds" / f"{sample_id}.npy", sampled)
                with open(out_split / "labels" / f"{sample_id}.json", "w") as f:
                    json.dump({
                        "is_drone": 0,
                        "center_rel_normalized": [0.0, 0.0, 0.0],
                        "size_normalized": [0.0, 0.0, 0.0],
                        "yolo_bbox": list(neg_bbox),
                        "yolo_confidence": float(fake_conf),
                        "best_gt_iou": 0.0,
                        "scale": float(scale),
                        "centroid_cv": centroid.tolist(),
                    }, f)
                n_neg += 1

    return n_pos, n_fp, n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="dataset_painted")
    ap.add_argument("--yolo_subdir", default="yolo_predictions")
    ap.add_argument("--sources", nargs="+", default=[
        "dataset_nh_v4:nh", "dataset_city_v4:city", "dataset_coast_v4:coast",
    ], help="<path>:<tag> pares")
    args = ap.parse_args()
    out_root = Path(args.output)
    sources = [tuple(s.split(":")) for s in args.sources]

    total_pos = 0
    total_fp = 0
    total_neg = 0
    for src_path, tag in sources:
        if not Path(src_path).exists():
            print(f"SKIP: {src_path} does not exist")
            continue
        rng = np.random.RandomState(SEED + hash(tag) % 1000)
        random.seed(SEED + hash(tag) % 1000)
        n_p, n_fp, n_n = process_dataset(src_path, tag, out_root, rng, args.yolo_subdir)
        print(f"{src_path}: POS={n_p}  FP={n_fp}  NEG={n_n}")
        total_pos += n_p
        total_fp += n_fp
        total_neg += n_n

    total = total_pos + total_fp + total_neg
    print(f"\n{'='*50}")
    print(f"TOTAL: {total} samples")
    print(f"  Positives (drone w/ GT): {total_pos} ({total_pos/total*100:.1f}%)")
    print(f"  False Positives (YOLO err): {total_fp} ({total_fp/total*100:.1f}%)")
    print(f"  Negatives (no detection): {total_neg} ({total_neg/total*100:.1f}%)")
    print(f"Output: {out_root.absolute()}")


if __name__ == "__main__":
    main()
