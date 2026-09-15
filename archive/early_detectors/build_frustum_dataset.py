#!/usr/bin/env python3
"""ARCHIVED. First frustum dataset builder.

For each (frame, 2D box), projects the point-cloud points into image space and
keeps ONLY those inside the box. Each box becomes one frustum sample, so a frame
with three boxes yields three samples.

Frame convention: the cloud is in the CV camera frame (X=right, Y=down,
Z=forward); the projection is u = X*FX/Z + CX, v = Y*FY/Z + CY, the same as in
depth_to_pointcloud.

Usage:
    python3 build_frustum_dataset.py <dataset_dir> [--output <dir>] [--viz N]

Output:
    <output>/point_clouds/<src>_frame_XXXXXX_obj<i>.npy   (Nx3)
    <output>/labels/<src>_frame_XXXXXX_obj<i>.json
    <output>/viz/<src>_frame_XXXXXX_obj<i>.jpg            (optional)

Superseded by dataset_generation/build/frustum_dataset.py, which adds N=4096,
the 25% box expansion and the NORM_MODE switch.
"""
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


# Camera intrinsics — MESMOS do generate_dataset_urban.py
IMAGE_W, IMAGE_H = 1280, 720
FOV_H_DEG = 90
FX = IMAGE_W / (2 * np.tan(np.radians(FOV_H_DEG / 2)))
FY = FX  # square pixels (as in the generator)
CX, CY = IMAGE_W / 2, IMAGE_H / 2


def airsim_to_cv(p):
    """AirSim NED (X=fwd, Y=right, Z=down) → CV (X=right, Y=down, Z=fwd)"""
    return [p[1], p[2], p[0]]


def crop_frustum(pc, bbox_2d, margin_px=2):
    """
    Keep the cloud points (CV frame) that project inside the 2D box.
    Retorna subset Nx3.
    """
    x1, y1, x2, y2 = bbox_2d
    z = pc[:, 2]
    valid_z = z > 0.1
    z_safe = np.where(valid_z, z, 1.0)
    u = pc[:, 0] * FX / z_safe + CX
    v = pc[:, 1] * FY / z_safe + CY
    inside = (
        valid_z &
        (u >= x1 - margin_px) & (u <= x2 + margin_px) &
        (v >= y1 - margin_px) & (v <= y2 + margin_px)
    )
    return pc[inside]


def render_viz(image, bbox_2d, label, n_points, dist):
    """The original image with the box highlighted, plus frustum information."""
    out = image.copy()
    x1, y1, x2, y2 = bbox_2d
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Highlight overlay (semi-transparent)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
    txt = f"{label}  n_pts={n_points}  dist={dist:.1f}m"
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    return out


def yolo_to_bbox_px(line, W=IMAGE_W, H=IMAGE_H):
    parts = line.strip().split()
    if len(parts) != 5: return None
    _, xc, yc, bw, bh = parts
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(xc - bw/2), int(yc - bh/2), int(xc + bw/2), int(yc + bh/2)


def process_dataset(root, out_dir, source_tag, viz_limit=20):
    """Process one dataset (city/nh/coast) and append it to the output."""
    root = Path(root)
    pc_dir = root / "pointnet" / "point_clouds"
    json_dir = root / "pointnet" / "labels_3d"
    img_dir = root / "yolo" / "images"
    lbl_dir = root / "yolo" / "labels"

    n_frames = 0
    n_samples = 0
    n_empty_crops = 0
    viz_count = 0

    # itera train e val
    for split in ("train", "val"):
        for img_path in sorted((img_dir / split).glob("*.jpg")):
            frame = img_path.stem
            yolo_path = lbl_dir / split / f"{frame}.txt"
            pc_path = pc_dir / f"{frame}.npy"
            json_path = json_dir / f"{frame}.json"

            if not (yolo_path.exists() and pc_path.exists()):
                continue
            pc = np.load(pc_path).astype(np.float32)
            yolo_lines = [l for l in open(yolo_path) if l.strip()]
            if not yolo_lines:
                continue
            labels_3d = []
            if json_path.exists():
                try:
                    labels_3d = json.load(open(json_path))
                except Exception:
                    pass

            # Load the image only if a visualisation is being produced
            image = None

            for i, line in enumerate(yolo_lines):
                bbox = yolo_to_bbox_px(line)
                if bbox is None: continue
                frustum_pts = crop_frustum(pc, bbox)
                if len(frustum_pts) < 5:
                    n_empty_crops += 1
                    continue

                # Find the matching 3D label (matched by box)
                matched = None
                for d in labels_3d:
                    bb = d.get("bbox_2d")
                    if not bb: continue
                    if abs(bb[0] - bbox[0]) <= 3 and abs(bb[1] - bbox[1]) <= 3:
                        matched = d
                        break

                # Build the frustum label
                label = {
                    "source": source_tag,
                    "frame": frame,
                    "split": split,
                    "bbox_2d": list(bbox),
                    "n_points": int(len(frustum_pts)),
                }
                if matched:
                    # Convert the AirSim center/box3D to the CV frame (aligned with the cloud)
                    if "center" in matched:
                        label["center_cv"] = airsim_to_cv(matched["center"])
                    if "box3D_min" in matched:
                        bmin_cv = airsim_to_cv(matched["box3D_min"])
                        bmax_cv = airsim_to_cv(matched["box3D_max"])
                        label["box3D_min_cv"] = [min(bmin_cv[k], bmax_cv[k]) for k in range(3)]
                        label["box3D_max_cv"] = [max(bmin_cv[k], bmax_cv[k]) for k in range(3)]
                    label["distance_m"] = matched.get("distance_m")
                    label["drone_name"] = matched.get("name", "drone")

                # Save
                sample_id = f"{source_tag}_{frame}_obj{i}"
                np.save(out_dir / "point_clouds" / f"{sample_id}.npy",
                        frustum_pts.astype(np.float32))
                with open(out_dir / "labels" / f"{sample_id}.json", "w") as f:
                    json.dump(label, f, indent=2)
                n_samples += 1

                # Viz (limitado)
                if viz_count < viz_limit:
                    if image is None:
                        image = cv2.imread(str(img_path))
                    if image is not None:
                        dist = label.get("distance_m") or 0.0
                        name = label.get("drone_name", "drone")
                        viz = render_viz(image, bbox, name, len(frustum_pts), dist)
                        cv2.imwrite(str(out_dir / "viz" / f"{sample_id}.jpg"), viz)
                        viz_count += 1

            n_frames += 1

    return n_frames, n_samples, n_empty_crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="dataset_frustum",
                    help="output directory (default: dataset_frustum)")
    ap.add_argument("--inputs", nargs="+", default=[
        "dataset_urban_city", "dataset_urban_nh", "dataset_urban_coast"],
        help="Datasets de origem (defaults: city + nh + coast)")
    ap.add_argument("--viz", type=int, default=20,
                    help="how many visualisations to save PER dataset (default: 20)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    (out_dir / "point_clouds").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_samples = 0
    total_empty = 0
    for ds_path in args.inputs:
        tag = Path(ds_path).name.replace("dataset_urban_", "")
        print(f"\n=== Processando {ds_path} (tag={tag}) ===")
        n_f, n_s, n_e = process_dataset(ds_path, out_dir, tag, viz_limit=args.viz)
        print(f"  frames processados: {n_f}")
        print(f"  frustum samples:    {n_s}")
        print(f"  crops vazios (skip): {n_e}")
        total_frames += n_f
        total_samples += n_s
        total_empty += n_e

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_samples} frustum samples de {total_frames} frames")
    print(f"  ({total_empty} skipped because the crop was empty)")
    print(f"  Output: {out_dir.absolute()}")


if __name__ == "__main__":
    main()
