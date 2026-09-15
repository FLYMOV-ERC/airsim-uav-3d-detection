#!/usr/bin/env python3
"""
Post-processor: render a visualisation for EVERY frame of a dataset
(image + YOLO labels + the 3D information from the JSON).

Usage:
    python visualization/render_dataset.py <dataset_dir> [--out-dir DIR] [--splits train val]

<dataset_dir> is a dataset root holding ``yolo/images/<split>``,
``yolo/labels/<split>`` and ``pointnet/labels_3d``.  The renders are written to
``<dataset_dir>/visualizations_all`` unless --out-dir says otherwise.
"""
import argparse
import json
import cv2
from pathlib import Path

IMAGE_W, IMAGE_H = 1280, 720


def render_one(img_path, yolo_path, json_path, env_label=""):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # Read the 3D labels (for the range and the name)
    info_3d = {}
    if json_path and json_path.exists():
        try:
            data = json.load(open(json_path))
            for d in data:
                bb = tuple(d["bbox_2d"])
        # Prefer distance_m (straight from the API) over center[2] (a relative Z)
                dist = d.get("distance_m")
                if dist is None:
                    c = d.get("center", [0, 0, 0])
                    dist = (c[0]**2 + c[1]**2 + c[2]**2) ** 0.5
                info_3d[bb] = (d.get("name", "?"), dist)
        except Exception:
            pass

    # Read the YOLO label (normalised)
    if not yolo_path.exists():
        return img
    with open(yolo_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = parts
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x_min = int((xc - bw/2) * w)
        y_min = int((yc - bh/2) * h)
        x_max = int((xc + bw/2) * w)
        y_max = int((yc + bh/2) * h)

        # Tenta achar info 3D correspondente
        name = "drone"
        depth_str = ""
        for bb_key, (n, z) in info_3d.items():
            if abs(bb_key[0] - x_min) <= 2 and abs(bb_key[1] - y_min) <= 2:
                name = n
                depth_str = f" {z:.1f}m"
                break

        # Desenha
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{name}{depth_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Black background so the text stays legible
        cv2.rectangle(img, (x_min, max(0, y_min - th - 6)),
                      (x_min + tw + 4, y_min), (0, 0, 0), -1)
        cv2.putText(img, label, (x_min + 2, max(th, y_min - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if env_label:
        cv2.putText(img, env_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return img


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Render image + YOLO box + 3D label overlays for every frame "
                    "of a dataset.")
    ap.add_argument("dataset_dir", type=Path,
                    help="dataset root containing yolo/ and pointnet/labels_3d/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: <dataset_dir>/visualizations_all)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="which yolo splits to render (default: train val)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.dataset_dir
    if not root.exists():
        raise SystemExit(f"directory does not exist: {root}")

    out_dir = args.out_dir or (root / "visualizations_all")
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for split in args.splits:
        img_dir = root / "yolo" / "images" / split
        lbl_dir = root / "yolo" / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            frame = img_path.stem
            yolo_path = lbl_dir / f"{frame}.txt"
            json_path = root / "pointnet" / "labels_3d" / f"{frame}.json"
            vis = render_one(img_path, yolo_path, json_path,
                             env_label=f"{root.name} [{split}]  {frame}")
            if vis is not None:
                cv2.imwrite(str(out_dir / f"{frame}_{split}.jpg"), vis)
                count += 1

    print(f"Rendered {count} visualisations into {out_dir}/")


if __name__ == "__main__":
    main()
