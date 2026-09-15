#!/usr/bin/env python3
"""ARCHIVED. Step 4 of the segmentation-labeling chain: derive 2D boxes from the masks.

This is the step Section 7.2.3(i) reports produced too many false positives to be
usable.
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

CLASSES = ["DRONE", "BIRD", "PLANE"]

def yolo_box(xmin, ymin, xmax, ymax, W, H):
    """Convert bbox to YOLO format (normalized cx, cy, w, h)"""
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return cx, cy, w, h

def main():
    ap = argparse.ArgumentParser(description="Generate YOLO and COCO labels from segmentation masks")
    ap.add_argument("--dataset", default="dataset", help="Dataset directory")
    ap.add_argument("--cameras", nargs="+", default=["front_center", "bottom_center", "back_center"],
                    help="Camera names to process")
    ap.add_argument("--seg_map", default="seg_color_map.json", help="Color map file")
    ap.add_argument("--min_area", type=int, default=50, help="Minimum area for bounding box")
    ap.add_argument("--format", choices=["yolo", "coco", "both"], default="both",
                    help="Output format(s)")
    args = ap.parse_args()

    ds = Path(args.dataset)

    if not Path(args.seg_map).exists():
        print(f"Error: Color map file '{args.seg_map}' not found!")
        print("Please run 03_map_seg_colors.py first to create the color mapping.")
        return

    with open(args.seg_map) as f:
        color_map = json.load(f)

    color_to_cid = {}
    for bgr_str, cls in color_map.items():
        b, g, r = map(int, bgr_str.split(","))
        if cls in CLASSES:
            cid = CLASSES.index(cls)
            color_to_cid[(b, g, r)] = cid
        else:
            print(f"Warning: Class '{cls}' in color map not found in CLASSES list")

    if not color_to_cid:
        print("Error: No valid color mappings found!")
        return

    print(f"Color mappings loaded: {len(color_to_cid)} colors")
    for (b, g, r), cid in color_to_cid.items():
        print(f"  BGR({b},{g},{r}) -> {CLASSES[cid]} (id={cid})")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    }
    ann_id = 1
    img_id = 1

    for cam in args.cameras:
        seg_dir = ds / "seg" / cam
        img_dir = ds / "images" / cam
        lab_dir = ds / "labels" / cam

        if not seg_dir.exists():
            print(f"Skipping {cam}: segmentation directory not found")
            continue

        if args.format in ["yolo", "both"]:
            lab_dir.mkdir(parents=True, exist_ok=True)

        seg_files = sorted(seg_dir.glob("*.png"))
        if not seg_files:
            print(f"No segmentation files found in {seg_dir}")
            continue

        print(f"\n Processing camera: {cam} ({len(seg_files)} frames)")

        for seg_path in tqdm(seg_files, desc=f"Generating labels for {cam}"):
            seg = cv2.imread(str(seg_path))
            if seg is None:
                print(f"Warning: Could not read {seg_path}")
                continue

            H, W = seg.shape[:2]
            base_name = seg_path.stem

            if args.format in ["coco", "both"]:
                coco["images"].append({
                    "id": img_id,
                    "file_name": f"{cam}/{seg_path.name}",
                    "width": W,
                    "height": H
                })

            yolo_labels = []

            unique_colors = np.unique(seg.reshape(-1, 3), axis=0)

            for color in unique_colors:
                b, g, r = int(color[0]), int(color[1]), int(color[2])

                if (b, g, r) not in color_to_cid:
                    continue

                cid = color_to_cid[(b, g, r)]

                mask = np.all(seg == color, axis=2).astype(np.uint8) * 255

                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in cnts:
                    x, y, w, h = cv2.boundingRect(cnt)

                    if w * h < args.min_area:
                        continue

                    if args.format in ["yolo", "both"]:
                        cx, cy, ww, hh = yolo_box(x, y, x+w, y+h, W, H)
                        yolo_labels.append(f"{cid} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

                    if args.format in ["coco", "both"]:
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cid,
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "area": int(w * h),
                            "iscrowd": 0
                        })
                        ann_id += 1

            if args.format in ["yolo", "both"] and yolo_labels:
                label_file = lab_dir / f"{base_name}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_labels) + "\n")

            img_id += 1

    if args.format in ["coco", "both"]:
        coco_file = ds / "annotations_coco.json"
        with open(coco_file, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"\n COCO annotations saved to: {coco_file}")
        print(f"   Total images: {len(coco['images'])}")
        print(f"   Total annotations: {len(coco['annotations'])}")

    if args.format in ["yolo", "both"]:
        for cam in args.cameras:
            lab_dir = ds / "labels" / cam
            if lab_dir.exists():
                num_labels = len(list(lab_dir.glob("*.txt")))
                if num_labels > 0:
                    print(f"\n YOLO labels for {cam}: {num_labels} files in {lab_dir}")

    print("\n Summary:")
    for i, cls in enumerate(CLASSES):
        if args.format in ["coco", "both"]:
            count = sum(1 for ann in coco["annotations"] if ann["category_id"] == i)
            print(f"   {cls}: {count} detections")

if __name__ == "__main__":
    main()