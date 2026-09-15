#!/usr/bin/env python3
"""ARCHIVED. Step 4, faster variant of the mask-to-box step.

Part of route (i) of Section 7.2.3.
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import time

def yolo_box(xmin, ymin, xmax, ymax, W, H):
    """Convert bbox to YOLO format (normalized cx, cy, w, h)"""
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return cx, cy, w, h

def process_image_fast(seg_path, color_to_cid, min_area=50):
    """Process a single image efficiently"""
    seg = cv2.imread(str(seg_path))
    if seg is None:
        return None, []

    H, W = seg.shape[:2]
    detections = []

    # Process each known color
    for (b, g, r), cid in color_to_cid.items():
        # Create mask for this color (vectorized operation)
        mask = (seg[:,:,0] == b) & (seg[:,:,1] == g) & (seg[:,:,2] == r)

        if not mask.any():
            continue

        # Convert to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < min_area:
                continue

            detections.append({
                'cid': cid,
                'bbox': (x, y, w, h),
                'bbox_norm': yolo_box(x, y, x+w, y+h, W, H)
            })

    return (W, H), detections

def main():
    ap = argparse.ArgumentParser(description="Fast YOLO and COCO label generation")
    ap.add_argument("--dataset", default="dataset", help="Dataset directory")
    ap.add_argument("--cameras", nargs="+", default=["front_center"],
                    help="Camera names to process")
    ap.add_argument("--seg_map", default="seg_color_map.json", help="Color map file")
    ap.add_argument("--min_area", type=int, default=50, help="Minimum area for bounding box")
    ap.add_argument("--format", choices=["yolo", "coco", "both"], default="both",
                    help="Output format(s)")
    ap.add_argument("--max_frames", type=int, default=None, help="Process only N frames (for testing)")
    args = ap.parse_args()

    ds = Path(args.dataset)

    if not Path(args.seg_map).exists():
        print(f"Error: Color map file '{args.seg_map}' not found!")
        print("Please run: python3 03_map_seg_colors_auto.py --seg_dir dataset/seg/front_center")
        return

    with open(args.seg_map) as f:
        color_map = json.load(f)

    # Get class names from color map
    classes = list(set(color_map.values()))
    classes.sort()  # Ensure consistent ordering

    print(f"Classes found: {classes}")

    # Build color to class ID mapping
    color_to_cid = {}
    for bgr_str, cls in color_map.items():
        b, g, r = map(int, bgr_str.split(","))
        cid = classes.index(cls)
        color_to_cid[(b, g, r)] = cid
        print(f"  BGR({b},{g},{r}) -> {cls} (id={cid})")

    # Initialize COCO format
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }
    ann_id = 1
    img_id = 1

    total_start = time.time()

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
        if args.max_frames:
            seg_files = seg_files[:args.max_frames]

        if not seg_files:
            print(f"No segmentation files found in {seg_dir}")
            continue

        print(f"\n Processing camera: {cam} ({len(seg_files)} frames)")

        cam_start = time.time()
        detection_stats = {cls: 0 for cls in classes}

        for seg_path in tqdm(seg_files, desc=f"{cam}"):
            base_name = seg_path.stem

            # Process image
            dims, detections = process_image_fast(seg_path, color_to_cid, args.min_area)

            if dims is None:
                continue

            W, H = dims

            # Add to COCO if needed
            if args.format in ["coco", "both"]:
                coco["images"].append({
                    "id": img_id,
                    "file_name": f"{cam}/{seg_path.name}",
                    "width": W,
                    "height": H
                })

            # Write YOLO labels
            if args.format in ["yolo", "both"] and detections:
                label_file = lab_dir / f"{base_name}.txt"
                with open(label_file, "w") as f:
                    for det in detections:
                        cx, cy, w, h = det['bbox_norm']
                        f.write(f"{det['cid']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                        if det['cid'] < len(classes):
                            detection_stats[classes[det['cid']]] += 1

            # Add COCO annotations
            if args.format in ["coco", "both"]:
                for det in detections:
                    x, y, w, h = det['bbox']
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": det['cid'],
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": int(w * h),
                        "iscrowd": 0
                    })
                    ann_id += 1

            img_id += 1

        cam_time = time.time() - cam_start
        fps = len(seg_files) / cam_time

        print(f"⏱  Processed in {cam_time:.1f}s ({fps:.1f} frames/sec)")
        print(f"Detections by class:")
        for cls in classes:
            print(f"     {cls}: {detection_stats[cls]}")

    # Save COCO annotations
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

    total_time = time.time() - total_start
    print(f"\n⏱  Total processing time: {total_time:.1f}s")

if __name__ == "__main__":
    main()