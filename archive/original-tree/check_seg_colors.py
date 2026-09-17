#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def check_segmentation_colors(seg_dir, num_samples=10):
    """Check what colors exist in segmentation images"""
    seg_dir = Path(seg_dir)
    seg_files = sorted(seg_dir.glob("*.png"))[:num_samples]

    print(f"📸 Checking {len(seg_files)} segmentation images...\n")

    for i, seg_path in enumerate(seg_files, 1):
        print(f"Frame {seg_path.stem}:")

        seg = cv2.imread(str(seg_path))
        if seg is None:
            print("   ❌ Could not load image")
            continue

        h, w = seg.shape[:2]

        # Get unique colors
        pixels = seg.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)

        print(f"   Image size: {w}x{h}")
        print(f"   Unique colors: {len(unique_colors)}")

        # Count pixels for each color
        for color in unique_colors:
            b, g, r = color
            mask = np.all(seg == color, axis=2)
            count = np.sum(mask)
            percentage = (count / (h * w)) * 100

            # Identify color
            if (b, g, r) == (0, 0, 0):
                color_name = "BLACK (background)"
            elif (b, g, r) == (255, 255, 255):
                color_name = "WHITE (might be background)"
            else:
                color_name = "OBJECT"

            print(f"      BGR({b:3d},{g:3d},{r:3d}) -> {color_name:20s} : {count:7d} pixels ({percentage:6.2f}%)")

        print()

def main():
    ap = argparse.ArgumentParser(description="Check segmentation colors")
    ap.add_argument("--seg_dir", default="dataset/seg/front_center", help="Segmentation directory")
    ap.add_argument("--samples", type=int, default=10, help="Number of samples to check")
    ap.add_argument("--camera", default=None, help="Camera name (overrides seg_dir)")
    args = ap.parse_args()

    if args.camera:
        seg_dir = Path("dataset/seg") / args.camera
    else:
        seg_dir = Path(args.seg_dir)

    if not seg_dir.exists():
        print(f"❌ Directory not found: {seg_dir}")
        return

    check_segmentation_colors(seg_dir, args.samples)

if __name__ == "__main__":
    main()