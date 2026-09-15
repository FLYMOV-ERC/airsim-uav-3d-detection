#!/usr/bin/env python3
"""ARCHIVED. Step 3, final corrected variant of the colour-mapping step.

Part of route (i) of Section 7.2.3.
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

def find_object_colors(seg_dir, num_samples=10):
    """Find colors that represent objects (not background)"""
    seg_dir = Path(seg_dir)
    seg_files = sorted(seg_dir.glob("*.png"))[:num_samples]

    if not seg_files:
        return []

    all_colors = Counter()

    for seg_path in seg_files:
        seg = cv2.imread(str(seg_path))
        if seg is None:
            continue

        # Get unique colors in this image
        pixels = seg.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        h, w = seg.shape[:2]
        total_pixels = h * w

        for color, count in zip(unique_colors, counts):
            b, g, r = int(color[0]), int(color[1]), int(color[2])

            # Skip white (background) and black
            if (b, g, r) == (255, 255, 255) or (b, g, r) == (0, 0, 0):
                continue

            # Only count if it's a significant portion (more than 10 pixels)
            if count > 10:
                all_colors[(b, g, r)] += 1

    # Return colors sorted by frequency
    return [color for color, freq in all_colors.most_common()]

def main():
    ap = argparse.ArgumentParser(description="Correctly map segmentation colors to classes")
    ap.add_argument("--seg_dir", required=True, help="Path to segmentation directory")
    ap.add_argument("--out", default="seg_color_map.json", help="Output color map file")
    ap.add_argument("--classes", nargs="+", default=["DRONE"], help="Class names")
    ap.add_argument("--samples", type=int, default=20, help="Number of images to sample")
    args = ap.parse_args()

    seg_dir = Path(args.seg_dir)

    print("Detecting object colors (excluding white background)...")
    colors = find_object_colors(seg_dir, args.samples)

    if not colors:
        print("No object colors found (only background)")
        print("\n This might mean:")
        print("   1. The dataset only has background (no objects)")
        print("   2. All objects are white (same as background)")
        print("\n Let me check the first frame in detail...")

        # Check first frame
        first_frame = sorted(seg_dir.glob("*.png"))[0]
        seg = cv2.imread(str(first_frame))
        h, w = seg.shape[:2]

        pixels = seg.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        print(f"\nFrame: {first_frame.name}")
        print(f"Size: {w}x{h}")
        print(f"Unique colors: {len(unique_colors)}")

        for color, count in zip(unique_colors, counts):
            b, g, r = int(color[0]), int(color[1]), int(color[2])
            percentage = (count / (h * w)) * 100
            print(f"   BGR({b:3d},{g:3d},{r:3d}): {count:7d} pixels ({percentage:6.2f}%)")

        # If there are non-white colors, use them
        object_colors = []
        for color, count in zip(unique_colors, counts):
            b, g, r = int(color[0]), int(color[1]), int(color[2])
            if (b, g, r) != (255, 255, 255) and (b, g, r) != (0, 0, 0) and count > 50:
                object_colors.append((b, g, r))

        if object_colors:
            colors = object_colors
            print(f"\n Found {len(colors)} object colors in first frame")

    if colors:
        print(f"\n Object colors found: {len(colors)}")

        # Create color map
        color_map = {}
        num_to_map = min(len(colors), len(args.classes))

        for i in range(num_to_map):
            b, g, r = colors[i]
            color_key = f"{b},{g},{r}"
            color_map[color_key] = args.classes[i]
            print(f"   {args.classes[i]} -> BGR({b},{g},{r})")

        # Save color map
        with open(args.out, "w") as f:
            json.dump(color_map, f, indent=2)

        print(f"\n Color map saved to: {args.out}")

        # Verify
        print("\n Verifying on sample frames...")
        sample_files = sorted(seg_dir.glob("*.png"))[:3]
        for seg_path in sample_files:
            seg = cv2.imread(str(seg_path))
            h, w = seg.shape[:2]

            print(f"\nFrame {seg_path.stem}:")
            for color_key, cls in color_map.items():
                b, g, r = map(int, color_key.split(","))
                mask = np.all(seg == [b, g, r], axis=2)
                pixel_count = np.sum(mask)
                if pixel_count > 0:
                    percentage = (pixel_count / (h * w)) * 100
                    print(f"   {cls}: {pixel_count} pixels ({percentage:.2f}%)")
    else:
        print("\n  No object colors detected. Your segmentation might be all background.")
        print("   Please check your AirSim settings and ensure objects have different IDs.")

if __name__ == "__main__":
    main()