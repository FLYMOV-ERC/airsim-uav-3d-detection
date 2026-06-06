#!/usr/bin/env python3
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

def find_unique_colors(img, min_pixels=100):
    """Find unique colors in image that cover at least min_pixels"""
    # Reshape to list of pixels
    pixels = img.reshape(-1, 3)

    # Convert to tuple for counting
    pixel_tuples = [tuple(p) for p in pixels]

    # Count occurrences
    color_counts = Counter(pixel_tuples)

    # Filter by minimum pixel count and exclude black (background)
    significant_colors = []
    for color, count in color_counts.items():
        if count >= min_pixels and color != (0, 0, 0):
            significant_colors.append((color, count))

    # Sort by count (most common first)
    significant_colors.sort(key=lambda x: x[1], reverse=True)

    return significant_colors

def auto_detect_classes(seg_dir, num_samples=5):
    """Automatically detect object classes from segmentation images"""
    seg_dir = Path(seg_dir)
    imgs = sorted(seg_dir.glob("*.png"))[:num_samples]

    if not imgs:
        return None

    all_colors = []

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        colors = find_unique_colors(img, min_pixels=50)
        all_colors.extend([c[0] for c in colors[:10]])  # Top 10 colors per image

    # Find most common colors across all samples
    color_freq = Counter(all_colors)

    # Get most frequent colors (excluding black background)
    top_colors = []
    for color, freq in color_freq.most_common(10):  # Get more colors in case some are background
        if color != (0, 0, 0):  # Exclude black background
            top_colors.append(color)

    return top_colors

def main():
    ap = argparse.ArgumentParser(description="Auto-map segmentation colors to classes")
    ap.add_argument("--seg_dir", required=True, help="Path to segmentation directory")
    ap.add_argument("--out", default="seg_color_map.json", help="Output color map file")
    ap.add_argument("--classes", nargs="+", default=["DRONE"], help="Class names (in order of detection)")
    args = ap.parse_args()

    seg_dir = Path(args.seg_dir)

    print("🔍 Auto-detecting segmentation colors...")
    colors = auto_detect_classes(seg_dir)

    if not colors:
        print("❌ Error: Could not detect colors")
        return

    print(f"✅ Found {len(colors)} unique object colors")

    # Create color map
    color_map = {}

    # Assign colors to classes based on detection order
    num_classes_to_map = min(len(colors), len(args.classes))

    for i in range(num_classes_to_map):
        b, g, r = colors[i]
        color_key = f"{b},{g},{r}"
        color_map[color_key] = args.classes[i]
        print(f"  {args.classes[i]} -> BGR({b},{g},{r})")

    # Save color map
    with open(args.out, "w") as f:
        json.dump(color_map, f, indent=2)

    print(f"\n✅ Color map saved to: {args.out}")

    # Verify on first image
    print("\n📊 Verifying on first image...")
    imgs = sorted(seg_dir.glob("*.png"))
    if imgs:
        img = cv2.imread(str(imgs[0]))
        h, w = img.shape[:2]

        for color_key, cls in color_map.items():
            b, g, r = map(int, color_key.split(","))
            mask = np.all(img == [b, g, r], axis=2)
            pixel_count = np.sum(mask)
            percentage = (pixel_count / (h * w)) * 100
            if pixel_count > 0:
                print(f"  {cls}: {pixel_count} pixels ({percentage:.2f}%)")

if __name__ == "__main__":
    main()