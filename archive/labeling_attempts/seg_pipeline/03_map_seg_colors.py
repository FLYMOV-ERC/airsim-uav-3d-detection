#!/usr/bin/env python3
"""ARCHIVED. Step 3 of the segmentation-labeling chain: map object IDs to stencil colours.

Part of route (i) of Section 7.2.3.
"""

import json
import argparse
import cv2
import numpy as np
from pathlib import Path

CLASSES = ["DRONE", "BIRD", "PLANE"]

def main():
    ap = argparse.ArgumentParser(description="Map segmentation colors to classes")
    ap.add_argument("--seg_dir", required=True, help="Path to segmentation directory (e.g., dataset/seg/front_center)")
    ap.add_argument("--out", default="seg_color_map.json", help="Output color map file")
    args = ap.parse_args()

    seg_dir = Path(args.seg_dir)
    imgs = sorted(seg_dir.glob("*.png"))

    if not imgs:
        print(f"Error: No images found in {seg_dir}")
        return

    print(f"Loading first image: {imgs[0]}")
    img = cv2.imread(str(imgs[0]))
    if img is None:
        print(f"Error: Could not load image {imgs[0]}")
        return

    disp = img.copy()
    color_map = {}
    selected_colors = []

    def on_mouse(event, x, y, flags, param):
        nonlocal color_map, disp, selected_colors
        if event == cv2.EVENT_LBUTTONDOWN:
            b, g, r = map(int, img[y, x])
            color_key = f"{b},{g},{r}"

            if color_key not in [c[0] for c in selected_colors]:
                current_class_idx = len(selected_colors)
                if current_class_idx < len(CLASSES):
                    cls = CLASSES[current_class_idx]
                    color_map[color_key] = cls
                    selected_colors.append((color_key, cls))
                    print(f"{cls} -> BGR({b},{g},{r})")

                    cv2.circle(disp, (x, y), 8, (0, 255, 0), 2)
                    cv2.putText(disp, cls, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Segmentation Color Mapping", disp)

                    if len(selected_colors) == len(CLASSES):
                        print("\n All classes mapped! Press any key to save and exit.")

    cv2.namedWindow("Segmentation Color Mapping")
    cv2.setMouseCallback("Segmentation Color Mapping", on_mouse)

    print(f"\n Click on a pixel of each class in this order: {', '.join(CLASSES)}")
    print("Press ESC to cancel, or any other key to save after mapping all classes.\n")

    while True:
        cv2.imshow("Segmentation Color Mapping", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Cancelled by user")
            cv2.destroyAllWindows()
            return
        elif key != 255 and len(selected_colors) == len(CLASSES):  # Any key when done
            break

    cv2.destroyAllWindows()

    with open(args.out, "w") as f:
        json.dump(color_map, f, indent=2)

    print(f"\n Color map saved to: {args.out}")
    print("Mapped colors:")
    for color_key, cls in selected_colors:
        print(f"  {cls}: BGR{tuple(map(int, color_key.split(',')))}")

if __name__ == "__main__":
    main()