#!/usr/bin/env python3
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

def draw_bbox(img, x, y, w, h, label, color=(0, 255, 0)):
    """Draw bounding box on image"""
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Add label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(img, (x, y-25), (x+label_size[0]+5, y), color, -1)
    cv2.putText(img, label, (x+2, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def process_single_image(rgb_path, seg_path, color_map_path, output_dir):
    """Process a single image and generate visualization"""

    # Load images
    print(f"\n📸 Processing image: {rgb_path}")
    rgb = cv2.imread(str(rgb_path))
    seg = cv2.imread(str(seg_path))

    if rgb is None or seg is None:
        print("❌ Error loading images")
        return

    H, W = rgb.shape[:2]
    print(f"   Image size: {W}x{H}")

    # Load color map
    with open(color_map_path) as f:
        color_map = json.load(f)

    # Get classes
    classes = list(set(color_map.values()))
    classes.sort()

    # Build color to class mapping
    color_to_class = {}
    for bgr_str, cls in color_map.items():
        b, g, r = map(int, bgr_str.split(","))
        color_to_class[(b, g, r)] = cls

    print(f"\n🎨 Color mappings:")
    for (b, g, r), cls in color_to_class.items():
        print(f"   BGR({b},{g},{r}) -> {cls}")

    # Create visualizations
    vis_seg = seg.copy()
    vis_rgb = rgb.copy()
    vis_mask = np.zeros_like(rgb)

    # Stats
    total_detections = 0
    yolo_annotations = []

    print(f"\n🔍 Finding objects...")

    # Process each color
    for (b, g, r), cls in color_to_class.items():
        # Create mask for this color
        mask = (seg[:,:,0] == b) & (seg[:,:,1] == g) & (seg[:,:,2] == r)

        if not mask.any():
            print(f"   {cls}: No pixels found")
            continue

        # Count pixels
        pixel_count = np.sum(mask)
        percentage = (pixel_count / (H * W)) * 100
        print(f"   {cls}: {pixel_count} pixels ({percentage:.2f}%)")

        # Convert to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"   {cls}: Found {len(contours)} regions")

        # Color for this class
        class_color = (0, 255, 0) if cls == "DRONE" else (255, 0, 0)

        # Process each contour
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Skip small detections
            if area < 50:
                continue

            total_detections += 1

            # Draw on mask visualization
            vis_mask[mask] = class_color

            # Draw bounding box on RGB
            draw_bbox(vis_rgb, x, y, w, h, f"{cls}_{i}", class_color)

            # Draw contour on segmentation
            cv2.drawContours(vis_seg, [contour], -1, (0, 255, 255), 2)

            # YOLO format (normalized)
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H

            yolo_annotations.append({
                'class': cls,
                'cx': cx,
                'cy': cy,
                'w': nw,
                'h': nh,
                'pixel_bbox': [x, y, w, h],
                'area': area
            })

            print(f"      Detection {i}: bbox=[{x},{y},{w},{h}], area={area}")

    print(f"\n📊 Total detections: {total_detections}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualizations
    cv2.imwrite(str(output_dir / "original_rgb.png"), rgb)
    cv2.imwrite(str(output_dir / "original_seg.png"), seg)
    cv2.imwrite(str(output_dir / "annotated_rgb.png"), vis_rgb)
    cv2.imwrite(str(output_dir / "annotated_seg.png"), vis_seg)
    cv2.imwrite(str(output_dir / "mask_overlay.png"), vis_mask)

    # Create side-by-side comparison
    comparison = np.hstack([rgb, vis_rgb])
    cv2.imwrite(str(output_dir / "comparison.png"), comparison)

    # Save YOLO format annotations
    yolo_file = output_dir / "annotations.txt"
    with open(yolo_file, "w") as f:
        for ann in yolo_annotations:
            class_id = classes.index(ann['class'])
            f.write(f"{class_id} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

    # Save detailed JSON
    json_file = output_dir / "annotations.json"
    with open(json_file, "w") as f:
        json.dump({
            'image': str(rgb_path),
            'width': W,
            'height': H,
            'classes': classes,
            'detections': yolo_annotations
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   - original_rgb.png: Original RGB image")
    print(f"   - original_seg.png: Original segmentation")
    print(f"   - annotated_rgb.png: RGB with bounding boxes")
    print(f"   - annotated_seg.png: Segmentation with contours")
    print(f"   - mask_overlay.png: Extracted masks")
    print(f"   - comparison.png: Side-by-side comparison")
    print(f"   - annotations.txt: YOLO format labels")
    print(f"   - annotations.json: Detailed annotations")

    # Display if possible (may not work in WSL without X server)
    try:
        cv2.imshow("Comparison", cv2.resize(comparison, (1600, 450)))
        print(f"\n👁️  Press any key to close the preview window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print(f"\n💡 Tip: Can't display window in WSL. Check the saved files in {output_dir}/")

def main():
    ap = argparse.ArgumentParser(description="Test annotation on a single image")
    ap.add_argument("--frame", type=int, default=1, help="Frame number to test (1-900)")
    ap.add_argument("--camera", default="front_center", help="Camera to use")
    ap.add_argument("--dataset", default="dataset", help="Dataset directory")
    ap.add_argument("--seg_map", default="seg_color_map.json", help="Color map file")
    ap.add_argument("--output", default="test_annotation", help="Output directory")
    args = ap.parse_args()

    # Build paths
    frame_name = f"{args.frame:06d}.png"
    rgb_path = Path(args.dataset) / "images" / args.camera / frame_name
    seg_path = Path(args.dataset) / "seg" / args.camera / frame_name

    if not rgb_path.exists():
        print(f"❌ RGB image not found: {rgb_path}")
        return

    if not seg_path.exists():
        print(f"❌ Segmentation image not found: {seg_path}")
        return

    if not Path(args.seg_map).exists():
        print(f"❌ Color map not found: {args.seg_map}")
        print("Run: python3 03_map_seg_colors_auto.py --seg_dir dataset/seg/front_center --classes DRONE")
        return

    process_single_image(rgb_path, seg_path, args.seg_map, args.output)

if __name__ == "__main__":
    main()