#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

CLASSES = ["DRONE", "BIRD", "PLANE"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

def draw_yolo_label(img, line, class_colors):
    """Draw YOLO format bounding box on image"""
    parts = line.strip().split()
    if len(parts) != 5:
        return

    cid = int(parts[0])
    cx, cy, w, h = map(float, parts[1:])

    H, W = img.shape[:2]

    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)

    x1 = max(0, min(x1, W-1))
    y1 = max(0, min(y1, H-1))
    x2 = max(0, min(x2, W-1))
    y2 = max(0, min(y2, H-1))

    color = class_colors.get(cid, (0, 255, 0))
    class_name = CLASSES[cid] if cid < len(CLASSES) else f"CLASS_{cid}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label_bg_height = 25
    cv2.rectangle(img, (x1, max(0, y1-label_bg_height)), (x1+len(class_name)*12, y1), color, -1)
    cv2.putText(img, class_name, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    ap = argparse.ArgumentParser(description="Preview dataset with bounding boxes overlay")
    ap.add_argument("--dataset", default="dataset", help="Dataset directory")
    ap.add_argument("--cameras", nargs="+", default=["front_center"],
                    help="Camera names to preview")
    ap.add_argument("--out_dir", default="previews", help="Output directory for preview videos")
    ap.add_argument("--fps", type=int, default=10, help="Output video FPS")
    ap.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (for testing)")
    ap.add_argument("--display", action="store_true", help="Display preview while generating")
    args = ap.parse_args()

    ds = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_colors = {i: COLORS[i % len(COLORS)] for i in range(len(CLASSES))}

    for cam in args.cameras:
        img_dir = ds / "images" / cam
        lab_dir = ds / "labels" / cam

        if not img_dir.exists():
            print(f"⚠️  Skipping {cam}: image directory not found")
            continue

        imgs = sorted(img_dir.glob("*.png"))
        if not imgs:
            print(f"⚠️  No images found in {img_dir}")
            continue

        if args.max_frames:
            imgs = imgs[:args.max_frames]

        print(f"\n📸 Processing camera: {cam} ({len(imgs)} frames)")

        sample = cv2.imread(str(imgs[0]))
        if sample is None:
            print(f"Error: Could not read {imgs[0]}")
            continue

        H, W = sample.shape[:2]

        out_file = out_dir / f"{cam}_preview.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(out_file), fourcc, args.fps, (W, H))

        stats = {cls: 0 for cls in CLASSES}
        frames_with_detections = 0

        for img_path in tqdm(imgs, desc=f"Generating preview for {cam}"):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            overlay = img.copy()

            label_file = lab_dir / img_path.with_suffix(".txt").name
            if label_file.exists():
                has_detection = False
                with open(label_file, "r") as f:
                    for line in f:
                        if line.strip():
                            draw_yolo_label(overlay, line, class_colors)
                            cid = int(line.strip().split()[0])
                            if cid < len(CLASSES):
                                stats[CLASSES[cid]] += 1
                            has_detection = True
                if has_detection:
                    frames_with_detections += 1

            frame_name = img_path.stem
            cv2.putText(overlay, f"Frame: {frame_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Camera: {cam}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset = 90
            for cls, color in zip(CLASSES, COLORS):
                count = stats[cls]
                cv2.putText(overlay, f"{cls}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25

            video_writer.write(overlay)

            if args.display:
                cv2.imshow(f"Preview - {cam}", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Preview interrupted by user")
                    break

        video_writer.release()

        print(f"✅ Preview saved to: {out_file}")
        print(f"   Frames with detections: {frames_with_detections}/{len(imgs)}")
        print(f"   Detection counts:")
        for cls in CLASSES:
            print(f"     {cls}: {stats[cls]}")

    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()