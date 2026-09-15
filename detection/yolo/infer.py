#!/usr/bin/env python3
"""
Run the trained YOLO11s over a dataset and save the per-frame predictions.

Output: <dataset>/yolo_predictions/<frame>.json holding a list of:
  [{"bbox_2d": [x1,y1,x2,y2], "confidence": 0.xx, "class": 0}, ...]
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


def process_dataset(model, dataset_root, conf_threshold=0.25, iou_threshold=0.5,
                    out_subdir="yolo_predictions"):
    root = Path(dataset_root)
    out_dir = root / out_subdir
    out_dir.mkdir(exist_ok=True)

    all_images = []
    for split in ("train", "val"):
        img_dir = root / "yolo" / "images" / split
        if img_dir.exists():
            all_images.extend(sorted(img_dir.glob("*.jpg")))

    n_with_det = 0
    n_total_det = 0
    print(f"\n=== {dataset_root}: {len(all_images)} images ===")

    # Batch inference (faster)
    batch_size = 32
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i+batch_size]
        results = model.predict(
            [str(p) for p in batch],
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=1280,
            verbose=False,
        )
        for img_path, r in zip(batch, results):
            frame = img_path.stem
            preds = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for j in range(len(boxes)):
                    preds.append({
                        "bbox_2d": [float(boxes[j][0]), float(boxes[j][1]),
                                    float(boxes[j][2]), float(boxes[j][3])],
                        "confidence": float(confs[j]),
                        "class": int(cls[j]),
                    })
                n_with_det += 1
                n_total_det += len(preds)
            with open(out_dir / f"{frame}.json", "w") as f:
                json.dump(preds, f, indent=2)
        if (i + batch_size) % 320 == 0:
            print(f"  {i+batch_size}/{len(all_images)}  ({n_with_det} with detections, {n_total_det} total)")
    print(f"  Final: {n_with_det}/{len(all_images)} frames with detections, {n_total_det} total detections")
    return n_with_det, n_total_det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/drone/drone_urban_v1/weights/best.pt")
    ap.add_argument("--datasets", nargs="+", default=[
        "dataset_nh_v2", "dataset_city_v3", "dataset_coast_v2"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out_subdir", default="yolo_predictions",
                    help="output sub-directory inside the dataset")
    args = ap.parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    total_frames = 0
    total_det = 0
    for ds in args.datasets:
        if not Path(ds).exists():
            print(f"SKIP: {ds} does not exist")
            continue
        nf, nd = process_dataset(model, ds, conf_threshold=args.conf, out_subdir=args.out_subdir)
        total_frames += nf
        total_det += nd

    print(f"\n{'='*50}\nTOTAL: {total_frames} frames with detections, {total_det} detections")


if __name__ == "__main__":
    main()
