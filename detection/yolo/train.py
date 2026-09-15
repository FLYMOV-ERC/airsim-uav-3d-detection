#!/usr/bin/env python3
"""Train the YOLOv11-small 2D drone detector.

This is the first stage of the pipeline: a one-stage detector for the single
class "drone", trained on the pixel-exact synthetic composites of Section 7.2.3
(see ``dataset_generation/synthetic/``).

The configuration reported in the dissertation is ``--imgsz 1280 --epochs 80``,
with the standard Ultralytics recipe left at its defaults -- SGD with Nesterov
momentum, weight decay, a cosine learning-rate schedule, mosaic/HSV/scale
augmentation and early stopping on the validation metric.  That run reaches
mAP50 = 0.994 on a held-out synthetic split (Section 7.5.1, Figure 7.7).

The defaults below are the script's own historical defaults (640 px, 100 epochs),
not the reported configuration; pass the flags explicitly to reproduce the
chapter:

    python detection/yolo/train.py \\
        --data datasets/dataset_urban_merged/yolo/dataset.yaml \\
        --imgsz 1280 --epochs 80 --name drone_synth_v1

Requires Ultralytics YOLO, which is AGPL-3.0; see NOTICE.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from common.config import CONFIG

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help="path to the YOLO dataset.yaml (default: "
                         "<data_root>/<datasets>/dataset_urban_merged/yolo/dataset.yaml)")
    ap.add_argument("--data-root", default=None,
                    help="override paths.data_root from configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="batch size")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="input image size; the dissertation uses 1280")
    ap.add_argument("--model", default="yolo11s.pt", help="base model (auto-downloaded)")
    ap.add_argument("--name", default="drone_urban_v1", help="run name")
    ap.add_argument("--project", default=None,
                    help="parent directory for runs (default: <data_root>/<runs>/drone)")
    ap.add_argument("--device", default="0", help="GPU id (e.g. 0) or 'cpu'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience")
    args = ap.parse_args()

    if args.data_root:
        CONFIG.set("paths.data_root", args.data_root)
    data = args.data or str(CONFIG.path("datasets", "dataset_urban_merged", "yolo", "dataset.yaml"))
    project = args.project or str(CONFIG.path("runs", "drone"))

    print(f"Training {args.model} on {data}")
    print(f"  epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}  device={args.device}")

    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        project=project,
        save=True,
        plots=True,
        verbose=True,
    )

    print("\n=== Training finished ===")
    print(f"Best model: {project}/{args.name}/weights/best.pt")
    print(f"Last model: {project}/{args.name}/weights/last.pt")

    print("\nFinal evaluation on the validation split:")
    metrics = model.val()
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
