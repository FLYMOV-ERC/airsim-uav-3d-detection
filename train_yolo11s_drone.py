#!/usr/bin/env python3
"""
Treina YOLO11s no dataset urbano de drones (city + nh + coast).
Mesma arquitetura usada no tracker /home/ericyos/tub/ubaswarm_ws/src/tracker/

Uso:
    python3 train_yolo11s_drone.py [--epochs 100] [--batch 16] [--imgsz 640]
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/ericyos/airsim/dataset_urban_merged/yolo/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="input image size")
    ap.add_argument("--model", default="yolo11s.pt", help="modelo base (auto-download)")
    ap.add_argument("--name", default="drone_urban_v1", help="nome do run")
    ap.add_argument("--device", default="0", help="GPU id (0) ou 'cpu'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="early stopping patience")
    args = ap.parse_args()

    print(f"Treinando {args.model} em {args.data}")
    print(f"  epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}  device={args.device}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        project="runs/drone",
        save=True,
        plots=True,
        verbose=True,
    )

    print(f"\n=== Treino concluído ===")
    print(f"Best model: runs/drone/{args.name}/weights/best.pt")
    print(f"Last model: runs/drone/{args.name}/weights/last.pt")

    # Avalia no val
    print(f"\nAvaliação final em val:")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
