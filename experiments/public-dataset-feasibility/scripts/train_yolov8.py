#!/usr/bin/env python3
"""Train a YOLOv8 detector on a YOLO-format dataset.

This is the training entry point used for all sixteen runs recorded in
``results/run-index.csv`` -- the thirteen VisDrone2019-DET attempts and the three
single-class 'drone' attempts on the Holybro recording.

Historical note, deliberately preserved
---------------------------------------
The original script instantiated the model as ``YOLO('yolov8n.yaml')`` while also
passing ``pretrained=True``. In Ultralytics, building from the architecture
``.yaml`` starts from random weights; ``pretrained=True`` is honoured when the
model is built from a checkpoint such as ``yolov8n.pt``. The default below keeps
``yolov8n.yaml`` so that re-running this script reproduces the recorded runs
exactly. This matters: it is the reason the VisDrone validation mAP figures in
``results/visdrone/`` must not be read as evidence that VisDrone is hard or
untrainable. Pass ``--model yolov8n.pt`` to start from the COCO checkpoint
instead.

Example
-------
    python train_yolov8.py --data ./datasets/holybro/yolo_dataset/data.yaml \
        --name yolov8_holybro3 --device mps
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data",
        required=True,
        help="path to the YOLO dataset descriptor (data.yaml)",
    )
    p.add_argument(
        "--model",
        default="yolov8n.yaml",
        help=(
            "model to build from. 'yolov8n.yaml' is the architecture definition "
            "and starts from random weights (this is what the recorded runs "
            "used); 'yolov8n.pt' starts from the COCO checkpoint. Default: "
            "yolov8n.yaml"
        ),
    )
    p.add_argument("--name", required=True, help="run name under runs/detect/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' picks mps when available and falls back to cpu; or pass "
        "'cpu', 'mps', '0', etc. explicitly. Default: auto",
    )
    p.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="passed straight through to Ultralytics (default: on)",
    )
    p.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false"
    )
    return p.parse_args()


def resolve_device(requested):
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    args = parse_args()
    from ultralytics import YOLO

    device = resolve_device(args.device)
    print("Using device: %s" % device)
    print("Building model from: %s" % args.model)
    if args.model.endswith(".yaml"):
        print(
            "NOTE: building from an architecture yaml starts from random "
            "weights, regardless of --pretrained."
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
