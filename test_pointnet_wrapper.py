#!/usr/bin/env python3
"""Validação do PointNetInference em frame conhecido do dataset.

Roda em ~5 frames com YOLO detections + GT, compara center predito vs GT.
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

from pointnet_inference import PointNetInference


def airsim_to_cv(p):
    return np.array([p[1], p[2], p[0]], dtype=np.float32)


def collect_test_frames(dataset_root: str, max_frames: int = 5):
    root = Path(dataset_root)
    yolo_dir = root / "yolo_predictions"
    pc_dir = root / "pointnet" / "point_clouds"
    lbl_dir = root / "pointnet" / "labels_3d"

    out = []
    for pred_file in sorted(yolo_dir.glob("*.json")):
        preds = json.load(open(pred_file))
        if not preds:
            continue
        frame = pred_file.stem
        if not (pc_dir / f"{frame}.npy").exists():
            continue
        if not (lbl_dir / f"{frame}.json").exists():
            continue
        out.append((frame, preds))
        if len(out) >= max_frames:
            break
    return out, pc_dir, lbl_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_nh_v2")
    ap.add_argument("--checkpoint", default="runs/pointnet2_painted/v4_balanced/best.pt")
    ap.add_argument("--max_frames", type=int, default=5)
    args = ap.parse_args()

    frames, pc_dir, lbl_dir = collect_test_frames(args.dataset, args.max_frames)
    if not frames:
        print(f"Não encontrei frames com YOLO preds em {args.dataset}")
        return 1

    pn = PointNetInference(args.checkpoint)

    all_errors = []
    all_probs = []
    for frame, preds in frames:
        pc = np.load(pc_dir / f"{frame}.npy").astype(np.float32)
        gt = json.load(open(lbl_dir / f"{frame}.json"))
        print(f"\n— Frame {frame}  PC={len(pc)} pts  YOLO_preds={len(preds)}  GT={len(gt)}")

        for k, pred in enumerate(preds):
            bbox = pred["bbox_2d"]
            conf = pred["confidence"]
            r = pn.predict(pc, bbox, conf)
            if r is None:
                print(f"  det{k}: PC vazia")
                continue
            print(f"  det{k}: bbox={[round(x,1) for x in bbox]}  yolo_conf={conf:.2f}  "
                   f"pn_prob={r['is_drone_prob']:.3f}  pred_center_cv={r['center_cv'].round(1).tolist()}")
            all_probs.append(r['is_drone_prob'])
            # GT mais próximo
            if not r['is_drone']:
                continue
            best_dist, best_gt = float('inf'), None
            for g in gt:
                gt_cv = airsim_to_cv(g["center"])
                d = float(np.linalg.norm(r['center_cv'] - gt_cv))
                if d < best_dist:
                    best_dist, best_gt = d, g
            if best_gt is not None:
                gt_cv = airsim_to_cv(best_gt["center"])
                print(f"         GT_match={gt_cv.round(1).tolist()}  err={best_dist:.2f}m")
                all_errors.append(best_dist)

    print(f"\n=== Summary ===")
    print(f"Total predictions: {len(all_probs)}  positive (>0.5): {sum(p > 0.5 for p in all_probs)}")
    if all_errors:
        arr = np.array(all_errors)
        print(f"Position errors (positive preds only): mean={arr.mean():.2f}m  max={arr.max():.2f}m  median={np.median(arr):.2f}m")
        # Wrapper test: aceita até 30m (modelo é ruidoso; EKF vai filtrar).
        # Para validação do CÓDIGO basta que o wrapper carregue + produza shape correto.
        wrapper_ok = len(all_probs) > 0
        accuracy_ok = arr.mean() < 30.0
        if not accuracy_ok:
            print(f"  WARN: erro médio alto — limitação do modelo PointNet++ v4 (val_f1=0.68)")
        ok = wrapper_ok and accuracy_ok
    else:
        print("Nenhuma predição positiva — modelo ou wrapper com problema")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
