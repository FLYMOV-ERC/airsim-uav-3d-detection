#!/usr/bin/env python3
"""Train PointNet++ on the frustum-YOLO dataset (each YOLO box at conf=0.10 is one sample).

Input: (B, N=4096, 3) = xyz only (no probability channel -- the frustum is already cropped)
Output:
  - cls_logit (B,): is_drone (filters YOLO false positives)
  - center (B, 3): drone 3D center normalizado
  - size (B, 3): drone 3D size normalizado

Trained with focal loss + a WeightedRandomSampler at 50/50.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from detection.pointnet.pointnet2 import (
    PointNet2Painted, FocalLoss,
)


# ─────────────────────────────────────────────────────────────────────────────
class FrustumPN2Dataset(Dataset):
    def __init__(self, root, split="train", augment=False):
        self.root = Path(root) / split
        self.augment = augment
        self.samples = sorted((self.root / "point_clouds").glob("*.npy"))
        self.is_drone = []
        for p in self.samples:
            lp = self.root / "labels" / f"{p.stem}.json"
            self.is_drone.append(json.load(open(lp))["is_drone"])
        self.is_drone = np.array(self.is_drone)
        n_pos = int(self.is_drone.sum())
        n_neg = len(self.is_drone) - n_pos
        print(f"  {split}: {len(self.samples)} samples ({n_pos} pos, {n_neg} neg)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_path = self.samples[idx]
        lbl_path = self.root / "labels" / f"{pc_path.stem}.json"
        pc_xyz = np.load(pc_path).astype(np.float32)  # (N, 3)
        lbl = json.load(open(lbl_path))

        is_drone = float(lbl["is_drone"])
        center = np.array(lbl["center_rel_normalized"], dtype=np.float32)
        size = np.array(lbl["size_normalized"], dtype=np.float32)

        if self.augment:
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pc_xyz = pc_xyz @ R.T
            if is_drone > 0.5:
                center = center @ R.T
            pc_xyz = pc_xyz + np.random.normal(0, 0.01, pc_xyz.shape).astype(np.float32)

        return (
            torch.from_numpy(pc_xyz),
            torch.tensor(is_drone, dtype=torch.float32),
            torch.from_numpy(center),
            torch.from_numpy(size),
        )


# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(cls, c_pred, s_pred, is_drone, c_gt, s_gt, focal_fn,
                 w_cls=1.0, w_center=5.0, w_size=1.0):
    loss_cls = focal_fn(cls, is_drone)
    pos = is_drone > 0.5
    if pos.sum() > 0:
        loss_center = F.smooth_l1_loss(c_pred[pos], c_gt[pos])
        loss_size = F.smooth_l1_loss(s_pred[pos], s_gt[pos])
    else:
        loss_center = torch.tensor(0.0, device=cls.device)
        loss_size = torch.tensor(0.0, device=cls.device)
    total = w_cls * loss_cls + w_center * loss_center + w_size * loss_size
    return total, {"cls": loss_cls.item(),
                   "center": loss_center.item(),
                   "size": loss_size.item()}


def compute_metrics(cls, c_pred, s_pred, is_drone, c_gt, s_gt):
    pred = (torch.sigmoid(cls) > 0.5).float()
    acc = (pred == is_drone).float().mean().item()
    pos = is_drone > 0.5
    tp = ((pred == 1) & pos).float().sum().item()
    fp = ((pred == 1) & ~pos).float().sum().item()
    fn = ((pred == 0) & pos).float().sum().item()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    if pos.sum() > 0:
        c_mse = F.mse_loss(c_pred[pos], c_gt[pos]).item()
        s_mse = F.mse_loss(s_pred[pos], s_gt[pos]).item()
    else:
        c_mse = 0; s_mse = 0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "c_mse": c_mse, "s_mse": s_mse}


def train_epoch(model, loader, opt, device, focal_fn):
    model.train()
    stats = defaultdict(float); n = 0
    for pc, is_drone, c_gt, s_gt in loader:
        pc = pc.to(device); is_drone = is_drone.to(device)
        c_gt = c_gt.to(device); s_gt = s_gt.to(device)
        cls, c, s = model(pc)
        loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt, focal_fn)
        opt.zero_grad(); loss.backward(); opt.step()
        m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
        bs = pc.size(0)
        stats["loss"] += loss.item() * bs
        for k, v in parts.items(): stats[f"loss_{k}"] += v * bs
        for k, v in m.items(): stats[k] += v * bs
        n += bs
    return {k: v / n for k, v in stats.items()}


def eval_epoch(model, loader, device, focal_fn):
    model.eval()
    stats = defaultdict(float); n = 0
    with torch.no_grad():
        for pc, is_drone, c_gt, s_gt in loader:
            pc = pc.to(device); is_drone = is_drone.to(device)
            c_gt = c_gt.to(device); s_gt = s_gt.to(device)
            cls, c, s = model(pc)
            loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt, focal_fn)
            m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
            bs = pc.size(0)
            stats["loss"] += loss.item() * bs
            for k, v in parts.items(): stats[f"loss_{k}"] += v * bs
            for k, v in m.items(): stats[k] += v * bs
            n += bs
    return {k: v / n for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_frustum_pn2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--focal_alpha", type=float, default=0.5)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--output", default="runs/pointnet2_frustum/v1")
    ap.add_argument("--arch", default="pointnet", choices=["pointnet","convnet"])
    ap.add_argument("--warm_start", default="", help="checkpoint .pt for a cross warm start")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Focal alpha={args.focal_alpha} gamma={args.focal_gamma}")

    train_ds = FrustumPN2Dataset(args.data, split="train", augment=True)
    val_ds = FrustumPN2Dataset(args.data, split="val", augment=False)

    n_pos = train_ds.is_drone.sum()
    n_neg = len(train_ds) - n_pos
    weights = np.where(train_ds.is_drone == 1, 1.0 / n_pos, 1.0 / n_neg)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    print(f"  Sampler 50/50 (pos={n_pos}, neg={n_neg})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)

    # PointNet++ with in_channels=3 (no probability channel)
    if args.arch == "convnet":
        from detection.frustum.convnet import FrustumConvNet
        model = FrustumConvNet().to(device)
    else:
        model = PointNet2Painted(in_channels=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointNet++ params: {n_params/1e6:.2f}M  in_channels=3")

    # Cross warm start: load the weights of a checkpoint (e.g. the painted model) that match
    # by name and shape. Only sa1.mlp_convs.0.weight differs (3ch vs 4ch) and stays fresh.
    if args.warm_start:
        ck = torch.load(args.warm_start, map_location=device, weights_only=False)
        src = ck.get('model_state_dict', ck)
        tgt = model.state_dict()
        loaded, skipped = 0, []
        for k, v in src.items():
            if k in tgt and tgt[k].shape == v.shape:
                tgt[k] = v; loaded += 1
            else:
                skipped.append(k)
        model.load_state_dict(tgt)
        print(f"[warm-start] de {args.warm_start}: {loaded}/{len(tgt)} tensores carregados; "
              f"skip={skipped}")

    focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1 = 0.0
    log = []
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, device, focal_loss)
        va = eval_epoch(model, val_loader, device, focal_loss)
        sched.step()
        print(f"E{epoch:3d}/{args.epochs}  "
              f"tr_loss={tr['loss']:.3f} tr_f1={tr['f1']:.3f}  "
              f"va_loss={va['loss']:.3f} va_acc={va['acc']:.3f} va_f1={va['f1']:.3f} "
              f"va_prec={va['prec']:.3f} va_rec={va['rec']:.3f} "
              f"va_c_mse={va['c_mse']:.3f}")
        log.append({"epoch": epoch,
                    **{f"tr_{k}": v for k, v in tr.items()},
                    **{f"va_{k}": v for k, v in va.items()}})
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_f1,
                "val_stats": va,
                "args": vars(args),
            }, out_dir / "best.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
    }, out_dir / "last.pt")
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Best val F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()
