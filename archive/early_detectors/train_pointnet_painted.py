#!/usr/bin/env python3
"""ARCHIVED. Train the flat (non-hierarchical) PointPainting-style PointNet.

  - Input: (B, N=4096, 4) = xyz + YOLO probability
  - Output:
    * cls_logit: is_drone (handles YOLO false positives)
    * center: drone 3D centre (regression, normalized)
    * size: drone 3D size (regression, normalized)

Class imbalance:
  - roughly 88% positives, 12% negatives
  - weighted BCE, with more weight on the negatives
  - augmentation: random rotation about Y plus jitter

Precursor of the hierarchical PointNet++ version, whose architecture is now in
detection/pointnet/pointnet2.py.
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# =============================================================================
# Dataset
# =============================================================================
class PaintedDroneDataset(Dataset):
    def __init__(self, root, split="train", augment=False):
        self.root = Path(root) / split
        self.augment = augment
        self.samples = sorted((self.root / "point_clouds").glob("*.npy"))
        # Pre-load the labels, for the weighted sampling
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
        pc = np.load(pc_path).astype(np.float32)  # (N, 4)
        lbl = json.load(open(lbl_path))

        is_drone = float(lbl["is_drone"])
        center = np.array(lbl["center_rel_normalized"], dtype=np.float32)
        size = np.array(lbl["size_normalized"], dtype=np.float32)

        if self.augment:
            # Random rotation about Y (leaves the probability channel untouched)
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pc[:, :3] = pc[:, :3] @ R.T
            if is_drone > 0.5:
                center = center @ R.T
            # Small jitter (XYZ only)
            pc[:, :3] += np.random.normal(0, 0.01, pc[:, :3].shape).astype(np.float32)

        return (
            torch.from_numpy(pc),
            torch.tensor(is_drone, dtype=torch.float32),
            torch.from_numpy(center),
            torch.from_numpy(size),
        )


# =============================================================================
# PointNet PAINTED (4-channel input)
# =============================================================================
class PointNetPainted(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.cls_head = nn.Linear(128, 1)
        self.center_head = nn.Linear(128, 3)
        self.size_head = nn.Linear(128, 3)

    def forward(self, x):
        # x: (B, N, 4) - xyz + prob
        x = x.transpose(1, 2)  # (B, 4, N)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=2)[0]  # global feature (B, 1024)
        feat = self.fc(x)
        cls = self.cls_head(feat).squeeze(-1)
        center = self.center_head(feat)
        size = self.size_head(feat)
        return cls, center, size


# =============================================================================
# Loss
# =============================================================================
def compute_loss(cls, c_pred, s_pred, is_drone, c_gt, s_gt,
                 pos_weight=1.0, w_cls=1.0, w_center=5.0, w_size=1.0):
    """Weighted BCE + smooth_L1 regression (only positives)."""
    loss_cls = F.binary_cross_entropy_with_logits(
        cls, is_drone, pos_weight=torch.tensor(pos_weight, device=cls.device))
    pos_mask = is_drone > 0.5
    if pos_mask.sum() > 0:
        loss_center = F.smooth_l1_loss(c_pred[pos_mask], c_gt[pos_mask])
        loss_size = F.smooth_l1_loss(s_pred[pos_mask], s_gt[pos_mask])
    else:
        loss_center = torch.tensor(0.0, device=cls.device)
        loss_size = torch.tensor(0.0, device=cls.device)
    total = w_cls * loss_cls + w_center * loss_center + w_size * loss_size
    return total, {"cls": loss_cls.item(),
                   "center": loss_center.item(),
                   "size": loss_size.item()}


def compute_metrics(cls, c_pred, s_pred, is_drone, c_gt, s_gt):
    cls_pred = (torch.sigmoid(cls) > 0.5).float()
    acc = (cls_pred == is_drone).float().mean().item()
    pos = (is_drone > 0.5)
    tp = ((cls_pred == 1) & pos).float().sum().item()
    fp = ((cls_pred == 1) & ~pos).float().sum().item()
    fn = ((cls_pred == 0) & pos).float().sum().item()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    if pos.sum() > 0:
        c_mse = F.mse_loss(c_pred[pos], c_gt[pos]).item()
        s_mse = F.mse_loss(s_pred[pos], s_gt[pos]).item()
    else:
        c_mse = 0; s_mse = 0
    return {"acc": acc, "prec": prec, "rec": rec, "c_mse": c_mse, "s_mse": s_mse}


# =============================================================================
# Train/Eval
# =============================================================================
def train_epoch(model, loader, opt, device, pos_weight):
    model.train()
    stats = defaultdict(float); n = 0
    for pc, is_drone, c_gt, s_gt in loader:
        pc=pc.to(device); is_drone=is_drone.to(device)
        c_gt=c_gt.to(device); s_gt=s_gt.to(device)
        cls, c, s = model(pc)
        loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward(); opt.step()
        m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
        bs = pc.size(0)
        stats["loss"] += loss.item()*bs
        for k,v in parts.items(): stats[f"loss_{k}"] += v*bs
        for k,v in m.items(): stats[k] += v*bs
        n += bs
    return {k: v/n for k,v in stats.items()}


def eval_epoch(model, loader, device, pos_weight):
    model.eval()
    stats = defaultdict(float); n = 0
    with torch.no_grad():
        for pc, is_drone, c_gt, s_gt in loader:
            pc=pc.to(device); is_drone=is_drone.to(device)
            c_gt=c_gt.to(device); s_gt=s_gt.to(device)
            cls, c, s = model(pc)
            loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt, pos_weight=pos_weight)
            m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
            bs = pc.size(0)
            stats["loss"] += loss.item()*bs
            for k,v in parts.items(): stats[f"loss_{k}"] += v*bs
            for k,v in m.items(): stats[k] += v*bs
            n += bs
    return {k: v/n for k,v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_painted")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)  # N=4096 is heavy, so a smaller batch
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default="runs/pointnet_painted/drone_detector")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = PaintedDroneDataset(args.data, split="train", augment=True)
    val_ds = PaintedDroneDataset(args.data, split="val", augment=False)

    # Weighted sampler, to balance the classes
    n_pos = train_ds.is_drone.sum()
    n_neg = len(train_ds) - n_pos
    weights = np.where(train_ds.is_drone == 1, 1.0/n_pos, 1.0/n_neg)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)

    # the sampler already balances, so pos_weight=1 avoids a double correction
    pos_weight = 1.0
    print(f"  Class balance: {n_pos} pos / {n_neg} neg (sampler balanceia)")
    print(f"  BCE pos_weight: {pos_weight:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = PointNetPainted(in_channels=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointNet params: {n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_acc = 0.0
    log = []
    for epoch in range(1, args.epochs+1):
        train_stats = train_epoch(model, train_loader, opt, device, pos_weight)
        val_stats = eval_epoch(model, val_loader, device, pos_weight)
        sched.step()
        line = (f"E{epoch:3d}/{args.epochs}  "
                f"tr_loss={train_stats['loss']:.3f} tr_acc={train_stats['acc']:.3f}  "
                f"val_loss={val_stats['loss']:.3f} val_acc={val_stats['acc']:.3f} "
                f"val_prec={val_stats['prec']:.3f} val_rec={val_stats['rec']:.3f} "
                f"val_c_mse={val_stats['c_mse']:.3f}")
        print(line)
        log.append({"epoch": epoch,
                    **{f"tr_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()}})

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
                "val_stats": val_stats,
                "args": vars(args),
            }, out_dir / "best.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
    }, out_dir / "last.pt")
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
