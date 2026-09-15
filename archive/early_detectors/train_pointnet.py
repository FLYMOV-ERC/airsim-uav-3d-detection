#!/usr/bin/env python3
"""ARCHIVED. Train the plain PointNet drone detector (the Section 7.3.4 baseline).

  - Input: (N=512, 3) point cloud
  - Output:
    * cls_logit (B, 1): binary "is_drone"
    * center (B, 3): drone centre, NORMALIZED (relative to the cloud centroid, / scale)
    * size (B, 3): drone size, NORMALIZED (/ scale)
  - Loss: BCE(cls) + smooth_L1(center) * is_drone + smooth_L1(size) * is_drone

The training side of the first-attempt numbers reported in Section 7.3.4.
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Dataset
# =============================================================================
class FrustumDroneDataset(Dataset):
    def __init__(self, root, split="train", augment=False):
        self.root = Path(root) / split
        self.augment = augment
        self.samples = sorted((self.root / "point_clouds").glob("*.npy"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_path = self.samples[idx]
        lbl_path = self.root / "labels" / f"{pc_path.stem}.json"
        pc = np.load(pc_path).astype(np.float32)  # (N, 3), already normalised
        lbl = json.load(open(lbl_path))

        is_drone = float(lbl["is_drone"])
        if is_drone > 0.5:
            center = np.array(lbl["center_rel_normalized"], dtype=np.float32)
            size = np.array(lbl["size_normalized"], dtype=np.float32)
        else:
            center = np.zeros(3, dtype=np.float32)
            size = np.zeros(3, dtype=np.float32)

        # Simple augmentation: random rotation about Y, plus jitter
        if self.augment:
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pc = pc @ R.T
            if is_drone > 0.5:
                center = center @ R.T
            # Small jitter
            pc = pc + np.random.normal(0, 0.01, pc.shape).astype(np.float32)

        return (
            torch.from_numpy(pc),
            torch.tensor(is_drone, dtype=torch.float32),
            torch.from_numpy(center),
            torch.from_numpy(size),
        )


# =============================================================================
# PointNet Model (simples, baseado no original)
# =============================================================================
class PointNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared MLPs (a Conv1d is equivalent to a point-wise MLP)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
        )
        # Global max pool + FC
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        # Heads
        self.cls_head = nn.Linear(128, 1)
        self.center_head = nn.Linear(128, 3)
        self.size_head = nn.Linear(128, 3)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=2)[0]  # (B, 512) global feature
        feat = self.fc(x)
        cls = self.cls_head(feat).squeeze(-1)
        center = self.center_head(feat)
        size = self.size_head(feat)
        return cls, center, size


# =============================================================================
# Loss
# =============================================================================
def compute_loss(cls_logit, center_pred, size_pred,
                 is_drone, center_gt, size_gt,
                 w_cls=1.0, w_center=2.0, w_size=1.0):
    """Combined loss."""
    # Classification
    loss_cls = F.binary_cross_entropy_with_logits(cls_logit, is_drone)
    # Regression: positives only
    pos_mask = is_drone > 0.5
    n_pos = pos_mask.sum().item()
    if n_pos > 0:
        loss_center = F.smooth_l1_loss(center_pred[pos_mask], center_gt[pos_mask])
        loss_size = F.smooth_l1_loss(size_pred[pos_mask], size_gt[pos_mask])
    else:
        loss_center = torch.tensor(0.0, device=cls_logit.device)
        loss_size = torch.tensor(0.0, device=cls_logit.device)
    total = w_cls * loss_cls + w_center * loss_center + w_size * loss_size
    return total, {"cls": loss_cls.item(), "center": loss_center.item(), "size": loss_size.item()}


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(cls_logit, center_pred, size_pred,
                    is_drone, center_gt, size_gt):
    """Accuracy + center MSE em normalized space."""
    cls_pred = (torch.sigmoid(cls_logit) > 0.5).float()
    acc = (cls_pred == is_drone).float().mean().item()
    pos_mask = is_drone > 0.5
    if pos_mask.sum() > 0:
        center_mse = F.mse_loss(center_pred[pos_mask], center_gt[pos_mask]).item()
        size_mse = F.mse_loss(size_pred[pos_mask], size_gt[pos_mask]).item()
    else:
        center_mse = 0.0
        size_mse = 0.0
    return {"acc": acc, "center_mse": center_mse, "size_mse": size_mse}


# =============================================================================
# Train loop
# =============================================================================
def train_epoch(model, loader, optimizer, device, desc="train"):
    model.train()
    stats = defaultdict(float)
    n = 0
    for pc, is_drone, c_gt, s_gt in loader:
        pc = pc.to(device); is_drone = is_drone.to(device)
        c_gt = c_gt.to(device); s_gt = s_gt.to(device)

        cls, c, s = model(pc)
        loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)

        bs = pc.size(0)
        stats["loss"] += loss.item() * bs
        for k, v in parts.items(): stats[f"loss_{k}"] += v * bs
        for k, v in m.items(): stats[k] += v * bs
        n += bs
    return {k: v / n for k, v in stats.items()}


def eval_epoch(model, loader, device):
    model.eval()
    stats = defaultdict(float)
    n = 0
    with torch.no_grad():
        for pc, is_drone, c_gt, s_gt in loader:
            pc = pc.to(device); is_drone = is_drone.to(device)
            c_gt = c_gt.to(device); s_gt = s_gt.to(device)
            cls, c, s = model(pc)
            loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt)
            m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
            bs = pc.size(0)
            stats["loss"] += loss.item() * bs
            for k, v in parts.items(): stats[f"loss_{k}"] += v * bs
            for k, v in m.items(): stats[k] += v * bs
            n += bs
    return {k: v / n for k, v in stats.items()}


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_pointnet")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default="runs/pointnet/drone_detector")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    train_ds = FrustumDroneDataset(args.data, split="train", augment=True)
    val_ds = FrustumDroneDataset(args.data, split="val", augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = PointNetDetector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointNet params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    log = []
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device)
        val_stats = eval_epoch(model, val_loader, device)
        scheduler.step()

        line = (f"Epoch {epoch:3d}/{args.epochs}  "
                f"tr_loss={train_stats['loss']:.4f}  tr_acc={train_stats['acc']:.3f}  "
                f"val_loss={val_stats['loss']:.4f}  val_acc={val_stats['acc']:.3f}  "
                f"val_c_mse={val_stats['center_mse']:.4f}  val_s_mse={val_stats['size_mse']:.4f}")
        print(line)
        log.append({"epoch": epoch, **{f"tr_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()}})

        # Save best by val_acc
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
                "args": vars(args),
            }, out_dir / "best.pt")

    # Save last + log
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
    }, out_dir / "last.pt")
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best val acc: {best_val_acc:.3f}")
    print(f"Models saved to: {out_dir}")


if __name__ == "__main__":
    main()
