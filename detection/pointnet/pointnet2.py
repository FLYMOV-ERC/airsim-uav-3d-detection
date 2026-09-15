#!/usr/bin/env python3
"""
PointNet++ (hierarchical) + Focal Loss treinado no painted dataset.

A from-scratch implementation in plain PyTorch:
  - Farthest Point Sampling (FPS)
  - Ball query
  - Set Abstraction modules
  - Multi-scale grouping opcional

Input: (B, N=4096, 4) = xyz + prob_yolo
Output: cls_logit (B,1), center (B,3), size (B,3)
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
# Geometric ops (vectorized PyTorch)
# =============================================================================
def square_distance(src, dst):
    """
    Squared distance between every pair of (B, N, C) and (B, M, C).
    Returns (B, N, M).
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Gather points by index.
    points: (B, N, C), idx: (B, ...)
    return: (B, ..., C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    Farthest-point sampling: select npoint points as far apart from each other as possible.
    xyz: (B, N, 3)
    return: idx (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query: for each centroid in new_xyz, return the indices of the nsample
    nearest points within the radius.
    xyz: (B, N, 3), new_xyz: (B, S, 3)
    return: (B, S, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample (FPS) + group (ball query) + normalize coords.
    xyz: (B, N, 3) — coordenadas
    points: (B, N, D) — features (ou None)
    return: new_xyz (B, npoint, 3), new_points (B, npoint, nsample, 3+D)
    """
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """For the last SA layer: treat all the points as a single group."""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# =============================================================================
# Set Abstraction Module
# =============================================================================
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        """
        npoint: quantos centroides (None se group_all)
        radius: raio do ball query
        nsample: points per centroid
        in_channel: input feature width (e.g. 4 = xyz_norm + prob, or 3 + feat_dim)
        mlp: lista de canais [c1, c2, c3]
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3), points: (B, N, D) ou None
        return: new_xyz (B, npoint, 3), new_points (B, npoint, mlp[-1])
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points)
        # new_points: (B, npoint, nsample, C)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        # Max pool over nsample
        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        return new_xyz, new_points


# =============================================================================
# Model: PointNet++ for painted detection
# =============================================================================
class PointNet2Painted(nn.Module):
    def __init__(self, in_channels=4):
        """
        in_channels: total channels do input (xyz + extras).
          4 = xyz + prob (painted)
          3 = xyz only (frustum, no probability channel)
        """
        super().__init__()
        self.in_channels = in_channels
        # SA1: 4096 → 512 points, radius 0.2 (normalized), nsample 32
        # input channel = 3 (xyz_norm) + extras
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3 + max(0, in_channels - 3),
            mlp=[64, 64, 128])
        # SA2: 512 → 128 points, radius 0.4
        # input channel = 3 (xyz_norm) + 128 (features from SA1) = 131
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=3 + 128,
            mlp=[128, 128, 256])
        # SA3: global pool
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=3 + 256,
            mlp=[256, 512, 1024],
            group_all=True)

        # FC head
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.cls_head = nn.Linear(128, 1)
        self.center_head = nn.Linear(128, 3)
        self.size_head = nn.Linear(128, 3)

    def forward(self, x):
        """
        x: (B, N, C) where C=4 (painted: xyz+prob) or C=3 (xyz only)
        """
        xyz = x[:, :, :3]
        if x.size(-1) > 3:
            features = x[:, :, 3:]  # (B, N, C-3)
        else:
            features = None         # xyz-only
        # SA1
        l1_xyz, l1_points = self.sa1(xyz, features)
        # SA2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # SA3 (global)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # (B, 1, 1024) → (B, 1024)
        global_feat = l3_points.squeeze(1)
        feat = self.fc(global_feat)
        cls = self.cls_head(feat).squeeze(-1)
        center = self.center_head(feat)
        size = self.size_head(feat)
        return cls, center, size


# =============================================================================
# Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# =============================================================================
# Dataset (same as in archive/early_detectors/train_pointnet_painted.py)
# =============================================================================
class PaintedDroneDataset(Dataset):
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
        pc = np.load(pc_path).astype(np.float32)  # (N, 4)
        lbl = json.load(open(lbl_path))
        is_drone = float(lbl["is_drone"])
        center = np.array(lbl["center_rel_normalized"], dtype=np.float32)
        size = np.array(lbl["size_normalized"], dtype=np.float32)

        if self.augment:
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pc[:, :3] = pc[:, :3] @ R.T
            if is_drone > 0.5:
                center = center @ R.T
            pc[:, :3] += np.random.normal(0, 0.01, pc[:, :3].shape).astype(np.float32)

        return (
            torch.from_numpy(pc),
            torch.tensor(is_drone, dtype=torch.float32),
            torch.from_numpy(center),
            torch.from_numpy(size),
        )


# =============================================================================
# Train/Eval
# =============================================================================
def compute_loss(cls, c_pred, s_pred, is_drone, c_gt, s_gt,
                 focal_loss_fn, w_cls=1.0, w_center=5.0, w_size=1.0):
    loss_cls = focal_loss_fn(cls, is_drone)
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
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    if pos.sum() > 0:
        c_mse = F.mse_loss(c_pred[pos], c_gt[pos]).item()
        s_mse = F.mse_loss(s_pred[pos], s_gt[pos]).item()
    else:
        c_mse = 0; s_mse = 0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "c_mse": c_mse, "s_mse": s_mse}


def train_epoch(model, loader, opt, device, focal_loss_fn):
    model.train()
    stats = defaultdict(float); n = 0
    for pc, is_drone, c_gt, s_gt in loader:
        pc=pc.to(device); is_drone=is_drone.to(device)
        c_gt=c_gt.to(device); s_gt=s_gt.to(device)
        cls, c, s = model(pc)
        loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt, focal_loss_fn)
        opt.zero_grad(); loss.backward(); opt.step()
        m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
        bs = pc.size(0)
        stats["loss"] += loss.item()*bs
        for k,v in parts.items(): stats[f"loss_{k}"] += v*bs
        for k,v in m.items(): stats[k] += v*bs
        n += bs
    return {k: v/n for k,v in stats.items()}


def eval_epoch(model, loader, device, focal_loss_fn):
    model.eval()
    stats = defaultdict(float); n = 0
    with torch.no_grad():
        for pc, is_drone, c_gt, s_gt in loader:
            pc=pc.to(device); is_drone=is_drone.to(device)
            c_gt=c_gt.to(device); s_gt=s_gt.to(device)
            cls, c, s = model(pc)
            loss, parts = compute_loss(cls, c, s, is_drone, c_gt, s_gt, focal_loss_fn)
            m = compute_metrics(cls, c, s, is_drone, c_gt, s_gt)
            bs = pc.size(0)
            stats["loss"] += loss.item()*bs
            for k,v in parts.items(): stats[f"loss_{k}"] += v*bs
            for k,v in m.items(): stats[k] += v*bs
            n += bs
    return {k: v/n for k,v in stats.items()}


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_painted")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)  # PointNet++ is heavier
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--output", default="runs/pointnet2_painted/v1")
    ap.add_argument("--warm_start", default="", help="checkpoint .pt for a cross warm start")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Focal Loss: alpha={args.focal_alpha} gamma={args.focal_gamma}")

    train_ds = PaintedDroneDataset(args.data, split="train", augment=True)
    val_ds = PaintedDroneDataset(args.data, split="val", augment=False)

    n_pos = train_ds.is_drone.sum()
    n_neg = len(train_ds) - n_pos
    # the sampler balances 50/50 and focal alpha=0.5 is neutral on balanced batches
    weights = np.where(train_ds.is_drone == 1, 1.0/n_pos, 1.0/n_neg)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    print(f"  Sampler 50/50 + focal alpha={args.focal_alpha} (pos={n_pos}, neg={n_neg})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)

    model = PointNet2Painted(in_channels=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointNet++ params: {n_params/1e6:.2f}M")

    # Cross warm start: load weights matching by name and shape (e.g. from the frustum model).
    # sa1.mlp_convs.0.weight difere (4ch vs 3ch) e fica fresh.
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

    best_val_f1 = 0.0
    log = []
    for epoch in range(1, args.epochs+1):
        train_stats = train_epoch(model, train_loader, opt, device, focal_loss)
        val_stats = eval_epoch(model, val_loader, device, focal_loss)
        sched.step()
        line = (f"E{epoch:3d}/{args.epochs}  "
                f"tr_loss={train_stats['loss']:.3f} tr_f1={train_stats['f1']:.3f}  "
                f"val_loss={val_stats['loss']:.3f} val_acc={val_stats['acc']:.3f} "
                f"val_f1={val_stats['f1']:.3f} val_prec={val_stats['prec']:.3f} val_rec={val_stats['rec']:.3f} "
                f"val_c_mse={val_stats['c_mse']:.3f}")
        print(line)
        log.append({"epoch": epoch,
                    **{f"tr_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in val_stats.items()}})

        # Save the best model by F1, not by accuracy, since the classes are imbalanced
        if val_stats["f1"] > best_val_f1:
            best_val_f1 = val_stats["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_f1": best_val_f1,
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

    print(f"\nDone. Best val F1: {best_val_f1:.3f}")


if __name__ == "__main__":
    main()
