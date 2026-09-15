#!/usr/bin/env python3
"""FrustumSegNet (F-PointNet style): learns to segment every point as drone or background,
and uses ONLY the drone points (pooling weighted by the mask) to estimate cls / centre / size.
Replaces the depth-band heuristic with a learned segmentation -- robust to a frontal occluder."""
import torch, torch.nn as nn, torch.nn.functional as F


class FrustumSegNet(nn.Module):
    def __init__(self, in_xyz=3):
        super().__init__()
        # per-point MLP (no downsampling -> one output per point)
        self.mlp1 = nn.Sequential(nn.Conv1d(in_xyz,64,1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64,64,1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024), nn.ReLU())
        # per-point segmentation head (local 64 concatenated with global 1024)
        self.seg = nn.Sequential(nn.Conv1d(64+1024,256,1), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Conv1d(256,128,1), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128,1,1))
        # box head: built from the OBJECT feature (pooling weighted by the mask)
        self.box = nn.Sequential(nn.Linear(1024,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(512,256), nn.ReLU())
        self.cls_head=nn.Linear(256,1); self.center_head=nn.Linear(256,3); self.size_head=nn.Linear(256,3)

    def forward(self, x, return_seg=False):
        # x: (B,N,3) normalizado
        B,N,_=x.shape
        p=x.transpose(1,2)                       # (B,3,N)
        l=self.mlp1(p)                            # (B,64,N) local
        g=self.mlp2(l)                            # (B,1024,N)
        gpool=g.max(dim=2,keepdim=True)[0]        # (B,1024,1) global
        seg_logit=self.seg(torch.cat([l, gpool.expand(-1,-1,N)],dim=1)).squeeze(1)  # (B,N)
        # OBJECT pooling: weight the per-point global features by the foreground probability
        w=torch.sigmoid(seg_logit).unsqueeze(1)   # (B,1,N)
        obj=(g*w).sum(dim=2)/(w.sum(dim=2)+1e-6)  # (B,1024) feature do foreground
        h=self.box(obj)
        out=(self.cls_head(h).squeeze(-1), self.center_head(h), self.size_head(h))
        return (out+(seg_logit,)) if return_seg else out
