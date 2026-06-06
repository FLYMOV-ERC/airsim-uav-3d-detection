#!/usr/bin/env python3
"""Frustum-ConvNet (variante): agrupa o frustum em FATIAS de profundidade (z=fwd),
extrai feature por fatia com mini-PointNet, e aplica conv 1D ao longo das fatias.
Cabeças: cls (is_drone) + center(3) + size(3). Mesma interface de I/O do PointNet2Painted.
Lida com a distribuição de profundidade sem normalização por max (≠ gambiarra)."""
import torch, torch.nn as nn, torch.nn.functional as F


class FrustumConvNet(nn.Module):
    def __init__(self, n_slices=16, in_xyz=3):
        super().__init__()
        self.n_slices = n_slices
        # mini-PointNet por fatia (compartilhado): MLP sobre pontos → maxpool
        self.pn = nn.Sequential(
            nn.Conv1d(in_xyz, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        # conv 1D ao longo das fatias (sequência de profundidade)
        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(256, 128), nn.ReLU())
        self.cls_head = nn.Linear(128, 1)
        self.center_head = nn.Linear(128, 3)
        self.size_head = nn.Linear(128, 3)

    def forward(self, x):
        # x: (B, N, 3) já normalizado (centróide+escala). z=fwd em x[...,2]
        B, N, _ = x.shape
        z = x[:, :, 2]
        # bins de profundidade por quantis (robusto), índice 0..n_slices-1
        zmin = z.min(dim=1, keepdim=True)[0]; zmax = z.max(dim=1, keepdim=True)[0]
        binw = (zmax - zmin) / self.n_slices + 1e-6
        idx = ((z - zmin) / binw).long().clamp(0, self.n_slices - 1)  # (B,N)
        pf = self.pn(x.transpose(1, 2))  # (B,128,N)
        # max-pool por fatia (vetorizado, scatter_reduce amax) → (B,128,n_slices)
        slices = x.new_full((B, 128, self.n_slices), -1e4)
        slices.scatter_reduce_(2, idx.unsqueeze(1).expand(-1, 128, -1), pf,
                               reduce='amax', include_self=True)
        slices = torch.where(slices < -1e3, torch.zeros_like(slices), slices)
        c = self.conv(slices)                 # (B,256,n_slices)
        g = c.max(dim=2)[0]                    # (B,256)
        h = self.head(g)
        return self.cls_head(h).squeeze(-1), self.center_head(h), self.size_head(h)
