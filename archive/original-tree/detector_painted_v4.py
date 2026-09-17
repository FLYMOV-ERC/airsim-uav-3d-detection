#!/usr/bin/env python3
"""Detector 3D via PointNet++ Painted v4 (modelo `runs/pointnet2_painted/v4_balanced/best.pt`).

Pipeline (igual a build_painted_dataset.py):
  1. Pinta full PC com prob=confidence dentro do bbox, 0 fora
  2. Sample fixo N=4096
  3. Normaliza XYZ
  4. Forward PointNet++
  5. Desnormaliza
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch

from detector_base import Detector3D, IMAGE_W, IMAGE_H, FX, FY, CX, CY
from train_pointnet2_painted import PointNet2Painted


N_POINTS_PAINTED = 4096


class PaintedPointNetDetector(Detector3D):
    name = "painted_v4"

    def __init__(self,
                 checkpoint_path: str = "runs/pointnet2_painted/v4_balanced/best.pt",
                 device: Optional[str] = None,
                 cls_threshold: float = 0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cls_threshold = cls_threshold
        self.model = PointNet2Painted(in_channels=4).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[PaintedPN] {checkpoint_path}  device={self.device}")
        if isinstance(ckpt, dict) and "val_f1" in ckpt:
            print(f"           val_f1={ckpt['val_f1']:.3f}  epoch={ckpt.get('epoch','?')}")

    @staticmethod
    def _project_to_pixel(pc):
        z = pc[:, 2]
        valid = z > 0.1
        z_safe = np.where(valid, z, 1.0)
        u = pc[:, 0] * FX / z_safe + CX
        v = pc[:, 1] * FY / z_safe + CY
        return u, v, valid

    @staticmethod
    def _paint(pc, bbox, conf):
        x1, y1, x2, y2 = bbox
        u, v, valid_z = PaintedPointNetDetector._project_to_pixel(pc)
        inside = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        prob = np.where(inside, conf, 0.0).astype(np.float32)
        return np.concatenate([pc, prob[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def _sample_to_fixed(pts, n, rng=None):
        rng = rng or np.random.default_rng()
        if len(pts) >= n:
            idx = rng.choice(len(pts), n, replace=False)
        else:
            idx = rng.choice(len(pts), n, replace=True)
        return pts[idx]

    @staticmethod
    def _normalize_xyz(pc_xyz):
        centroid = pc_xyz.mean(axis=0)
        centered = pc_xyz - centroid
        scale = float(np.linalg.norm(centered, axis=1).max())
        scale = max(scale, 1e-6)
        return centered / scale, centroid, scale

    @torch.inference_mode()
    def predict(self, pc_cv: np.ndarray, depth: np.ndarray,
                bbox: tuple, confidence: float) -> Optional[Dict[str, Any]]:
        if pc_cv is None or len(pc_cv) < 100:
            return None
        painted = self._paint(pc_cv, bbox, confidence)
        sampled = self._sample_to_fixed(painted, N_POINTS_PAINTED)
        xyz_norm, centroid, scale = self._normalize_xyz(sampled[:, :3])
        sampled[:, :3] = xyz_norm

        x = torch.from_numpy(sampled).unsqueeze(0).to(self.device)
        cls_logit, center_norm, size_norm = self.model(x)
        is_drone_prob = float(torch.sigmoid(cls_logit).item())
        center_n = center_norm.squeeze(0).cpu().numpy()
        size_n = size_norm.squeeze(0).cpu().numpy()

        center_cv = center_n * scale + centroid
        size_cv = np.abs(size_n) * scale

        return {
            'is_drone_prob': is_drone_prob,
            'is_drone': is_drone_prob >= self.cls_threshold,
            'center_cv': center_cv.astype(np.float32),
            'size_cv': size_cv.astype(np.float32),
            'centroid_cv': centroid.astype(np.float32),
            'scale': scale,
            'method': self.name,
        }
