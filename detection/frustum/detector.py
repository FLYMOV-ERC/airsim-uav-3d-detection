#!/usr/bin/env python3
"""3D detector: PointNet++ over the frustum extracted from a YOLO box (model trained
on dataset_frustum_pn2 with YOLO at conf=0.10).

Pipeline:
  1. Crop the frustum: the cloud points inside the 2D box
  2. Resample to a fixed N=4096 (with replacement if there are too few)
  3. Normaliza XYZ (centroide + escala)
  4. Forward PointNet++ (in_channels=3)
  5. Desnormaliza
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root

from typing import Optional, Dict, Any
import numpy as np
import torch

from detection.base import Detector3D, IMAGE_W, IMAGE_H, FX, FY, CX, CY
from detection.pointnet.pointnet2 import PointNet2Painted


N_POINTS = 4096
MIN_FRUSTUM_PTS = 8


class FrustumPN2Detector(Detector3D):
    name = "frustum_pn2"

    def __init__(self,
                 checkpoint_path: str = "runs/pointnet2_frustum/v1/best.pt",
                 device: Optional[str] = None,
                 cls_threshold: float = 0.5,
                 band_m: float = 0.0,
                 arch: str = 'pointnet'):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cls_threshold = cls_threshold
        self.band_m = band_m
        if arch=='convnet':
            from detection.frustum.convnet import FrustumConvNet
            self.model = FrustumConvNet().to(self.device)
        elif arch=='segnet':
            from detection.frustum.segnet import FrustumSegNet
            self.model = FrustumSegNet().to(self.device)
        else:
            self.model = PointNet2Painted(in_channels=3).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[FrustumPN2] {checkpoint_path}  device={self.device}")
        if isinstance(ckpt, dict) and "val_f1" in ckpt:
            print(f"            val_f1={ckpt['val_f1']:.3f}  epoch={ckpt.get('epoch','?')}")

    @staticmethod
    def _project_to_pixel(pc):
        z = pc[:, 2]
        valid = z > 0.1
        z_safe = np.where(valid, z, 1.0)
        u = pc[:, 0] * FX / z_safe + CX
        v = pc[:, 1] * FY / z_safe + CY
        return u, v, valid

    @staticmethod
    def _crop_frustum(pc, bbox):
        x1, y1, x2, y2 = bbox
        u, v, valid = FrustumPN2Detector._project_to_pixel(pc)
        inside = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        return pc[inside]

    @staticmethod
    def _sample_to_fixed(pts, n, rng=None):
        rng = rng or np.random.default_rng()
        if len(pts) >= n:
            idx = rng.choice(len(pts), n, replace=False)
        else:
            idx = rng.choice(len(pts), n, replace=True)
        return pts[idx]

    @staticmethod
    def _normalize_pc(pc_xyz):
        import os
        mode = os.environ.get("NORM_MODE", "max")
        centroid = pc_xyz.mean(axis=0)
        centered = pc_xyz - centroid
        dists = np.linalg.norm(centered, axis=1)
        if mode == "p95":
            scale = float(np.percentile(dists, 95))
        elif mode == "fixed":
            scale = 8.0
        else:
            scale = float(dists.max())
        scale = max(scale, 1e-6)
        return centered / scale, centroid, scale

    @torch.inference_mode()
    def predict(self, pc_cv: np.ndarray, depth: np.ndarray,
                bbox: tuple, confidence: float) -> Optional[Dict[str, Any]]:
        if pc_cv is None or len(pc_cv) < 10:
            return None
        frustum = self._crop_frustum(pc_cv, bbox)
        if self.band_m > 0 and len(frustum) >= 5:
            z = frustum[:, 2]; z0 = np.percentile(z, 5.0)
            frustum = frustum[z <= z0 + self.band_m]
        if len(frustum) < MIN_FRUSTUM_PTS:
            return None
        sampled = self._sample_to_fixed(frustum, N_POINTS)
        xyz_norm, centroid, scale = self._normalize_pc(sampled)

        x = torch.from_numpy(xyz_norm.astype(np.float32)).unsqueeze(0).to(self.device)
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
            'n_frustum_pts': len(frustum),
        }
