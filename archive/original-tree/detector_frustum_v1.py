#!/usr/bin/env python3
"""Detector 3D via PointNet v1 + frustum (modelo `runs/pointnet/drone_detector/best.pt`).

Pipeline:
  1. Crop frustum: pontos do PC dentro do bbox 2D projetado
  2. Sample fixo N=512
  3. Normaliza XYZ (centroide + escala)
  4. Forward PointNet v1
  5. Desnormaliza: center_cv = center_rel * scale + centroid
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch

from detector_base import Detector3D, IMAGE_W, IMAGE_H, FX, FY, CX, CY
from train_pointnet import PointNetDetector


N_POINTS_FRUSTUM = 512


class FrustumPointNetDetector(Detector3D):
    name = "frustum_pointnet"

    def __init__(self,
                 checkpoint_path: str = "runs/pointnet/drone_detector/best.pt",
                 device: Optional[str] = None,
                 cls_threshold: float = 0.5,
                 bbox_pad_pct: float = 0.0):
        """
        bbox_pad_pct: percentual de padding na bbox antes de extrair frustum.
            0    = sem padding (bbox YOLO original)
            100  = duplica W e H
            300  = quadruplica W e H (necessário pra recuperar regime de treino)

            Motivação: dataset_pointnet foi construído com bboxes simGetDetections
            (generosas, com margem) → frustum captura drone+background → scale ~50m.
            YOLO dá bbox apertada → frustum só drone → scale ~0.4m.
            Modelo nunca viu scale tão pequena → falha. Padding restaura distribuição.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cls_threshold = cls_threshold
        self.bbox_pad_pct = float(bbox_pad_pct)
        self.model = PointNetDetector().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[FrustumPN] {checkpoint_path}  device={self.device}")
        if isinstance(ckpt, dict) and "val_acc" in ckpt:
            print(f"           val_acc={ckpt['val_acc']:.3f}  epoch={ckpt.get('epoch','?')}")

    @staticmethod
    def _project_to_pixel(pc: np.ndarray):
        z = pc[:, 2]
        valid = z > 0.1
        z_safe = np.where(valid, z, 1.0)
        u = pc[:, 0] * FX / z_safe + CX
        v = pc[:, 1] * FY / z_safe + CY
        return u, v, valid

    @staticmethod
    def _crop_frustum(pc, bbox):
        x1, y1, x2, y2 = bbox
        u, v, valid = FrustumPointNetDetector._project_to_pixel(pc)
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
        centroid = pc_xyz.mean(axis=0)
        centered = pc_xyz - centroid
        scale = float(np.linalg.norm(centered, axis=1).max())
        scale = max(scale, 1e-6)
        return centered / scale, centroid, scale

    @torch.inference_mode()
    def predict(self, pc_cv: np.ndarray, depth: np.ndarray,
                bbox: tuple, confidence: float) -> Optional[Dict[str, Any]]:
        if pc_cv is None or len(pc_cv) < 10:
            return None
        frustum = self._crop_frustum(pc_cv, bbox)
        if len(frustum) < 8:
            return None
        sampled = self._sample_to_fixed(frustum, N_POINTS_FRUSTUM)
        xyz_norm, centroid, scale = self._normalize_pc(sampled)

        x = torch.from_numpy(xyz_norm).unsqueeze(0).to(self.device)
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
