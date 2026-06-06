#!/usr/bin/env python3
"""Wrapper de inferência PointNet++ Painted v4.

Para cada bbox YOLO:
  paint full PC com prob=conf dentro do bbox, 0 fora
  → sample fixo N=4096
  → normaliza XYZ (centroide + escala)
  → forward na rede
  → desnormaliza center + size
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

from train_pointnet2_painted import PointNet2Painted


IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2
N_POINTS = 4096


class PointNetInference:
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
        print(f"[PointNet] {checkpoint_path}  device={self.device}")
        if isinstance(ckpt, dict) and "val_f1" in ckpt:
            print(f"           val_f1={ckpt['val_f1']:.3f}  epoch={ckpt.get('epoch', '?')}")

    @staticmethod
    def _project_to_pixel(pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = pc[:, 2]
        valid = z > 0.1
        z_safe = np.where(valid, z, 1.0)
        u = pc[:, 0] * FX / z_safe + CX
        v = pc[:, 1] * FY / z_safe + CY
        return u, v, valid

    @staticmethod
    def paint(pc: np.ndarray, bbox: Tuple[float, float, float, float],
              confidence: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        u, v, valid_z = PointNetInference._project_to_pixel(pc)
        inside = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        prob = np.where(inside, confidence, 0.0).astype(np.float32)
        return np.concatenate([pc, prob[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def _sample_to_fixed(pts: np.ndarray, n: int,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if len(pts) >= n:
            idx = rng.choice(len(pts), n, replace=False)
        else:
            idx = rng.choice(len(pts), n, replace=True)
        return pts[idx]

    @staticmethod
    def _normalize_xyz(pc_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        centroid = pc_xyz.mean(axis=0)
        centered = pc_xyz - centroid
        scale = float(np.linalg.norm(centered, axis=1).max())
        scale = max(scale, 1e-6)
        return centered / scale, centroid, scale

    @torch.inference_mode()
    def predict(self,
                pc: np.ndarray,
                bbox: Tuple[float, float, float, float],
                confidence: float,
                rng: Optional[np.random.Generator] = None
                ) -> Optional[Dict[str, Any]]:
        """
        pc: (N, 3) no frame CV (X=right, Y=down, Z=forward)
        bbox: (x1, y1, x2, y2) px
        confidence: 0..1 (YOLO)
        return: dict ou None se PC vazio
        """
        if pc is None or len(pc) < 100:
            return None
        painted = self.paint(pc, bbox, confidence)
        sampled = self._sample_to_fixed(painted, N_POINTS, rng)
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
        }
