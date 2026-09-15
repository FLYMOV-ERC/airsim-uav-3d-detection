#!/usr/bin/env python3
"""ARCHIVED. The plain-PointNet frustum detector (N=512, global max-pool).

Pipeline:
  1. Crop the frustum: the cloud points inside the projected 2D box
  2. Resample to a fixed N=512
  3. Normalise XYZ (centroid + scale)
  4. Forward pass through the plain PointNet
  5. Denormalise: center_cv = center_rel * scale + centroid

This is the "plain PointNet (classification baseline)" of Section 7.3.4, whose
accuracy of 0.97 and normalized centre error of about 3.6 ARE reported in the
dissertation. Provenance-critical.
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root

from typing import Optional, Dict, Any
import numpy as np
import torch

from detection.base import Detector3D, IMAGE_W, IMAGE_H, FX, FY, CX, CY
from archive.early_detectors.train_pointnet import PointNetDetector


N_POINTS_FRUSTUM = 512


class FrustumPointNetDetector(Detector3D):
    name = "frustum_pointnet"

    def __init__(self,
                 checkpoint_path: str = "runs/pointnet/drone_detector/best.pt",
                 device: Optional[str] = None,
                 cls_threshold: float = 0.5,
                 bbox_pad_pct: float = 0.0):
        """
        bbox_pad_pct: percentage padding applied to the box before extracting the frustum.
            0    = no padding (the original YOLO box)
            100  = duplica W e H
            300  = quadruples W and H (needed to recover the training regime)

            Motivation: dataset_pointnet was built from simGetDetections boxes
            (generous, with margin), so the frustum captures drone plus background and the scale is ~50 m.
            YOLO gives a tight box, so the frustum contains only the drone and the scale is ~0.4 m.
            The model never saw such a small scale and fails; padding restores the distribution.
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
