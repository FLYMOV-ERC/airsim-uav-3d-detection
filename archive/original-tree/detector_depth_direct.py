#!/usr/bin/env python3
"""Detector 3D direto via depth (sem rede neural pós-YOLO).

Para cada bbox YOLO:
  1. Pega centro do bbox (u, v)
  2. Lê depth no centro (ou mediana de uma janela 7×7 ao redor)
  3. Backprojeta: x_cv = (u - CX) * d / FX, y_cv = (v - CY) * d / FY, z_cv = d
  4. Estima size grossier do bbox em pixels + depth

Resultado: estimativa precisa (~50cm de erro) sem rede neural.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from detector_base import Detector3D, IMAGE_W, IMAGE_H, FX, FY, CX, CY


class DepthDirectDetector(Detector3D):
    name = "depth_direct"

    def __init__(self, window: int = 7,
                 min_depth: float = 0.5, max_depth: float = 300.0):
        self.window = max(1, int(window) | 1)  # garante ímpar
        self.half = self.window // 2
        self.min_depth = min_depth
        self.max_depth = max_depth

    def predict(self, pc_cv: np.ndarray, depth: np.ndarray,
                bbox: tuple, confidence: float) -> Optional[Dict[str, Any]]:
        x1, y1, x2, y2 = bbox
        u = int(round((x1 + x2) / 2))
        v = int(round((y1 + y2) / 2))
        if not (0 <= u < IMAGE_W and 0 <= v < IMAGE_H):
            return None

        # Janela ao redor do centro, mediana para robustez
        u0 = max(0, u - self.half); u1 = min(IMAGE_W, u + self.half + 1)
        v0 = max(0, v - self.half); v1 = min(IMAGE_H, v + self.half + 1)
        window = depth[v0:v1, u0:u1]
        valid = (window > self.min_depth) & (window < self.max_depth)
        if valid.sum() < 4:
            # tenta janela maior
            w2 = 21
            u0 = max(0, u - w2); u1 = min(IMAGE_W, u + w2 + 1)
            v0 = max(0, v - w2); v1 = min(IMAGE_H, v + w2 + 1)
            window = depth[v0:v1, u0:u1]
            valid = (window > self.min_depth) & (window < self.max_depth)
            if valid.sum() < 4:
                return None

        # Mediana dos valores válidos
        d = float(np.median(window[valid]))

        # Backproject
        x_cv = (u - CX) * d / FX
        y_cv = (v - CY) * d / FY
        z_cv = d
        center_cv = np.array([x_cv, y_cv, z_cv], dtype=np.float32)

        # Size estimate: bbox em pixels → meters via depth
        bw_px = x2 - x1
        bh_px = y2 - y1
        size_x = bw_px * d / FX
        size_y = bh_px * d / FY
        size_z = max(size_x, size_y) * 0.6  # heurística
        size_cv = np.array([size_x, size_y, size_z], dtype=np.float32)

        return {
            'is_drone_prob': 1.0,  # depth direct não classifica
            'is_drone': True,
            'center_cv': center_cv,
            'size_cv': size_cv,
            'centroid_cv': center_cv.copy(),
            'scale': 1.0,
            'method': self.name,
        }
