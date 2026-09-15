#!/usr/bin/env python3
"""ARCHIVED. Network-free 3D detector reading depth directly (no network after YOLO).

For each YOLO box:
  1. Take the box centre (u, v)
  2. Read the depth there (or the median of a 7x7 window around it)
  3. Back-project: x_cv = (u - CX) * d / FX, y_cv = (v - CY) * d / FY, z_cv = d
  4. Estimate a coarse size from the box in pixels plus the depth

A sensible null model, bounding the design space from below. Not reported in the
dissertation.
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root

from typing import Optional, Dict, Any
import numpy as np

from detection.base import Detector3D, IMAGE_W, IMAGE_H, FX, FY, CX, CY


class DepthDirectDetector(Detector3D):
    name = "depth_direct"

    def __init__(self, window: int = 7,
                 min_depth: float = 0.5, max_depth: float = 300.0):
        self.window = max(1, int(window) | 1)  # force an odd window
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

        # Window around the centre; the median is used for robustness
        u0 = max(0, u - self.half); u1 = min(IMAGE_W, u + self.half + 1)
        v0 = max(0, v - self.half); v1 = min(IMAGE_H, v + self.half + 1)
        window = depth[v0:v1, u0:u1]
        valid = (window > self.min_depth) & (window < self.max_depth)
        if valid.sum() < 4:
            # try a larger window
            w2 = 21
            u0 = max(0, u - w2); u1 = min(IMAGE_W, u + w2 + 1)
            v0 = max(0, v - w2); v1 = min(IMAGE_H, v + w2 + 1)
            window = depth[v0:v1, u0:u1]
            valid = (window > self.min_depth) & (window < self.max_depth)
            if valid.sum() < 4:
                return None

        # Median of the valid values
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
        size_z = max(size_x, size_y) * 0.6  # heuristic
        size_cv = np.array([size_x, size_y, size_z], dtype=np.float32)

        return {
            'is_drone_prob': 1.0,  # the direct-depth method does not classify
            'is_drone': True,
            'center_cv': center_cv,
            'size_cv': size_cv,
            'centroid_cv': center_cv.copy(),
            'scale': 1.0,
            'method': self.name,
        }
