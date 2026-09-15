#!/usr/bin/env python3
"""Common interface for the three post-YOLO 3D estimation methods.

Every method implements:
    predict(pc, depth, bbox, conf) -> dict ou None
        return:
            'is_drone_prob': float (1.0 for the direct-depth method)
            'is_drone': bool
            'center_cv': np.ndarray (3,) -- Cartesian position in the CV frame
            'size_cv': np.ndarray (3,) -- extents in the CV frame
            'method': str (method id, for logging)
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np


IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2


class Detector3D:
    """Interface base — subclasses implementam predict()."""

    name: str = "base"

    def predict(self, pc_cv: np.ndarray, depth: np.ndarray,
                bbox: tuple, confidence: float) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
