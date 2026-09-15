#!/usr/bin/env python3
"""Camera model and frame conversions -- the geometric core of the pipeline.

This module is deliberately dependency-light: NumPy plus the rotation helper of
``tracking.ekf_3d``.  It imports neither AirSim nor Ultralytics, so the voxel
detectors, the dataset builders and the evaluation code can use the geometry
without pulling in a simulator client or a YOLO runtime.

Frames (Chapter 2 of the dissertation, reused verbatim here):

* **NED world** -- AirSim's own convention, x North, y East, z Down.
* **Body FRD** -- x forward, y right, z down, attached to the ego platform.
* **Optical / CV** -- x right, y down, z forward, the frame the point clouds and
  the detector outputs live in.  Denoted :math:`\\mathcal{O}` in the chapter.

The camera is an ideal pinhole at 1280x720 with a 90-degree horizontal field of
view, giving ``fx = fy = 640`` px and a principal point at ``(640, 360)`` px, and
it is mounted 0.35 m forward and 0.5 m below the body origin with a fixed -15
degree pitch.  Those numbers come from ``configs/settings.json`` and are read
here through ``common.config`` so the two cannot drift apart.

Section 7.2.1 of the dissertation states the same model.
"""
from __future__ import annotations

import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import CONFIG
from tracking.ekf_3d import rotation_matrix

# ---------------------------------------------------------------------------
# Camera model -- mirrors the front_center block of configs/settings.json
# ---------------------------------------------------------------------------
IMAGE_W = int(CONFIG.get("camera.width", 1280))
IMAGE_H = int(CONFIG.get("camera.height", 720))
FOV_DEG = float(CONFIG.get("camera.fov_degrees", 90))

FX = IMAGE_W / (2 * np.tan(np.radians(FOV_DEG / 2)))
FY = FX                                   # square pixels, confirmed by calibration
CX, CY = IMAGE_W / 2, IMAGE_H / 2

CAMERA = "front_center"
#: Fixed downward tilt of the camera, radians (settings.json: "Pitch": -15).
CAMERA_PITCH_OFFSET = float(np.radians(CONFIG.get("camera.pitch_degrees", -15.0)))
#: Lever arm body origin -> optical centre, body FRD metres.
CAM_OFFSET_BODY = np.asarray(CONFIG.get("camera.offset_body", [0.35, 0.0, -0.5]), dtype=np.float64)
MIN_DEPTH = float(CONFIG.get("camera.min_depth", 0.1))
MAX_DEPTH = float(CONFIG.get("camera.max_depth", 250.0))


def quat_to_rpy(q) -> Tuple[float, float, float]:
    """Quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians, Tait-Bryan Z-Y-X."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = float(np.arctan2(sinr, cosr))
    sinp = 2 * (w * y - z * x)
    pitch = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = float(np.arctan2(siny, cosy))
    return roll, pitch, yaw


def airsim_pose_to_x_plat(pose, velocity_vec=None) -> np.ndarray:
    """AirSim pose + velocity -> platform state ``x_plat`` (9,).

    Returns ``[x, y, z, vx, vy, vz, roll, pitch, yaw]`` in the NED world frame,
    with the fixed camera tilt folded into the pitch so that downstream code can
    treat ``x_plat`` as the pose of the *optical* frame.
    """
    if velocity_vec is None:
        velocity_vec = (0.0, 0.0, 0.0)
    roll, pitch, yaw = quat_to_rpy(pose.orientation)
    pitch_eff = pitch + CAMERA_PITCH_OFFSET
    return np.array([
        pose.position.x_val, pose.position.y_val, pose.position.z_val,
        velocity_vec[0], velocity_vec[1], velocity_vec[2],
        roll, pitch_eff, yaw,
    ], dtype=np.float64)


def depth_to_pointcloud_cv(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """Planar depth image (H, W) -> point cloud (N, 3) in the optical/CV frame.

    Inverse pinhole map of Section 7.2.1: a pixel ``(u, v)`` with planar depth
    ``z`` becomes ``[(u - cx) z / fx, (v - cy) z / fy, z]``.  Returns outside
    ``[MIN_DEPTH, max_depth)`` are discarded.
    """
    h, w = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > MIN_DEPTH) & (depth < max_depth)
    z = depth[valid]
    u = u_grid[valid].astype(np.float32)
    v = v_grid[valid].astype(np.float32)
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def cv_to_frd(p_cv: np.ndarray) -> np.ndarray:
    """Optical/CV ``[right, down, forward]`` -> body FRD ``[forward, right, down]``."""
    return np.array([p_cv[2], p_cv[0], p_cv[1]])


def frd_to_spherical(p_frd: np.ndarray) -> Tuple[float, float, float]:
    """Body FRD position -> ``(range, azimuth, elevation)``, the (rho, alpha, beta)
    of Eq. (2.20) of the dissertation -- "Range-Bearing (Spherical) Measurements",
    Section 2.5.1 -- which is what the Chapter 5 EKF ingests.  Angles in radians."""
    x, y, z = p_frd
    d = float(np.sqrt(x * x + y * y + z * z))
    phi = float(np.arctan2(y, x))
    theta = float(np.arctan2(z, np.sqrt(x * x + y * y)))
    return d, phi, theta


def project_global_to_pixel(p_global: np.ndarray, x_plat: np.ndarray):
    """Project a global NED position to a pixel ``(u, v)`` plus depth.

    Returns ``(None, None, None)`` when the point falls behind the image plane.
    Note the transpose: world -> camera is ``R^T``, not ``R`` (the sign error that
    Section 1.3 of docs/methodology.md records).
    """
    R_m = rotation_matrix(x_plat[6], x_plat[7], x_plat[8])
    p_frd = R_m.T @ (p_global - x_plat[:3])   # world -> camera is R^T, not R
    x_frd, y_frd, z_frd = p_frd
    if x_frd <= 0.1:
        return None, None, None
    u = y_frd * FX / x_frd + CX
    v = z_frd * FY / x_frd + CY
    return int(u), int(v), float(x_frd)


