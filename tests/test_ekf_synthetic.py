#!/usr/bin/env python3
"""Validate EKFTracker3D on a synthetic trajectory.

Scenario:
  - target with roughly constant velocity plus small random acceleration
  - moving platform with an oscillating roll-pitch-yaw pose
  - measurements with Gaussian noise in spherical space (d, phi, theta, r)
Criterion: steady-state position error below 2 m.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root

import sys
import numpy as np

from tracking.ekf_3d import (
    EKFTracker3D, EKFParams,
    rotation_matrix, spherical_from_cartesian,
)


def simulate(n_steps: int = 200, dt: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)

    obj = np.zeros((7, n_steps))
    obj[:, 0] = [50.0, 30.0, -20.0, 2.0, -1.0, -0.5, 1.0]
    F = np.eye(7)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    for k in range(1, n_steps):
        x = F @ obj[:, k - 1]
        x[3:6] += rng.normal(0, 0.05, 3)
        obj[:, k] = x

    plat = np.zeros((9, n_steps))
    for k in range(n_steps):
        t = k * dt
        plat[0, k] = 0.5 * t
        plat[1, k] = 2.0 * np.sin(0.2 * t)
        plat[2, k] = -30.0 + 0.5 * np.sin(0.5 * t)
        plat[3, k] = 0.5
        plat[4, k] = 0.4 * np.cos(0.2 * t)
        plat[5, k] = 0.25 * np.cos(0.5 * t)
        plat[6, k] = 0.05 * np.sin(0.3 * t)            # roll
        plat[7, k] = np.radians(-15) + 0.05 * np.sin(0.2 * t)  # pitch (the -15 deg camera tilt)
        plat[8, k] = 0.1 * t                            # yaw

    sigma_phi, sigma_theta, sigma_r = 0.005, 0.005, 0.5
    alpha_d, sigma_d_min = 0.05, 0.5
    meas = np.zeros((4, n_steps))
    for k in range(n_steps):
        R_m = rotation_matrix(plat[6, k], plat[7, k], plat[8, k])
        p_c = R_m @ (obj[0:3, k] - plat[0:3, k])
        d, phi, theta = spherical_from_cartesian(p_c)
        r = obj[6, k]
        sigma_d_k = max(alpha_d * d, sigma_d_min)
        meas[0, k] = d + rng.normal(0, sigma_d_k)
        meas[1, k] = phi + rng.normal(0, sigma_phi)
        meas[2, k] = theta + rng.normal(0, sigma_theta)
        meas[3, k] = r + rng.normal(0, sigma_r)
    return obj, plat, meas


def main():
    n_steps = 200
    dt = 0.1
    obj, plat, meas = simulate(n_steps, dt)

    tracker = EKFTracker3D(meas[:, 0], plat[:, 0], params=EKFParams())
    est = np.zeros((7, n_steps))
    est[:, 0] = tracker.x.copy()
    pos_err = np.zeros(n_steps)
    pos_err[0] = float(np.linalg.norm(tracker.x[:3] - obj[:3, 0]))

    for k in range(1, n_steps):
        tracker.predict(dt)
        tracker.update(meas[:, k], plat[:, k])
        est[:, k] = tracker.x.copy()
        pos_err[k] = float(np.linalg.norm(tracker.x[:3] - obj[:3, k]))

    vel_err = np.linalg.norm(est[3:6, :] - obj[3:6, :], axis=0)
    rad_err = np.abs(est[6, :] - obj[6, :])

    print("\n=== EKF Validation (synthetic trajectory) ===")
    print(f"Frames: {n_steps}  dt={dt}s  duration={n_steps*dt}s")
    print(f"Position error (m):")
    print(f"  first 10 frames: mean={pos_err[:10].mean():.2f}  max={pos_err[:10].max():.2f}")
    print(f"  frames 10-50:    mean={pos_err[10:50].mean():.2f}  max={pos_err[10:50].max():.2f}")
    print(f"  last 100 (SS):   mean={pos_err[100:].mean():.2f}  max={pos_err[100:].max():.2f}")
    print(f"Velocity error (m/s) last 100: mean={vel_err[100:].mean():.2f}  max={vel_err[100:].max():.2f}")
    print(f"Radius error (m)   last 100: mean={rad_err[100:].mean():.2f}  max={rad_err[100:].max():.2f}")
    print(f"Final cov diag: {tracker.covariance_diag()}")

    ok = pos_err[100:].mean() < 2.0
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] steady-state position error < 2m: {pos_err[100:].mean():.2f}m")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
