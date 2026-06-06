#!/usr/bin/env python3
"""Recalcula labels_3d para dataset_*_v4 usando bbox manual + origin compensation.

PCs raw + metadata existentes. Substituir só labels_3d/*.json.

Frame de saída: CV camera (x=right, y=down, z=fwd) — mesmo que PCs raw.
"""
import json
import math
import numpy as np
from pathlib import Path
import argparse

# ─── Constantes ───────────────────────────────────────────────────────────────
VEHICLE_ORIGINS = {
    'Ego': (0.0, 0.0, -5.0),
    'Drone3': (15.0, -5.0, -5.0),
    'Drone4': (15.0, 5.0, -5.0),
    'Intruder1': (-15.0, 0.0, -5.0),
}
CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = np.radians(-15.0)
# Quadrotor1 4x: AABB body NED real (3.01, 3.93, 2.79) × 1.38 margem
DRONE_HALF = np.array([3.01, 3.93, 2.79]) * 1.38 / 2

FOV_H_DEG = 90.0
FOV_V_DEG = 2 * math.degrees(math.atan(math.tan(math.radians(FOV_H_DEG / 2)) * 720 / 1280))
MAX_DIST = 250.0


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def world_to_cv(p_world, ego_pos, ego_yaw):
    rel_world = np.array(p_world) - np.array(ego_pos)
    R_ego = Rz(ego_yaw)
    rel_body = R_ego.T @ rel_world
    rel_cam = rel_body - CAM_OFFSET_BODY
    a = -CAM_PITCH_RAD
    rc, rs = math.cos(a), math.sin(a)
    fwd = rc*rel_cam[0] + rs*rel_cam[2]
    right = rel_cam[1]
    down = -rs*rel_cam[0] + rc*rel_cam[2]
    return np.array([right, down, fwd], dtype=np.float32)


def make_corners_world(drone_pos, drone_yaw):
    local = np.array([
        [+DRONE_HALF[0], +DRONE_HALF[1], +DRONE_HALF[2]], [-DRONE_HALF[0], +DRONE_HALF[1], +DRONE_HALF[2]],
        [-DRONE_HALF[0], -DRONE_HALF[1], +DRONE_HALF[2]], [+DRONE_HALF[0], -DRONE_HALF[1], +DRONE_HALF[2]],
        [+DRONE_HALF[0], +DRONE_HALF[1], -DRONE_HALF[2]], [-DRONE_HALF[0], +DRONE_HALF[1], -DRONE_HALF[2]],
        [-DRONE_HALF[0], -DRONE_HALF[1], -DRONE_HALF[2]], [+DRONE_HALF[0], -DRONE_HALF[1], -DRONE_HALF[2]],
    ])
    R = Rz(drone_yaw)
    return (R @ local.T).T + np.array(drone_pos)


def in_fov(center_cv):
    """True se centro projetado cai dentro do FOV horizontal e vertical."""
    right, down, fwd = center_cv
    if fwd < 0.5 or fwd > MAX_DIST:
        return False
    az = math.degrees(math.atan2(right, fwd))
    el = math.degrees(math.atan2(down, fwd))
    return abs(az) <= FOV_H_DEG/2 * 0.95 and abs(el) <= FOV_V_DEG/2 * 0.95


def relabel_dataset(ds_dir: Path, verbose=False):
    meta_dir = ds_dir / 'metadata'
    out_dir = ds_dir / 'pointnet' / 'labels_3d'
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = 0; n_labels = 0; n_skip_fov = 0; n_skip_far = 0
    for meta_p in sorted(meta_dir.glob('*.json')):
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        ego_vehicle = np.array(meta['ego_xyz'], dtype=np.float64)
        ego_yaw = float(meta['ego_yaw_rad'])
        ego_origin = np.array(VEHICLE_ORIGINS['Ego'])
        ego_global = ego_vehicle + ego_origin

        labels = []
        for d in meta.get('drone_positions', []):
            name = d['name']
            xyz_yaw = d['xyz_yaw']
            d_vehicle = np.array(xyz_yaw[:3], dtype=np.float64)
            d_yaw = float(xyz_yaw[3])
            d_origin = np.array(VEHICLE_ORIGINS.get(name, (0.0, 0.0, 0.0)))
            d_global = d_vehicle + d_origin

            # Centro no frame CV
            center_cv = world_to_cv(d_global, ego_global, ego_yaw)
            dist = float(np.linalg.norm(center_cv))
            if dist > MAX_DIST:
                n_skip_far += 1
                continue
            if not in_fov(center_cv):
                n_skip_fov += 1
                continue
            # 8 corners no frame CV
            corners_world = make_corners_world(d_global, d_yaw)
            corners_cv = np.array([world_to_cv(c, ego_global, ego_yaw) for c in corners_world], dtype=np.float32)
            bmin = corners_cv.min(axis=0)
            bmax = corners_cv.max(axis=0)
            labels.append({
                'name': name,
                'center': [float(x) for x in center_cv.tolist()],
                'box3D_min': [float(x) for x in bmin.tolist()],
                'box3D_max': [float(x) for x in bmax.tolist()],
                'corners': [[float(x) for x in c.tolist()] for c in corners_cv],
                'distance_m': dist,
                'drone_yaw_rad': d_yaw,
            })
            n_labels += 1
        out_p = out_dir / f"{meta_p.stem}.json"
        out_p.write_text(json.dumps(labels, indent=2))
        n_frames += 1
        if verbose and n_frames <= 3:
            print(f"  {meta_p.stem}: {len(labels)} labels")
    print(f"{ds_dir.name}: {n_frames} frames, {n_labels} labels (skip fov={n_skip_fov}, far={n_skip_far})")
    return n_frames, n_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['dataset_nh_v4', 'dataset_city_v4', 'dataset_coast_v4'])
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    for ds in args.datasets:
        p = Path(ds)
        if not p.exists():
            print(f"skip {ds}: not found")
            continue
        relabel_dataset(p, verbose=args.verbose)


if __name__ == '__main__':
    main()
