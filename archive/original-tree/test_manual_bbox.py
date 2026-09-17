#!/usr/bin/env python3
"""Teste: bbox 3D MANUAL a partir de pose drone + extent Quadrotor1 4x conhecido.

Compara com bbox AirSim no MESMO PC. Vê visualmente qual bate.
"""
import json
import numpy as np
from pathlib import Path

CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = np.radians(-15.0)

# Quadrotor1 4x extents (medido empiricamente via simSpawnObject):
# body NED: X=fwd=0.48m, Y=right=5.45m, Z=down=1.49m
DRONE_HALF_X = 0.482 / 2
DRONE_HALF_Y = 5.455 / 2
DRONE_HALF_Z = 1.491 / 2


def quat_to_rot_yaw_only(yaw_rad):
    """Aproximação: drone hovering com roll=pitch=0."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def airsim_body_to_cv_camera(p_body):
    """body NED (fwd, right, down) → CV camera (right, down, fwd) com pitch+offset."""
    p = np.array(p_body, dtype=np.float64) - CAM_OFFSET_BODY
    a = -CAM_PITCH_RAD  # rotate body→cam = R_y(+15°)
    c, s = np.cos(a), np.sin(a)
    x_cam_fwd = c * p[0] + s * p[2]
    y_cam_right = p[1]
    z_cam_down = -s * p[0] + c * p[2]
    return np.array([y_cam_right, z_cam_down, x_cam_fwd])


def write_ply(path, points, colors):
    n = len(points)
    with open(path, 'wb') as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode('ascii'))
        for p, c in zip(points, colors):
            f.write(np.array(p, dtype=np.float32).tobytes())
            f.write(np.array(c, dtype=np.uint8).tobytes())


def make_wireframe(corners, color, pts_per_edge=200):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0, 1, pts_per_edge)[:, None]
        pts.append(corners[a] * (1 - t) + corners[b] * t)
    pts = np.vstack(pts)
    cols = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))
    return pts, cols


def make_bbox_corners_world(drone_pos, drone_yaw):
    """8 corners do Quadrotor1 4x no frame mundo NED."""
    # Local corners (body frame: x=fwd, y=right, z=down)
    local = np.array([
        [+DRONE_HALF_X, +DRONE_HALF_Y, +DRONE_HALF_Z],
        [-DRONE_HALF_X, +DRONE_HALF_Y, +DRONE_HALF_Z],
        [-DRONE_HALF_X, -DRONE_HALF_Y, +DRONE_HALF_Z],
        [+DRONE_HALF_X, -DRONE_HALF_Y, +DRONE_HALF_Z],
        [+DRONE_HALF_X, +DRONE_HALF_Y, -DRONE_HALF_Z],
        [-DRONE_HALF_X, +DRONE_HALF_Y, -DRONE_HALF_Z],
        [-DRONE_HALF_X, -DRONE_HALF_Y, -DRONE_HALF_Z],
        [+DRONE_HALF_X, -DRONE_HALF_Y, -DRONE_HALF_Z],
    ])
    R = quat_to_rot_yaw_only(drone_yaw)
    return (R @ local.T).T + np.array(drone_pos)


def world_to_cv_camera(p_world, ego_pos, ego_yaw):
    """world NED → CV camera frame considerando ego pose + camera offset + pitch."""
    R_ego = quat_to_rot_yaw_only(ego_yaw)
    # ponto relativo ao ego body
    rel_body = R_ego.T @ (np.array(p_world) - np.array(ego_pos))
    # aplica cam offset (subtraindo) e pitch
    return airsim_body_to_cv_camera(rel_body)


def main():
    out_dir = Path('viz_manual_bbox')
    out_dir.mkdir(exist_ok=True)

    # 5 samples de cada env (frames com metadata)
    samples = []
    for env, ds in [('nh', 'dataset_nh_v4'), ('city', 'dataset_city_v4'), ('coast', 'dataset_coast_v4')]:
        for fp in sorted((Path(ds)/'metadata').glob('*.json'))[:4]:
            meta = json.load(open(fp))
            lbl_p = Path(ds) / 'pointnet' / 'labels_3d' / f"{fp.stem}.json"
            pc_p = Path(ds) / 'pointnet' / 'point_clouds' / f"{fp.stem}.npy"
            if not lbl_p.exists() or not pc_p.exists(): continue
            labels = json.load(open(lbl_p))
            if not labels: continue
            samples.append((env, ds, fp.stem, meta, labels))

    print(f"{len(samples)} frames analisados")
    for env, ds, frame, meta, labels in samples[:10]:
        pc = np.load(Path(ds) / 'pointnet' / 'point_clouds' / f"{frame}.npy").astype(np.float32)
        if len(pc) > 50000:
            idx = np.random.choice(len(pc), 50000, replace=False)
            pc = pc[idx]
        # Cores cinzas por depth
        z_min, z_max = pc[:,2].min(), pc[:,2].max()
        dn = (pc[:,2] - z_min) / (z_max - z_min + 1e-6)
        pc_cols = np.column_stack([150 + 80*(1-dn), 150 + 80*(1-dn), 180 + 60*(1-dn)]).clip(0,255).astype(np.uint8)

        all_pts = pc; all_cols = pc_cols

        ego_pos = np.array(meta['ego_xyz'])
        ego_yaw = meta['ego_yaw_rad']
        drone_map = {d['name']: d['xyz_yaw'] for d in meta['drone_positions']}

        for l in labels:
            name = l['name']
            if name not in drone_map: continue
            xyz_yaw = drone_map[name]
            drone_pos = xyz_yaw[:3]
            drone_yaw = xyz_yaw[3]

            # BBOX MANUAL (calculada de pose + extent conhecido)
            corners_world = make_bbox_corners_world(drone_pos, drone_yaw)
            corners_cv = np.array([world_to_cv_camera(c, ego_pos, ego_yaw) for c in corners_world])
            wp, wc = make_wireframe(corners_cv, color=(0, 200, 255))  # AZUL CLARO
            all_pts = np.vstack([all_pts, wp])
            all_cols = np.vstack([all_cols, wc])
            # marcador no centro (mundo→cv)
            center_cv = world_to_cv_camera(drone_pos, ego_pos, ego_yaw)
            # esfera
            sph = []
            for _ in range(300):
                v = np.random.randn(3); v = v/(np.linalg.norm(v)+1e-9)*0.3
                sph.append(center_cv + v)
            sph = np.array(sph)
            all_pts = np.vstack([all_pts, sph])
            all_cols = np.vstack([all_cols, np.tile([0, 200, 255], (300, 1)).astype(np.uint8)])

            # BBOX AIRSIM (do label) — VERMELHO
            from viz_raw_pc_with_bbox import airsim_to_cv, make_wireframe as mw_v
            center_air = airsim_to_cv(l['center'])
            if 'box3D_min' in l:
                bmin = airsim_to_cv(l['box3D_min'])
                bmax = airsim_to_cv(l['box3D_max'])
                size_air = np.abs(bmax - bmin)
            else:
                size_air = np.array([2,2,2])
            wp, wc = mw_v(center_air, size_air, (255, 60, 60))
            all_pts = np.vstack([all_pts, wp])
            all_cols = np.vstack([all_cols, wc])
            print(f"  {env}/{frame}/{name}: dist={l['distance_m']:.1f}m")
            print(f"    manual center_cv: {center_cv.round(2).tolist()}")
            print(f"    airsim center_cv: {center_air.round(2).tolist()}  size={size_air.round(2).tolist()}")
            print(f"    diff manual-air center: {(center_cv - center_air).round(2).tolist()}m")

        out_p = out_dir / f"{env}_{frame}.ply"
        write_ply(out_p, all_pts, all_cols)

    print(f"\nDone. PLYs em {out_dir.absolute()}")
    print("Legend: VERMELHA=AirSim box3D  AZUL=MANUAL (pose + extent conhecido)")


if __name__ == "__main__":
    main()
