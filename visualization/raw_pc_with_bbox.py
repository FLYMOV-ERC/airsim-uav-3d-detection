#!/usr/bin/env python3
"""Write PLYs of the RAW (un-normalised) point cloud plus the 3D label box.

Mostra:
  - PC raw (frame CV: x=right, y=down, z=fwd) cinza
  - 3D label box WITHOUT the fix (airsim_to_cv applied directly) -- RED
  - 3D label box WITH the pitch + lever-arm fix -- GREEN
  - Esfera no center label

Useful for debugging the box-versus-cloud offset.
"""
import json
import numpy as np
from pathlib import Path
import sys

CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = np.radians(-15.0)


def airsim_to_cv(p):
    return np.array([p[1], p[2], p[0]], dtype=np.float32)


def airsim_body_to_cv_camera(p_body):
    p = np.array(p_body, dtype=np.float32) - CAM_OFFSET_BODY
    a = -CAM_PITCH_RAD
    c, s = np.cos(a), np.sin(a)
    x_cam_fwd = c * p[0] + s * p[2]
    y_cam_right = p[1]
    z_cam_down = -s * p[0] + c * p[2]
    return np.array([y_cam_right, z_cam_down, x_cam_fwd], dtype=np.float32)


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


def make_wireframe(center, size, color, pts_per_edge=200, min_size=0.2):
    cx, cy, cz = center
    sx = max(size[0], min_size); sy = max(size[1], min_size); sz = max(size[2], min_size)
    corners = np.array([
        [cx-sx/2, cy-sy/2, cz-sz/2], [cx+sx/2, cy-sy/2, cz-sz/2],
        [cx+sx/2, cy+sy/2, cz-sz/2], [cx-sx/2, cy+sy/2, cz-sz/2],
        [cx-sx/2, cy-sy/2, cz+sz/2], [cx+sx/2, cy-sy/2, cz+sz/2],
        [cx+sx/2, cy+sy/2, cz+sz/2], [cx-sx/2, cy+sy/2, cz+sz/2],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0, 1, pts_per_edge)[:, None]
        pts.append(corners[a] * (1 - t) + corners[b] * t)
    pts = np.vstack(pts)
    cols = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))
    return pts, cols


def make_sphere(center, color, radius=0.3, n=500):
    pts = []
    for _ in range(n):
        v = np.random.randn(3); v = v / (np.linalg.norm(v) + 1e-9) * radius
        pts.append(np.array(center) + v)
    pts = np.array(pts)
    cols = np.tile(np.array(color, dtype=np.uint8), (n, 1))
    return pts, cols


def main():
    out_dir = Path('viz_raw_pc')
    out_dir.mkdir(exist_ok=True)

    # Five frames from each environment
    frames_to_check = []
    for env in ['nh', 'city', 'coast']:
        ds = f'dataset_{env}_v4'
        for fp in sorted((Path(ds)/'pointnet/labels_3d').glob('*.json'))[:3]:
            try:
                lbls = json.load(open(fp))
            except: continue
            if not lbls: continue
            frames_to_check.append((env, ds, fp.stem, lbls))

    print(f"Processando {len(frames_to_check)} frames")
    for env, ds, frame, lbls in frames_to_check:
        pc_p = Path(ds) / 'pointnet' / 'point_clouds' / f'{frame}.npy'
        if not pc_p.exists(): continue
        pc = np.load(pc_p).astype(np.float32)
        # The point cloud is already in the CV (camera-tilted) frame
        # Subsample to keep the view responsive
        if len(pc) > 50000:
            idx = np.random.choice(len(pc), 50000, replace=False)
            pc = pc[idx]
        # Colour by depth (closer = lighter)
        z_min, z_max = pc[:,2].min(), pc[:,2].max()
        depth_norm = (pc[:,2] - z_min) / (z_max - z_min + 1e-6)
        pc_colors = (np.column_stack([
            120 + 100 * (1 - depth_norm),
            120 + 100 * (1 - depth_norm),
            150 + 80 * (1 - depth_norm),
        ]).clip(0, 255)).astype(np.uint8)

        all_pts = pc
        all_cols = pc_colors

        # For each drone in the frame
        for i, l in enumerate(lbls):
            center_body = l['center']
            # WITHOUT the fix (red)
            center_v1 = airsim_to_cv(center_body)
            if 'box3D_min' in l:
                bmin_v1 = airsim_to_cv(l['box3D_min'])
                bmax_v1 = airsim_to_cv(l['box3D_max'])
                size_v1 = np.abs(bmax_v1 - bmin_v1)
            else:
                size_v1 = np.array([2, 2, 2])
            # WITH the fix (green)
            center_v2 = airsim_body_to_cv_camera(center_body)
            if 'box3D_min' in l:
                bmin_v2 = airsim_body_to_cv_camera(l['box3D_min'])
                bmax_v2 = airsim_body_to_cv_camera(l['box3D_max'])
                size_v2 = np.abs(bmax_v2 - bmin_v2)
            else:
                size_v2 = np.array([2, 2, 2])

            # WITHOUT the fix -- red
            wp, wc = make_wireframe(center_v1, size_v1, (255, 60, 60))
            sp, sc = make_sphere(center_v1, (255, 0, 0), radius=0.4)
            all_pts = np.vstack([all_pts, wp, sp])
            all_cols = np.vstack([all_cols, wc, sc])

            # WITH the fix -- green
            wp, wc = make_wireframe(center_v2, size_v2, (60, 255, 60))
            sp, sc = make_sphere(center_v2, (0, 255, 0), radius=0.4)
            all_pts = np.vstack([all_pts, wp, sp])
            all_cols = np.vstack([all_cols, wc, sc])

            print(f"  {env}/{frame}/{l['name']}: dist={l['distance_m']:.1f}m  "
                  f"center_raw_body={[round(x,2) for x in center_body]}  "
                  f"v1 (no fix) center_cv={center_v1.round(2).tolist()}  "
                  f"v2 (with fix) center_cv={center_v2.round(2).tolist()}  "
                  f"diff_v1-v2 Y={center_v1[1]-center_v2[1]:.2f}m Z={center_v1[2]-center_v2[2]:.2f}m")

        out_p = out_dir / f"{env}_{frame}.ply"
        write_ply(out_p, all_pts, all_cols)

    print(f"\nDone. PLYs em {out_dir.absolute()}")
    print("Legend: RED = no fix (rel_pose used directly), GREEN = with the pitch + offset fix")


if __name__ == "__main__":
    main()
