#!/usr/bin/env python3
"""Visualiza PC raw + bbox labels_3d de dataset_v6 pra confirmar match."""
import json
import numpy as np
from pathlib import Path
import sys
ds = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/test_v6_nh')
out = Path('viz_v6_check')
out.mkdir(exist_ok=True)


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


def make_wf(corners, color, n=200):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0, 1, n)[:, None]
        pts.append(corners[a] * (1 - t) + corners[b] * t)
    pts = np.vstack(pts)
    return pts, np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))


def make_sphere(c0, color, r=0.3, n=400):
    pts = np.array([c0 + (lambda v: v/(np.linalg.norm(v)+1e-9)*r)(np.random.randn(3)) for _ in range(n)])
    return pts, np.tile(np.array(color, dtype=np.uint8), (n, 1))


for label_p in sorted((ds/'pointnet/labels_3d').glob('*.json'))[:5]:
    pc_p = ds/'pointnet/point_clouds'/f'{label_p.stem}.npy'
    if not pc_p.exists(): continue
    pc = np.load(pc_p).astype(np.float32)
    labels = json.loads(label_p.read_text())
    if not labels: continue

    if len(pc) > 80000:
        idx = np.random.choice(len(pc), 80000, replace=False); pc = pc[idx]
    z_min, z_max = pc[:,2].min(), pc[:,2].max()
    dn = (pc[:,2]-z_min)/(z_max-z_min+1e-6)
    pc_cols = np.column_stack([150+80*(1-dn), 150+80*(1-dn), 180+60*(1-dn)]).clip(0,255).astype(np.uint8)

    all_pts = pc; all_cols = pc_cols
    for l in labels:
        corners = np.array(l['corners'])
        wp, wc = make_wf(corners, (0, 100, 255))
        all_pts = np.vstack([all_pts, wp]); all_cols = np.vstack([all_cols, wc])
        sp, sc = make_sphere(np.array(l['center']), (255, 200, 0))
        all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])
    write_ply(out/f'{label_p.stem}.ply', all_pts, all_cols)
    print(f'{label_p.stem}: {len(labels)} drones')
print(f'\nPLY: {out.absolute()}')
