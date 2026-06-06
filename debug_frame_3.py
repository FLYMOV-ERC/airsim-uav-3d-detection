#!/usr/bin/env python3
"""Debug: gera PLY do frame_000003 marcando pts dentro de cada bbox."""
import json, numpy as np
from pathlib import Path

ds = Path('/tmp/test_v6_nh')
labels = json.load(open(ds/'pointnet/labels_3d/frame_000003.json'))
pc = np.load(ds/'pointnet/point_clouds/frame_000003.npy').astype(np.float32)
print(f'PC: {len(pc)} pts, n_labels: {len(labels)}')


def write_ply(path, pts, cols):
    n = len(pts)
    with open(path, 'wb') as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode('ascii'))
        for p, c in zip(pts, cols):
            f.write(np.array(p, dtype=np.float32).tobytes())
            f.write(np.array(c, dtype=np.uint8).tobytes())


def wf(corners, color, n=200):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0,1,n)[:,None]
        pts.append(corners[a]*(1-t)+corners[b]*t)
    pts = np.vstack(pts)
    return pts, np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))


def sph(c0, color, r=0.3, n=200):
    pts = np.array([c0 + (lambda v: v/(np.linalg.norm(v)+1e-9)*r)(np.random.randn(3)) for _ in range(n)])
    return pts, np.tile(np.array(color, dtype=np.uint8), (n, 1))


# Sub-sample PC
if len(pc) > 80000:
    idx = np.random.choice(len(pc), 80000, replace=False); pc_sub = pc[idx]
else:
    pc_sub = pc
# Cinza
z_min, z_max = pc_sub[:,2].min(), pc_sub[:,2].max()
dn = (pc_sub[:,2]-z_min)/(z_max-z_min+1e-6)
pc_cols = np.column_stack([150+80*(1-dn), 150+80*(1-dn), 180+60*(1-dn)]).clip(0,255).astype(np.uint8)

all_pts = pc_sub; all_cols = pc_cols

# Pra cada label: bbox + center + 30 pts mais próximos do center destacados
COLORS = [(0,100,255), (255,100,0)]
for i, l in enumerate(labels):
    color = COLORS[i % 2]
    corners = np.array(l['corners'])
    wp, wc = wf(corners, color)
    all_pts = np.vstack([all_pts, wp]); all_cols = np.vstack([all_cols, wc])
    center = np.array(l['center'])
    sp, sc = sph(center, (255, 255, 0), r=0.5)
    all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])

    # Pts dentro da AABB do bbox (em VERDE forte)
    bmin = np.array(l['box3D_min']); bmax = np.array(l['box3D_max'])
    inside_mask = ((pc[:,0]>=bmin[0])&(pc[:,0]<=bmax[0])&
                   (pc[:,1]>=bmin[1])&(pc[:,1]<=bmax[1])&
                   (pc[:,2]>=bmin[2])&(pc[:,2]<=bmax[2]))
    pts_inside = pc[inside_mask]
    print(f'L{i} {l["name"]}: bbox={(bmax-bmin).round(2).tolist()} center={center.round(2).tolist()} '
          f'n_inside={len(pts_inside)}')
    if len(pts_inside) > 0:
        cols_in = np.tile(np.array([0, 255, 0], dtype=np.uint8), (len(pts_inside), 1))
        all_pts = np.vstack([all_pts, pts_inside]); all_cols = np.vstack([all_cols, cols_in])

    # Pts mais próximos do center (50, em MAGENTA)
    dists = np.linalg.norm(pc - center, axis=1)
    nearest_idx = np.argsort(dists)[:50]
    pts_near = pc[nearest_idx]
    near_dists = dists[nearest_idx]
    print(f'  Near 5 dists: {near_dists[:5].round(2).tolist()}')
    print(f'  Near 5 pts (x,y,z): \n' +
          '\n'.join([f'    {p.round(2).tolist()}' for p in pts_near[:5]]))
    cols_near = np.tile(np.array([255, 0, 255], dtype=np.uint8), (len(pts_near), 1))
    all_pts = np.vstack([all_pts, pts_near]); all_cols = np.vstack([all_cols, cols_near])

write_ply('/tmp/debug_f3.ply', all_pts, all_cols)
print('PLY: /tmp/debug_f3.ply')
