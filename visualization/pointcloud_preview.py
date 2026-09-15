#!/usr/bin/env python3
"""Preview dataset samples as:
  - PLY (openable in MeshLab / CloudCompare)
  - PNG render 2D (matplotlib)

Output: viz_pc/<sample_id>.{ply,png}
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np


def write_ply(path, points_xyz, colors_rgb):
    """points_xyz: (N,3), colors_rgb: (N,3) uint8."""
    n = len(points_xyz)
    with open(path, 'wb') as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode('ascii'))
        for p, c in zip(points_xyz, colors_rgb):
            f.write(np.array([p[0], p[1], p[2]], dtype=np.float32).tobytes())
            f.write(np.array([c[0], c[1], c[2]], dtype=np.uint8).tobytes())


def make_bbox_3d_wireframe(center, size, color=(0, 255, 0), pts_per_edge=200,
                             min_size=0.05):
    """3D wireframe of the axis-aligned box. Clamps a minimum size so tiny boxes stay visible."""
    cx, cy, cz = center
    # Clamp the minimum size so the box stays visible
    sx = max(size[0], min_size)
    sy = max(size[1], min_size)
    sz = max(size[2], min_size)
    corners = np.array([
        [cx - sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy - sy/2, cz - sz/2],
        [cx + sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy + sy/2, cz - sz/2],
        [cx - sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy - sy/2, cz + sz/2],
        [cx + sx/2, cy + sy/2, cz + sz/2],
        [cx - sx/2, cy + sy/2, cz + sz/2],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0, 1, pts_per_edge)[:, None]
        pts.append(corners[a] * (1 - t) + corners[b] * t)
    pts = np.vstack(pts)
    cols = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))
    return pts, cols


def make_center_sphere(center, color=(255, 0, 0), radius=0.04, n=400):
    """Dense sphere of points at the drone centre."""
    pts = []
    for _ in range(n):
        v = np.random.randn(3)
        v = v / (np.linalg.norm(v) + 1e-9) * radius
        pts.append(np.array(center) + v)
    pts = np.array(pts)
    cols = np.tile(np.array(color, dtype=np.uint8), (n, 1))
    return pts, cols


def render_pc_2d(points_xyz, colors_rgb, title="", out_path="", center=None, size=None):
    """3 views: XY (top), XZ (front), YZ (side)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors_norm = colors_rgb.astype(np.float32) / 255.0
    views = [
        ('XY (top)', 0, 1),
        ('XZ (front)', 0, 2),
        ('YZ (side)', 1, 2),
    ]
    for ax, (name, i, j) in zip(axes, views):
        ax.scatter(points_xyz[:, i], points_xyz[:, j], c=colors_norm, s=2)
        if center is not None:
            ax.scatter([center[i]], [center[j]], c='lime', s=200, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        if size is not None and center is not None:
            from matplotlib.patches import Rectangle
            w = size[i]; h = size[j]
            ax.add_patch(Rectangle((center[i]-w/2, center[j]-h/2), w, h,
                                    fill=False, edgecolor='lime', linewidth=2))
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["painted", "frustum"], required=True)
    ap.add_argument("--n", type=int, default=10, help="quantos samples")
    ap.add_argument("--output", default="viz_pc")
    ap.add_argument("--only_pos", action="store_true",
                    help="positive samples only (is_drone=1)")
    args = ap.parse_args()

    if args.dataset == "painted":
        root = Path("dataset_painted_v5")
    else:
        root = Path("dataset_frustum_pn2_v2")

    out = Path(args.output) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    # Collect samples from train and val
    samples = []
    for split in ['train', 'val']:
        lbl_dir = root / split / 'labels'
        pc_dir = root / split / 'point_clouds'
        for lbl_p in lbl_dir.glob('*.json'):
            try:
                lbl = json.load(open(lbl_p))
            except: continue
            if args.only_pos and not lbl.get('is_drone'):
                continue
            pc_p = pc_dir / f'{lbl_p.stem}.npy'
            if pc_p.exists():
                samples.append((pc_p, lbl_p, lbl))

    random.seed(42)
    random.shuffle(samples)
    samples = samples[:args.n]
    print(f"Visualizando {len(samples)} samples de {args.dataset}")

    for pc_p, lbl_p, lbl in samples:
        try:
            pc = np.load(pc_p).astype(np.float32)
        except Exception as e:
            print(f"  skip {pc_p.name}: {e}")
            continue

        xyz = pc[:, :3]
        # Colours: grey by default; red where prob > 0 (painted)
        n = len(xyz)
        colors = np.full((n, 3), 180, dtype=np.uint8)  # cinza claro
        if pc.shape[1] >= 4:  # painted (xyz + prob)
            prob = pc[:, 3]
            inside = prob > 0
            colors[inside] = [220, 80, 80]   # red inside the YOLO box
            colors[~inside] = [180, 180, 180]
        # Otherwise (frustum): XYZ only, everything grey

        # Center label
        center = None
        size = None
        if lbl.get('is_drone'):
            center = np.array(lbl.get('center_rel_normalized', [0,0,0]))
            sz = lbl.get('size_normalized', [0,0,0])
            size = np.array(sz)

        sid = lbl_p.stem
        # Paint BLUE the cloud points near the label centre (drone candidates)
        if center is not None:
            d = np.linalg.norm(xyz - center, axis=1)
            near = d < 0.15  # raio em frame normalizado
            colors[near] = [0, 100, 255]   # blue = points near the label
        all_xyz = xyz
        all_colors = colors
        if center is not None:
            sph_pts, sph_cols = make_center_sphere(center, color=(255, 0, 0), radius=0.04)
            all_xyz = np.vstack([all_xyz, sph_pts])
            all_colors = np.vstack([all_colors, sph_cols])
            if size is not None and size.sum() > 0:
                bb_pts, bb_cols = make_bbox_3d_wireframe(center, size, color=(0, 255, 0))
                all_xyz = np.vstack([all_xyz, bb_pts])
                all_colors = np.vstack([all_colors, bb_cols])
        # PLY
        write_ply(out / f"{sid}.ply", all_xyz, all_colors)
        # PNG
        title = f"{sid}  is_drone={lbl.get('is_drone')}  yolo_conf={lbl.get('yolo_confidence', '?'):.2f}  iou={lbl.get('best_gt_iou', 0):.2f}"
        render_pc_2d(xyz, colors, title, str(out / f"{sid}.png"), center, size)

    print(f"Saved {len(samples)} samples to {out.absolute()}")
    print("Abra os PNGs ou os PLYs (MeshLab/CloudCompare).")


if __name__ == "__main__":
    main()
