#!/usr/bin/env python3
"""
Converte frustum point clouds (.npy) → .ply pra visualização no CloudCompare.
Inclui bbox 3D do drone como wireframe vermelho + centro verde.

Uso:
    python3 convert_frustum_to_ply.py [--limit 10]
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def write_ply(path, points, colors):
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    rec = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rec["x"] = points[:, 0]; rec["y"] = points[:, 1]; rec["z"] = points[:, 2]
    rec["r"] = colors[:, 0]; rec["g"] = colors[:, 1]; rec["b"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def edges_of_box(bmin, bmax, n=12):
    """12 edges of axis-aligned box, sampled with n points each."""
    pts = []
    for axis in range(3):
        for i in (0, 1):
            for j in (0, 1):
                for k in range(n):
                    t = k / max(1, n - 1)
                    p = [0.0, 0.0, 0.0]
                    p[axis] = bmin[axis] + t * (bmax[axis] - bmin[axis])
                    other = [a for a in range(3) if a != axis]
                    p[other[0]] = bmin[other[0]] if i == 0 else bmax[other[0]]
                    p[other[1]] = bmin[other[1]] if j == 0 else bmax[other[1]]
                    pts.append(p)
    return np.array(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_frustum")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / "ply"
    out_dir.mkdir(exist_ok=True)

    files = sorted((root / "point_clouds").glob("*.npy"))[:args.limit]
    print(f"Convertendo {len(files)} frustums:")
    for npy_path in files:
        name = npy_path.stem
        pc = np.load(npy_path).astype(np.float32)
        # Pontos do frustum em CINZA
        scene_colors = np.full((len(pc), 3), 150, dtype=np.uint8)
        all_pts = [pc]
        all_cols = [scene_colors]

        # Lê label pra adicionar bbox 3D + centro
        lbl_path = root / "labels" / f"{name}.json"
        info = ""
        if lbl_path.exists():
            d = json.load(open(lbl_path))
            info = f"  ({d['drone_name']} @ {d.get('distance_m', 0):.1f}m, {d['n_points']} pts)"
            if "box3D_min_cv" in d and "box3D_max_cv" in d:
                edges = edges_of_box(d["box3D_min_cv"], d["box3D_max_cv"])
                all_pts.append(edges)
                all_cols.append(np.full((len(edges), 3), [255, 0, 0], dtype=np.uint8))
            if "center_cv" in d:
                c = np.array([d["center_cv"]], dtype=np.float32)
                all_pts.append(c)
                all_cols.append(np.array([[0, 255, 0]], dtype=np.uint8))

        all_pts = np.concatenate(all_pts, axis=0)
        all_cols = np.concatenate(all_cols, axis=0)
        out_path = out_dir / f"{name}.ply"
        write_ply(out_path, all_pts, all_cols)
        print(f"  {name}.ply{info}")

    print(f"\nOutput: {out_dir}")
    print(f"Caminho Windows: \\\\wsl$\\Ubuntu\\home\\ericyos\\airsim\\{out_dir}")


if __name__ == "__main__":
    main()
