#!/usr/bin/env python3
"""
Gera PLY com a PC INTEIRA do frame + pontos do frustum highlighted +
bbox 3D do drone. Útil pra validar visualmente que o frustum está
recortando a região correta.

Cores:
  - cinza claro: PC inteira (fora do frustum)
  - amarelo: pontos dentro do frustum
  - vermelho: wireframe bbox 3D drone
  - verde: centro drone
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2


def write_ply(path, points, colors):
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
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
        f.write(header); f.write(rec.tobytes())


def edges_of_box(bmin, bmax, n=12):
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
    out_dir = root / "ply_context"
    out_dir.mkdir(exist_ok=True)

    files = sorted((root / "labels").glob("*.json"))[:args.limit]
    print(f"Gerando {len(files)} PLYs com contexto:")
    for lbl_path in files:
        name = lbl_path.stem
        d = json.load(open(lbl_path))
        # Carrega PC original do frame (não o frustum recortado)
        src = d["source"]  # city, nh, coast
        frame = d["frame"]
        full_pc_path = Path(f"dataset_urban_{src}/pointnet/point_clouds/{frame}.npy")
        if not full_pc_path.exists():
            print(f"  SKIP {name}: PC original não encontrado")
            continue
        full_pc = np.load(full_pc_path).astype(np.float32)

        # Projeta cada ponto: dentro ou fora do bbox 2D?
        x1, y1, x2, y2 = d["bbox_2d"]
        z = full_pc[:, 2]
        valid_z = z > 0.1
        z_safe = np.where(valid_z, z, 1.0)
        u = full_pc[:, 0] * FX / z_safe + CX
        v = full_pc[:, 1] * FY / z_safe + CY
        inside = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

        # Cores: fora=cinza claro, dentro=amarelo
        colors = np.zeros((len(full_pc), 3), dtype=np.uint8)
        colors[~inside] = [80, 80, 80]      # cinza escuro pra fora
        colors[inside] = [255, 230, 0]      # amarelo pra dentro do frustum

        all_pts = [full_pc]
        all_cols = [colors]

        # Bbox 3D drone (vermelho)
        if "box3D_min_cv" in d:
            edges = edges_of_box(d["box3D_min_cv"], d["box3D_max_cv"])
            all_pts.append(edges)
            all_cols.append(np.full((len(edges), 3), [255, 0, 0], dtype=np.uint8))
        # Centro (verde)
        if "center_cv" in d:
            c = np.array([d["center_cv"]], dtype=np.float32)
            all_pts.append(c)
            all_cols.append(np.array([[0, 255, 0]], dtype=np.uint8))

        all_pts = np.concatenate(all_pts, axis=0)
        all_cols = np.concatenate(all_cols, axis=0)
        out_path = out_dir / f"{name}.ply"
        write_ply(out_path, all_pts, all_cols)
        print(f"  {name}.ply  ({d['drone_name']} @ {d.get('distance_m', 0):.1f}m,"
              f" frustum={inside.sum()} pts, full={len(full_pc)} pts)")

    print(f"\nOutput: {out_dir}")
    print(f"Cores: cinza=cena, amarelo=DENTRO do frustum, vermelho=bbox drone, verde=centro")


if __name__ == "__main__":
    main()
