#!/usr/bin/env python3
"""
Constrói dataset misto 70/30 pra PointNet:
  - 70% samples POSITIVOS: frustum recortado pela bbox 2D (drone visível)
  - 30% samples NEGATIVOS: crops aleatórios da PC fora de qualquer bbox

Cada sample: (N=512 pontos, 3) + label JSON com:
  - is_drone: 1/0
  - center_cv: [x, y, z] do drone (apenas se is_drone=1)
  - size: [w, h, d] do drone (apenas se is_drone=1)

Lê os 3 datasets v2 (NH, City, Coast).
Output: dataset_pointnet/{train,val}/{point_clouds/*.npy, labels/*.json}
"""
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

# Camera intrinsics
IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2

N_POINTS = 512  # tamanho fixo do input pro PointNet
SEED = 42


def airsim_to_cv(p):
    """AirSim [fwd, right, down] → CV [right, down, fwd]"""
    return [p[1], p[2], p[0]]


def project_to_pixel(pc):
    """Projeta pontos CV-frame em coordenadas de pixel"""
    z = pc[:, 2]
    valid = z > 0.1
    z_safe = np.where(valid, z, 1.0)
    u = pc[:, 0] * FX / z_safe + CX
    v = pc[:, 1] * FY / z_safe + CY
    return u, v, valid


def sample_to_fixed(pc, n=N_POINTS, rng=None):
    """Sample/pad para n pontos fixo. Substitui se < n."""
    if rng is None:
        rng = np.random
    if len(pc) >= n:
        idx = rng.choice(len(pc), n, replace=False)
    else:
        idx = rng.choice(len(pc), n, replace=True)
    return pc[idx]


def normalize_pc(pc):
    """Centra na origem do PC e normaliza pra escala unitária. Retorna (pc_norm, centroid, scale)."""
    centroid = pc.mean(axis=0)
    pc_centered = pc - centroid
    scale = np.linalg.norm(pc_centered, axis=1).max()
    scale = max(scale, 1e-6)
    pc_norm = pc_centered / scale
    return pc_norm.astype(np.float32), centroid.astype(np.float32), float(scale)


def crop_frustum(pc, bbox_2d):
    """Pontos do PC dentro da bbox 2D projetada."""
    x1, y1, x2, y2 = bbox_2d
    u, v, valid = project_to_pixel(pc)
    inside = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return pc[inside]


def random_negative_bbox(W=IMAGE_W, H=IMAGE_H, min_size=40, max_size=200, rng=None):
    """Bbox 2D aleatória pra crop negativo."""
    if rng is None:
        rng = random
    w = rng.randint(min_size, max_size)
    h = rng.randint(min_size // 2, max_size // 2)
    x1 = rng.randint(0, W - w)
    y1 = rng.randint(0, H - h)
    return (x1, y1, x1 + w, y1 + h)


def bbox_iou(b1, b2):
    """IoU entre 2 bboxes 2D"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def yolo_to_bbox_px(line, W=IMAGE_W, H=IMAGE_H):
    parts = line.strip().split()
    if len(parts) != 5: return None
    _, xc, yc, bw, bh = parts
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return (int(xc - bw/2), int(yc - bh/2), int(xc + bw/2), int(yc + bh/2))


def process_dataset(src_root, src_tag, out_root, rng):
    """Processa um dataset (NH/City/Coast) e gera samples."""
    src = Path(src_root)
    pc_dir = src / "pointnet" / "point_clouds"
    lbl_dir = src / "pointnet" / "labels_3d"
    yolo_dir = src / "yolo" / "labels"

    n_pos = 0
    n_neg = 0

    for split in ("train", "val"):
        out_split = out_root / split
        (out_split / "point_clouds").mkdir(parents=True, exist_ok=True)
        (out_split / "labels").mkdir(parents=True, exist_ok=True)

        for yolo_p in sorted((yolo_dir / split).glob("*.txt")):
            frame = yolo_p.stem
            pc_p = pc_dir / f"{frame}.npy"
            json_p = lbl_dir / f"{frame}.json"
            if not pc_p.exists():
                continue
            pc = np.load(pc_p).astype(np.float32)
            if len(pc) < 10:
                continue

            yolo_lines = [l for l in open(yolo_p) if l.strip()]
            api_labels = []
            if json_p.exists():
                try:
                    api_labels = json.load(open(json_p))
                except Exception:
                    pass

            # === POSITIVOS: frustums das bboxes 2D ===
            api_bboxes = []
            for i, line in enumerate(yolo_lines):
                bbox = yolo_to_bbox_px(line)
                if bbox is None: continue
                api_bboxes.append(bbox)
                frustum = crop_frustum(pc, bbox)
                if len(frustum) < 8:
                    continue

                # Match label da API
                matched = None
                for d in api_labels:
                    bb = d.get("bbox_2d")
                    if not bb: continue
                    if abs(bb[0] - bbox[0]) <= 3 and abs(bb[1] - bbox[1]) <= 3:
                        matched = d
                        break

                # Sample to N e normaliza
                sampled = sample_to_fixed(frustum, N_POINTS, rng)
                pc_norm, centroid, scale = normalize_pc(sampled)

                # Centro do drone em CV (do snapshot)
                if matched:
                    center_cv = airsim_to_cv(matched["center"])
                    # Normaliza relativo ao centroid do frustum
                    center_rel = ((np.array(center_cv) - centroid) / scale).tolist()
                    distance = matched.get("distance_m", float(np.linalg.norm(center_cv)))
                    drone_name = matched.get("name", "drone")
                else:
                    center_cv = sampled.mean(axis=0).tolist()
                    center_rel = [0.0, 0.0, 0.0]
                    distance = float(np.linalg.norm(sampled.mean(axis=0)))
                    drone_name = "drone"

                # Size do drone (fixo ~1m, ou da box3D se houver)
                if matched and "box3D_min" in matched and "box3D_max" in matched:
                    bmin = airsim_to_cv(matched["box3D_min"])
                    bmax = airsim_to_cv(matched["box3D_max"])
                    size = [abs(bmax[k] - bmin[k]) for k in range(3)]
                else:
                    size = [1.0, 0.3, 1.0]
                size_norm = [s / scale for s in size]

                sample_id = f"{src_tag}_{frame}_obj{i}_pos"
                np.save(out_split / "point_clouds" / f"{sample_id}.npy", pc_norm)
                with open(out_split / "labels" / f"{sample_id}.json", "w") as f:
                    json.dump({
                        "is_drone": 1,
                        "center_cv_raw": list(map(float, center_cv)),
                        "center_rel_normalized": list(map(float, center_rel)),
                        "size_raw_m": size,
                        "size_normalized": list(map(float, size_norm)),
                        "distance_m": float(distance),
                        "drone_name": drone_name,
                        "scale": scale,
                        "centroid_cv": centroid.tolist(),
                        "bbox_2d": list(bbox),
                        "n_pts_original": int(len(frustum)),
                    }, f)
                n_pos += 1

            # === NEGATIVOS: crops aleatórios SEM overlap com bbox 2D do drone ===
            # Quantos negativos pra dar 30% do total (positivos = 70%)
            # → para n positivos, queremos n*(0.30/0.70) ≈ n*0.43 negativos
            n_negative = max(1, int(len(api_bboxes) * 0.43))
            attempts = 0
            negatives_added = 0
            while negatives_added < n_negative and attempts < 20:
                attempts += 1
                neg_bbox = random_negative_bbox(rng=rng)
                # Garante que não overlap com nenhuma bbox positiva
                max_iou = max((bbox_iou(neg_bbox, b) for b in api_bboxes), default=0.0)
                if max_iou > 0.1:
                    continue

                neg_frustum = crop_frustum(pc, neg_bbox)
                if len(neg_frustum) < 8:
                    continue

                sampled = sample_to_fixed(neg_frustum, N_POINTS, rng)
                pc_norm, centroid, scale = normalize_pc(sampled)

                sample_id = f"{src_tag}_{frame}_neg{negatives_added}"
                np.save(out_split / "point_clouds" / f"{sample_id}.npy", pc_norm)
                with open(out_split / "labels" / f"{sample_id}.json", "w") as f:
                    json.dump({
                        "is_drone": 0,
                        "scale": scale,
                        "centroid_cv": centroid.tolist(),
                        "bbox_2d": list(neg_bbox),
                        "n_pts_original": int(len(neg_frustum)),
                    }, f)
                n_neg += 1
                negatives_added += 1

    return n_pos, n_neg


def main():
    out_root = Path("dataset_pointnet")
    rng = np.random.RandomState(SEED)

    sources = [
        ("dataset_nh_v2", "nh"),
        ("dataset_city_v3", "city"),
        ("dataset_coast_v2", "coast"),
    ]

    total_pos = 0
    total_neg = 0
    for src_path, tag in sources:
        if not Path(src_path).exists():
            print(f"  SKIP: {src_path} não existe")
            continue
        print(f"\n=== Processando {src_path} (tag={tag}) ===")
        rng = np.random.RandomState(SEED + hash(tag) % 1000)
        random.seed(SEED + hash(tag) % 1000)
        n_p, n_n = process_dataset(src_path, tag, out_root, rng)
        print(f"  POS: {n_p}  NEG: {n_n}  ratio: {n_p/(n_p+n_n)*100:.1f}/{n_n/(n_p+n_n)*100:.1f}")
        total_pos += n_p
        total_neg += n_n

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_pos + total_neg} samples")
    print(f"  Positivos (frustums): {total_pos} ({total_pos/(total_pos+total_neg)*100:.1f}%)")
    print(f"  Negativos (crops):    {total_neg} ({total_neg/(total_pos+total_neg)*100:.1f}%)")
    print(f"Output: {out_root.absolute()}")


if __name__ == "__main__":
    main()
