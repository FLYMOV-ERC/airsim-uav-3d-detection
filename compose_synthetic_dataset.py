#!/usr/bin/env python3
"""Compõe dataset sintético: BG (sem drone) + sprites RGBA = imagem labelled.

Bbox 2D é PIXEL-PERFECT (sabemos onde colamos).
Label 3D é estimada: distância via escala do sprite.

Output: dataset_synth/{train,val}/{yolo,pointnet,visualizations}
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import cv2


IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))   # = 640
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2

# Drone real Quadrotor1 4x: ~5.5m x 5.5m x 2m (X=fwd Y=right Z=down)
DRONE_PHYS_W = 5.50
DRONE_PHYS_H = 2.00


def load_backgrounds(bg_root):
    """Lista todos backgrounds disponíveis (envs múltiplos)."""
    bgs = []
    for env_dir in Path(bg_root).iterdir():
        if not env_dir.is_dir():
            continue
        rgb_dir = env_dir / "rgb"
        depth_dir = env_dir / "depth"
        meta_dir = env_dir / "meta"
        if not rgb_dir.exists():
            continue
        for rgb_p in sorted(rgb_dir.glob("*.jpg")):
            depth_p = depth_dir / f"{rgb_p.stem}.npy"
            meta_p = meta_dir / f"{rgb_p.stem}.json"
            bgs.append({
                "rgb": rgb_p,
                "depth": depth_p if depth_p.exists() else None,
                "meta": meta_p if meta_p.exists() else None,
                "env": env_dir.name,
            })
    return bgs


def load_sprites(sprite_root):
    """Carrega todos sprites RGBA + metadata."""
    sprite_dir = Path(sprite_root)
    meta_p = sprite_dir / "meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"meta.json não encontrado em {sprite_dir}")
    meta = json.load(open(meta_p))
    sprites = []
    for m in meta:
        sprite_p = sprite_dir / f"{m['id']}.png"
        if not sprite_p.exists():
            continue
        img = cv2.imread(str(sprite_p), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            continue
        sprites.append({
            "img": img,         # BGRA
            "meta": m,
        })
    return sprites


def overlay_alpha(bg, sprite_bgra, x, y):
    """Cola sprite RGBA em bg na posição (x, y) = canto superior esquerdo.

    Retorna bg modificado in-place (return reference) e bbox efetivo
    (x1, y1, x2, y2) considerando clip nas bordas.
    """
    H, W = bg.shape[:2]
    sh, sw = sprite_bgra.shape[:2]
    # clipping
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(W, x + sw); y2 = min(H, y + sh)
    if x2 <= x1 or y2 <= y1:
        return bg, None
    # offset no sprite
    sx1 = x1 - x; sy1 = y1 - y
    sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)
    sprite_roi = sprite_bgra[sy1:sy2, sx1:sx2]
    alpha = sprite_roi[..., 3:4].astype(np.float32) / 255.0
    rgb = sprite_roi[..., :3].astype(np.float32)
    bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * rgb + (1 - alpha) * bg_roi
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg, (x1, y1, x2, y2)


def estimate_distance_from_size(sprite_w_px, sprite_meta):
    """Distância a partir do tamanho do sprite na imagem composta.

    sprite_meta tem o tamanho ORIGINAL e a distância na qual foi capturado.
    Se renderizamos em scale s, a distância atual = original_dist / s.
    """
    orig_w = sprite_meta["sprite_size"][0]
    orig_dist = sprite_meta["distance_m"]
    scale = sprite_w_px / orig_w
    if scale < 1e-3:
        return orig_dist * 100
    new_dist = orig_dist / scale
    return new_dist


def random_position(W, H, sprite_w, sprite_h, rng, restrict_to_sky=False):
    """Posição aleatória.
    restrict_to_sky=True: limita y na metade superior (sprite distante deve ficar no céu).
    """
    x = rng.randint(0, max(1, W - sprite_w))
    if restrict_to_sky:
        y_max = int(H * 0.55) - sprite_h
        y = rng.randint(0, max(1, y_max))
    else:
        y = rng.randint(0, max(1, H - sprite_h))
    return x, y


def compose_frame(bg_orig, sprites_to_place, rng):
    """
    sprites_to_place: lista de (sprite_dict, scale_factor)
    return: bg_composto, lista de labels {bbox_2d, distance_m, sprite_meta}
    """
    bg = bg_orig.copy()
    labels = []
    placed_bboxes = []
    for sprite_dict, scale in sprites_to_place:
        sprite_meta = sprite_dict["meta"]
        sprite_img = sprite_dict["img"]
        # Resize
        orig_h, orig_w = sprite_img.shape[:2]
        new_w = max(8, int(orig_w * scale))
        new_h = max(4, int(orig_h * scale))
        if new_w > IMAGE_W - 10 or new_h > IMAGE_H - 10:
            continue
        resized = cv2.resize(sprite_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Drone PEQUENO (longe) deve ficar acima do horizonte (céu)
        # Threshold: sprite < 80 pixels de largura = "distante" → restringe ao céu
        restrict_sky = new_w < 80
        # Position com até 15 tentativas pra evitar overlap > 50%
        tries = 0
        placed_bb = None
        while tries < 15:
            x, y = random_position(IMAGE_W, IMAGE_H, new_w, new_h, rng, restrict_to_sky=restrict_sky)
            this_bb = (x, y, x + new_w, y + new_h)
            # Calcula MAX oclusão (intersection / min_area dos 2 boxes)
            max_occl = 0.0
            for pb in placed_bboxes:
                ix1 = max(this_bb[0], pb[0]); iy1 = max(this_bb[1], pb[1])
                ix2 = min(this_bb[2], pb[2]); iy2 = min(this_bb[3], pb[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    area1 = new_w * new_h
                    area2 = (pb[2]-pb[0])*(pb[3]-pb[1])
                    occl = inter / min(area1, area2)
                    max_occl = max(max_occl, occl)
            if max_occl <= 0.5:
                placed_bb = this_bb; break
            tries += 1
        if placed_bb is None:
            continue
        _, bbox_clipped = overlay_alpha(bg, resized, placed_bb[0], placed_bb[1])
        if bbox_clipped is None:
            continue
        placed_bboxes.append(placed_bb)
        # Distância estimada
        dist = estimate_distance_from_size(new_w, sprite_meta)
        labels.append({
            "bbox_2d": list(bbox_clipped),
            "distance_m": float(dist),
            "scale_used": float(scale),
            "sprite_id": sprite_meta["id"],
            "sprite_yaw_deg": sprite_meta["yaw_deg"],
            "sprite_pitch_deg": sprite_meta["pitch_deg"],
            "sprite_roll_deg": sprite_meta["roll_deg"],
        })
    return bg, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds", default="backgrounds")
    ap.add_argument("--sprites", default="sprites")
    ap.add_argument("--output", default="dataset_synth")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--min_drones", type=int, default=1)
    ap.add_argument("--max_drones", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vis_every", type=int, default=50)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    bgs = load_backgrounds(args.backgrounds)
    sprites = load_sprites(args.sprites)
    print(f"Loaded {len(bgs)} backgrounds and {len(sprites)} sprites")
    if not bgs or not sprites:
        print("Faltam dados.")
        return

    out_dir = Path(args.output)
    (out_dir / "yolo" / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "yolo" / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "yolo" / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "yolo" / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels_3d").mkdir(exist_ok=True)
    (out_dir / "visualizations").mkdir(exist_ok=True)

    total = args.n_train + args.n_val
    for i in range(total):
        split = "train" if i < args.n_train else "val"
        # Pick BG
        bg_info = rng.choice(bgs)
        bg = cv2.imread(str(bg_info["rgb"]))
        if bg is None:
            continue
        # Pick N drones
        n = rng.randint(args.min_drones, args.max_drones)
        sprites_to_place = []
        for _ in range(n):
            s = rng.choice(sprites)
            # Escala simula distância: 0.4-1.5 do tamanho original
            # (sprite original foi capturado em distance 10-60m)
            # min 0.4 evita drones MUITO pequenos (parece anão se baixo na imagem)
            scale = rng.uniform(0.4, 1.5)
            sprites_to_place.append((s, scale))
        # Compose
        composed, labels = compose_frame(bg, sprites_to_place, np_rng)
        if not labels:
            continue
        # Save
        frame_name = f"frame_{i:06d}"
        cv2.imwrite(str(out_dir / "yolo" / "images" / split / f"{frame_name}.jpg"),
                    composed, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # YOLO label
        yolo_lines = []
        for lbl in labels:
            x1, y1, x2, y2 = lbl["bbox_2d"]
            xc = (x1+x2)/2 / IMAGE_W
            yc = (y1+y2)/2 / IMAGE_H
            bw = (x2-x1) / IMAGE_W
            bh = (y2-y1) / IMAGE_H
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(out_dir / "yolo" / "labels" / split / f"{frame_name}.txt", "w") as f:
            f.write("\n".join(yolo_lines))
        # 3D label (apenas distância estimada — pode ser usada por PointNet wrapper)
        with open(out_dir / "labels_3d" / f"{frame_name}.json", "w") as f:
            json.dump({
                "frame": frame_name,
                "background_env": bg_info["env"],
                "n_drones": len(labels),
                "labels": labels,
            }, f, indent=2)
        # Visualization
        if i % args.vis_every == 0:
            viz = composed.copy()
            for lbl in labels:
                x1, y1, x2, y2 = lbl["bbox_2d"]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, f"D:{lbl['distance_m']:.0f}m", (x1, max(15, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(out_dir / "visualizations" / f"{frame_name}.jpg"), viz)
        if i % 100 == 0:
            print(f"  [{i}/{total}]  n_drones={n}  env={bg_info['env']}")

    # dataset.yaml
    with open(out_dir / "yolo" / "dataset.yaml", "w") as f:
        f.write(f"""train: {Path(args.output).absolute()}/yolo/images/train
val: {Path(args.output).absolute()}/yolo/images/val
nc: 1
names: ['drone']
""")
    print(f"\nDone! {total} synthetic frames → {out_dir.absolute()}")


if __name__ == "__main__":
    main()
