#!/usr/bin/env python3
"""
Pós-processador: gera visualização para CADA frame de um dataset
(combina imagem + labels YOLO + info 3D do JSON).

Uso:
    python3 render_all_visualizations.py <dataset_dir>
    # ex: python3 render_all_visualizations.py teste_nh_high
"""
import sys
import json
import cv2
from pathlib import Path

IMAGE_W, IMAGE_H = 1280, 720


def render_one(img_path, yolo_path, json_path, env_label=""):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # Lê 3D labels (para distância e nome)
    info_3d = {}
    if json_path and json_path.exists():
        try:
            data = json.load(open(json_path))
            for d in data:
                bb = tuple(d["bbox_2d"])
                # Prefere distance_m (vem direto da API) sobre center[2] (que é Z relativo)
                dist = d.get("distance_m")
                if dist is None:
                    c = d.get("center", [0, 0, 0])
                    dist = (c[0]**2 + c[1]**2 + c[2]**2) ** 0.5
                info_3d[bb] = (d.get("name", "?"), dist)
        except Exception:
            pass

    # Lê YOLO label (normalizado)
    if not yolo_path.exists():
        return img
    with open(yolo_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = parts
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x_min = int((xc - bw/2) * w)
        y_min = int((yc - bh/2) * h)
        x_max = int((xc + bw/2) * w)
        y_max = int((yc + bh/2) * h)

        # Tenta achar info 3D correspondente
        name = "drone"
        depth_str = ""
        for bb_key, (n, z) in info_3d.items():
            if abs(bb_key[0] - x_min) <= 2 and abs(bb_key[1] - y_min) <= 2:
                name = n
                depth_str = f" {z:.1f}m"
                break

        # Desenha
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{name}{depth_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Fundo preto pro texto ficar legível
        cv2.rectangle(img, (x_min, max(0, y_min - th - 6)),
                      (x_min + tw + 4, y_min), (0, 0, 0), -1)
        cv2.putText(img, label, (x_min + 2, max(th, y_min - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if env_label:
        cv2.putText(img, env_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return img


def main():
    if len(sys.argv) < 2:
        print("uso: python3 render_all_visualizations.py <dataset_dir>")
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"diretório não existe: {root}")
        sys.exit(1)

    out_dir = root / "visualizations_all"
    out_dir.mkdir(exist_ok=True)

    count = 0
    for split in ("train", "val"):
        img_dir = root / "yolo" / "images" / split
        lbl_dir = root / "yolo" / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            frame = img_path.stem
            yolo_path = lbl_dir / f"{frame}.txt"
            json_path = root / "pointnet" / "labels_3d" / f"{frame}.json"
            vis = render_one(img_path, yolo_path, json_path,
                             env_label=f"{root.name} [{split}]  {frame}")
            if vis is not None:
                cv2.imwrite(str(out_dir / f"{frame}_{split}.jpg"), vis)
                count += 1

    print(f"Renderizadas {count} visualizações em {out_dir}/")


if __name__ == "__main__":
    main()
