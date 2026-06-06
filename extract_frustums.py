#!/usr/bin/env python3
"""
Extrator de Frustums - Pipeline Frustum-PointNet
Usa detecções 2D (YOLO) para extrair frustums da point cloud
"""

import numpy as np
import json
from pathlib import Path
import cv2

def extract_frustum_from_bbox_2d(point_cloud, bbox_2d, K, img_width, img_height):
    """
    Extrai frustum da point cloud baseado na bounding box 2D

    Args:
        point_cloud: Nuvem de pontos completa (N, 3)
        bbox_2d: Bounding box 2D (x_min, y_min, x_max, y_max)
        K: Matriz intrínseca da câmera
        img_width, img_height: Dimensões da imagem

    Returns:
        frustum_points: Pontos dentro do frustum
        frustum_mask: Máscara booleana dos pontos selecionados
    """

    x_min, y_min, x_max, y_max = bbox_2d

    # Adiciona margem à bbox (10%)
    margin = 0.1
    width = x_max - x_min
    height = y_max - y_min

    x_min = max(0, x_min - width * margin)
    x_max = min(img_width, x_max + width * margin)
    y_min = max(0, y_min - height * margin)
    y_max = min(img_height, y_max + height * margin)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Para cada ponto 3D, projeta para 2D e verifica se está na bbox
    frustum_mask = np.zeros(len(point_cloud), dtype=bool)

    for i, point in enumerate(point_cloud):
        x, y, z = point

        if z > 0:  # Ponto na frente da câmera
            # Projeta para 2D
            u = fx * x / z + cx
            v = fy * y / z + cy

            # Verifica se está dentro da bounding box
            if x_min <= u <= x_max and y_min <= v <= y_max:
                frustum_mask[i] = True

    frustum_points = point_cloud[frustum_mask]

    return frustum_points, frustum_mask

def normalize_frustum(frustum_points, center_method='median'):
    """
    Normaliza frustum para centralizar no objeto

    Args:
        frustum_points: Pontos do frustum
        center_method: 'median' ou 'mean' para calcular centro

    Returns:
        normalized_points: Pontos normalizados
        center: Centro usado para normalização
    """

    if len(frustum_points) == 0:
        return frustum_points, np.zeros(3)

    # Calcula centro
    if center_method == 'median':
        center = np.median(frustum_points, axis=0)
    else:
        center = np.mean(frustum_points, axis=0)

    # Centraliza
    normalized_points = frustum_points - center

    # Opcionalmente, normaliza escala
    max_dist = np.max(np.linalg.norm(normalized_points, axis=1))
    if max_dist > 0:
        normalized_points = normalized_points / max_dist

    return normalized_points, center

def process_dataset(dataset_path):
    """
    Processa todo o dataset para extrair frustums
    """

    dataset_path = Path(dataset_path)

    # Diretórios
    metadata_dir = dataset_path / "metadata"
    pointcloud_dir = dataset_path / "pointnet_dataset" / "point_clouds"
    frustum_dir = dataset_path / "pointnet_dataset" / "frustums"
    frustum_dir.mkdir(exist_ok=True)

    # Processa cada frame
    metadata_files = sorted(metadata_dir.glob("*.json"))

    print(f"Processando {len(metadata_files)} frames...")

    stats = {
        'total_frames': 0,
        'total_frustums': 0,
        'avg_points_per_frustum': []
    }

    for meta_file in metadata_files:
        frame_name = meta_file.stem

        # Carrega metadados
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        # Carrega point cloud
        pc_file = pointcloud_dir / f"{frame_name}.npy"
        if not pc_file.exists():
            continue

        point_cloud = np.load(pc_file)

        # Matriz da câmera
        K = np.array(metadata['camera_matrix'])
        img_width, img_height = metadata['image_size']

        # Processa cada bounding box 2D
        frustums = []
        for i, bbox_2d_info in enumerate(metadata.get('bboxes_2d', [])):
            bbox_2d = bbox_2d_info['bbox']

            # Extrai frustum
            frustum_points, frustum_mask = extract_frustum_from_bbox_2d(
                point_cloud, bbox_2d, K, img_width, img_height
            )

            if len(frustum_points) > 10:  # Mínimo de pontos
                # Normaliza frustum
                normalized_frustum, center = normalize_frustum(frustum_points)

                # Pega label 3D correspondente
                bbox_3d_info = metadata['bboxes_3d'][i] if i < len(metadata['bboxes_3d']) else None

                frustum_data = {
                    'points': normalized_frustum,
                    'original_center': center.tolist(),
                    'num_points': len(normalized_frustum),
                    'bbox_2d': bbox_2d,
                    'bbox_3d': bbox_3d_info,
                    'class': bbox_2d_info['class_name']
                }

                frustums.append(frustum_data)
                stats['avg_points_per_frustum'].append(len(normalized_frustum))

        # Salva frustums
        if frustums:
            frustum_file = frustum_dir / f"{frame_name}.npz"

            # Salva múltiplos frustums em um arquivo
            np.savez(frustum_file,
                     frustums=[f['points'] for f in frustums],
                     labels=[f['class'] for f in frustums],
                     bboxes_3d=[f['bbox_3d'] for f in frustums])

            stats['total_frustums'] += len(frustums)

        stats['total_frames'] += 1

        if stats['total_frames'] % 50 == 0:
            print(f"   Processados {stats['total_frames']} frames, "
                  f"{stats['total_frustums']} frustums extraídos")

    # Estatísticas finais
    print("\n" + "="*60)
    print("EXTRAÇÃO DE FRUSTUMS COMPLETA!")
    print("="*60)
    print(f"Frames processados: {stats['total_frames']}")
    print(f"Total de frustums: {stats['total_frustums']}")
    if stats['avg_points_per_frustum']:
        avg_points = np.mean(stats['avg_points_per_frustum'])
        print(f"Média de pontos por frustum: {avg_points:.1f}")

    # Cria split train/val
    create_train_val_split(frustum_dir)

def create_train_val_split(frustum_dir, train_ratio=0.8):
    """
    Cria divisão treino/validação
    """

    frustum_files = list(frustum_dir.glob("*.npz"))
    np.random.shuffle(frustum_files)

    n_train = int(len(frustum_files) * train_ratio)

    train_files = frustum_files[:n_train]
    val_files = frustum_files[n_train:]

    # Salva listas
    with open(frustum_dir.parent / "train.txt", 'w') as f:
        for file in train_files:
            f.write(f"{file.stem}\n")

    with open(frustum_dir.parent / "val.txt", 'w') as f:
        for file in val_files:
            f.write(f"{file.stem}\n")

    print(f"\nSplit criado:")
    print(f"   Treino: {len(train_files)} arquivos")
    print(f"   Validação: {len(val_files)} arquivos")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "dataset_frustum_pointnet"

    process_dataset(dataset_path)