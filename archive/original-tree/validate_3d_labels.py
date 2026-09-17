#!/usr/bin/env python3
"""
Validação das labels 3D e point clouds
Verifica se as posições 3D dos drones estão corretas
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

print("\n" + "="*70)
print("🔍 VALIDAÇÃO DAS LABELS 3D")
print("="*70)

dataset_dir = Path("dataset_final_api")

# Verifica se o dataset existe
if not dataset_dir.exists():
    print("❌ Dataset não encontrado!")
    exit(1)

# Lista alguns frames para validar
pointcloud_dir = dataset_dir / "pointnet" / "point_clouds"
labels_3d_dir = dataset_dir / "pointnet" / "labels_3d"
vis_dir = dataset_dir / "visualizations"

pointcloud_files = sorted(list(pointcloud_dir.glob("*.npy")))[:5]  # Primeiros 5 frames

if not pointcloud_files:
    print("❌ Nenhuma point cloud encontrada!")
    exit(1)

print(f"📊 Validando {len(pointcloud_files)} frames...")

for pc_file in pointcloud_files:
    frame_name = pc_file.stem
    print(f"\n🔍 Frame: {frame_name}")
    print("-" * 50)

    # Carrega point cloud
    point_cloud = np.load(pc_file)
    print(f"   Point cloud: {point_cloud.shape[0]:,} pontos")

    # Carrega labels 3D
    labels_file = labels_3d_dir / f"{frame_name}.json"
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_3d = json.load(f)

        print(f"   Labels 3D: {len(labels_3d)} drones detectados")

        for i, label in enumerate(labels_3d):
            print(f"\n   Drone {i+1}:")
            print(f"      Nome: {label.get('name', 'unknown')}")
            print(f"      Centro 3D: ({label['center'][0]:.2f}, {label['center'][1]:.2f}, {label['center'][2]:.2f})")
            print(f"      Tamanho: ({label['size'][0]:.2f}, {label['size'][1]:.2f}, {label['size'][2]:.2f})")
            print(f"      BBox 2D: {label['bbox_2d']}")

            # Validações
            center = label['center']

            # Verifica se a posição faz sentido
            if abs(center[0]) > 100 or abs(center[1]) > 100 or abs(center[2]) > 100:
                print("      ⚠️ AVISO: Posição muito distante (>100m)")

            if center[2] < 1:  # Z muito pequeno
                print("      ⚠️ AVISO: Drone muito próximo (Z < 1m)")

            # Verifica se o tamanho faz sentido para um drone
            size = label['size']
            if any(s > 5 for s in size):
                print("      ⚠️ AVISO: Tamanho muito grande para um drone (>5m)")
            if any(s < 0.1 for s in size):
                print("      ⚠️ AVISO: Tamanho muito pequeno (<0.1m)")

    else:
        print(f"   ⚠️ Sem labels 3D")

    # Estatísticas da point cloud
    if len(point_cloud) > 0:
        print(f"\n   📊 Estatísticas da Point Cloud:")
        print(f"      X: min={point_cloud[:, 0].min():.2f}, max={point_cloud[:, 0].max():.2f}")
        print(f"      Y: min={point_cloud[:, 1].min():.2f}, max={point_cloud[:, 1].max():.2f}")
        print(f"      Z: min={point_cloud[:, 2].min():.2f}, max={point_cloud[:, 2].max():.2f}")

print("\n" + "="*70)
print("📊 CRIANDO VISUALIZAÇÃO 3D")
print("="*70)

# Pega um frame para visualização detalhada
sample_frame = pointcloud_files[0].stem
point_cloud = np.load(pointcloud_files[0])
labels_file = labels_3d_dir / f"{sample_frame}.json"

if labels_file.exists():
    with open(labels_file, 'r') as f:
        labels_3d = json.load(f)

    # Cria figura 3D
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Point cloud completa
    ax1 = fig.add_subplot(121, projection='3d')

    # Amostra pontos para visualização (muito pontos deixa lento)
    sample_size = min(5000, len(point_cloud))
    indices = np.random.choice(len(point_cloud), sample_size, replace=False)
    sampled_points = point_cloud[indices]

    ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                c=sampled_points[:, 2], cmap='viridis', s=0.1, alpha=0.5)

    # Adiciona bounding boxes 3D
    for label in labels_3d:
        center = label['center']
        size = label['size']

        # Cria os 8 vértices da bounding box
        x_range = [center[0] - size[0]/2, center[0] + size[0]/2]
        y_range = [center[1] - size[1]/2, center[1] + size[1]/2]
        z_range = [center[2] - size[2]/2, center[2] + size[2]/2]

        # Desenha as arestas da caixa
        for x in x_range:
            for y in y_range:
                ax1.plot([x, x], [y, y], z_range, 'r-', linewidth=2)

        for x in x_range:
            for z in z_range:
                ax1.plot([x, x], y_range, [z, z], 'r-', linewidth=2)

        for y in y_range:
            for z in z_range:
                ax1.plot(x_range, [y, y], [z, z], 'r-', linewidth=2)

        # Marca o centro
        ax1.scatter([center[0]], [center[1]], [center[2]],
                   c='red', s=100, marker='x')

        # Adiciona texto com nome
        ax1.text(center[0], center[1], center[2],
                label.get('name', 'drone'), fontsize=8)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Point Cloud com Bounding Boxes 3D\n{sample_frame}')

    # Plot 2: Vista de cima (XY)
    ax2 = fig.add_subplot(122)
    ax2.scatter(sampled_points[:, 0], sampled_points[:, 1],
                c=sampled_points[:, 2], cmap='viridis', s=0.1, alpha=0.5)

    # Adiciona bounding boxes na vista de cima
    for label in labels_3d:
        center = label['center']
        size = label['size']

        # Desenha retângulo
        rect = plt.Rectangle((center[0] - size[0]/2, center[1] - size[1]/2),
                            size[0], size[1],
                            fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)

        # Marca centro
        ax2.plot(center[0], center[1], 'rx', markersize=10)
        ax2.text(center[0], center[1], label.get('name', 'drone'), fontsize=8)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Vista de Cima (XY)')
    ax2.axis('equal')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('validation_3d.png', dpi=100)
    print(f"\n✅ Visualização salva: validation_3d.png")

    # Cria comparação 2D-3D
    print("\n📊 VALIDANDO CORRESPONDÊNCIA 2D-3D")
    print("-" * 50)

    # Carrega imagem RGB correspondente
    img_path = dataset_dir / "visualizations" / f"{sample_frame}.jpg"
    if img_path.exists():
        img = cv2.imread(str(img_path))

        # Verifica se as posições 3D correspondem às bboxes 2D
        for i, label in enumerate(labels_3d):
            bbox_2d = label['bbox_2d']
            center_3d = label['center']

            print(f"\n   Drone {i+1} ({label.get('name', 'unknown')}):")
            print(f"      BBox 2D: [{bbox_2d[0]}, {bbox_2d[1]}] a [{bbox_2d[2]}, {bbox_2d[3]}]")
            print(f"      Centro 3D: ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})")

            # Calcula centro da bbox 2D
            center_2d_x = (bbox_2d[0] + bbox_2d[2]) / 2
            center_2d_y = (bbox_2d[1] + bbox_2d[3]) / 2

            # Verifica proporções
            img_height, img_width = img.shape[:2]
            rel_x = center_2d_x / img_width
            rel_y = center_2d_y / img_height

            print(f"      Centro 2D relativo: ({rel_x:.2%}, {rel_y:.2%})")

            # Validação básica
            if center_3d[1] > 0 and rel_x < 0.5:
                print("      ⚠️ Inconsistência: Drone à direita (Y>0) mas bbox à esquerda")
            elif center_3d[1] < 0 and rel_x > 0.5:
                print("      ⚠️ Inconsistência: Drone à esquerda (Y<0) mas bbox à direita")
            else:
                print("      ✅ Posição 2D-3D consistente")

print("\n" + "="*70)
print("✅ VALIDAÇÃO COMPLETA!")
print("="*70)