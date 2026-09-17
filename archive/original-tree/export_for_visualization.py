#!/usr/bin/env python3
"""
Exporta point clouds e bounding boxes para visualização em softwares externos
Gera arquivos .ply que podem ser abertos no CloudCompare, MeshLab, etc.
"""

import numpy as np
import json
from pathlib import Path

print("\n" + "="*70)
print("📦 EXPORTANDO POINT CLOUDS PARA VISUALIZAÇÃO")
print("="*70)

dataset_dir = Path("dataset_final_api")
export_dir = Path("export_for_viewer")
export_dir.mkdir(exist_ok=True)

def numpy_to_ply(points, colors=None, filename="pointcloud.ply"):
    """
    Converte numpy array para formato PLY
    """
    with open(filename, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]} {point[1]} {point[2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
            else:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def create_box_wireframe(center, size, num_points=100):
    """
    Cria wireframe de uma bounding box 3D
    """
    cx, cy, cz = center
    sx, sy, sz = size

    # Define os 8 vértices da caixa
    vertices = []
    for dx in [-sx/2, sx/2]:
        for dy in [-sy/2, sy/2]:
            for dz in [-sz/2, sz/2]:
                vertices.append([cx + dx, cy + dy, cz + dz])

    # Define as 12 arestas da caixa
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Cria pontos ao longo das arestas
    edge_points = []
    points_per_edge = num_points // 12

    for v1, v2 in edges:
        for t in np.linspace(0, 1, points_per_edge):
            point = np.array(vertices[v1]) * (1-t) + np.array(vertices[v2]) * t
            edge_points.append(point)

    return np.array(edge_points)

# Processa alguns frames
pointcloud_dir = dataset_dir / "pointnet" / "point_clouds"
labels_3d_dir = dataset_dir / "pointnet" / "labels_3d"

pc_files = sorted(list(pointcloud_dir.glob("*.npy")))[:5]  # Primeiros 5 frames

print(f"\n📊 Exportando {len(pc_files)} frames...")

for pc_file in pc_files:
    frame_name = pc_file.stem
    print(f"\n🔧 Processando {frame_name}...")

    # Carrega point cloud
    point_cloud = np.load(pc_file)

    # Amostra para não ficar muito pesado
    if len(point_cloud) > 50000:
        indices = np.random.choice(len(point_cloud), 50000, replace=False)
        point_cloud = point_cloud[indices]

    # Carrega labels 3D
    labels_file = labels_3d_dir / f"{frame_name}.json"
    bbox_points = []

    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_3d = json.load(f)

        # Cria pontos para as bounding boxes
        for label in labels_3d:
            box_wireframe = create_box_wireframe(
                label['center'],
                label['size'],
                num_points=200
            )
            bbox_points.append(box_wireframe)

    # Combina point cloud com bboxes
    if bbox_points:
        all_bbox_points = np.vstack(bbox_points)

        # Point cloud em cinza/azul
        pc_colors = np.ones((len(point_cloud), 3)) * 150  # Cinza
        pc_colors[:, 2] = 200  # Mais azul

        # Bboxes em vermelho
        bbox_colors = np.zeros((len(all_bbox_points), 3))
        bbox_colors[:, 0] = 255  # Vermelho

        # Combina tudo
        all_points = np.vstack([point_cloud, all_bbox_points])
        all_colors = np.vstack([pc_colors, bbox_colors]).astype(np.uint8)
    else:
        all_points = point_cloud
        all_colors = np.ones((len(point_cloud), 3)) * 150
        all_colors[:, 2] = 200
        all_colors = all_colors.astype(np.uint8)

    # Salva como PLY
    output_file = export_dir / f"{frame_name}.ply"
    numpy_to_ply(all_points, all_colors, output_file)
    print(f"   ✅ Salvo: {output_file}")

# Cria também um arquivo XYZ simples
print("\n📊 Criando arquivo XYZ...")
sample_pc = np.load(pc_files[0])
if len(sample_pc) > 10000:
    indices = np.random.choice(len(sample_pc), 10000, replace=False)
    sample_pc = sample_pc[indices]

xyz_file = export_dir / "sample_pointcloud.xyz"
np.savetxt(xyz_file, sample_pc, fmt='%.6f')
print(f"   ✅ Salvo: {xyz_file}")

# Cria arquivo de instruções
instructions = """
====================================================================
📖 COMO VISUALIZAR AS POINT CLOUDS
====================================================================

1. CLOUDCOMPARE (Recomendado):
   - Download: https://www.cloudcompare.org/
   - Abra o CloudCompare
   - Arraste os arquivos .ply para a janela
   - Point cloud aparece em cinza/azul
   - Bounding boxes aparecem em vermelho

2. MESHLAB:
   - Download: https://www.meshlab.net/
   - File > Import Mesh
   - Selecione o arquivo .ply

3. WINDOWS 3D VIEWER:
   - Clique direito no arquivo .ply
   - Abrir com > 3D Viewer

4. ONLINE (sem instalar nada):
   - https://3dviewer.net/
   - Arraste o arquivo .ply

ARQUIVOS GERADOS:
- frame_XXXXXX.ply: Point cloud + bounding boxes
- sample_pointcloud.xyz: Formato simples (só pontos)

CORES:
- Cinza/Azul: Point cloud do ambiente
- Vermelho: Bounding boxes dos drones

====================================================================
"""

with open(export_dir / "LEIA_ME.txt", 'w', encoding='utf-8') as f:
    f.write(instructions)

print("\n" + "="*70)
print("✅ EXPORTAÇÃO COMPLETA!")
print("="*70)
print(f"\n📁 Arquivos salvos em: {export_dir.absolute()}")
print("\nPara visualizar:")
print("1. Copie a pasta 'export_for_viewer' para o Windows")
print("2. Use CloudCompare ou MeshLab para abrir os arquivos .ply")
print("3. As bounding boxes aparecem em VERMELHO")
print("4. A point cloud aparece em CINZA/AZUL")