#!/usr/bin/env python3
"""ARCHIVED. Export point clouds and bounding boxes for external viewers.

Writes .ply files openable in CloudCompare, MeshLab and similar. Functionally
duplicated by visualization/ply/pointcloud.py, which is the one kept.
"""

import numpy as np
import json
from pathlib import Path

print("\n" + "="*70)
print("EXPORTING POINT CLOUDS FOR VISUALISATION")
print("="*70)

dataset_dir = Path("dataset_final_api")
export_dir = Path("export_for_viewer")
export_dir.mkdir(exist_ok=True)

def numpy_to_ply(points, colors=None, filename="pointcloud.ply"):
    """
    Convert a NumPy array into PLY format
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
    Build the wireframe of a 3D bounding box
    """
    cx, cy, cz = center
    sx, sy, sz = size

    # Define the 8 vertices of the box
    vertices = []
    for dx in [-sx/2, sx/2]:
        for dy in [-sy/2, sy/2]:
            for dz in [-sz/2, sz/2]:
                vertices.append([cx + dx, cy + dy, cz + dz])

    # Define the 12 edges of the box
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Generate points along the edges
    edge_points = []
    points_per_edge = num_points // 12

    for v1, v2 in edges:
        for t in np.linspace(0, 1, points_per_edge):
            point = np.array(vertices[v1]) * (1-t) + np.array(vertices[v2]) * t
            edge_points.append(point)

    return np.array(edge_points)

# Process a few frames
pointcloud_dir = dataset_dir / "pointnet" / "point_clouds"
labels_3d_dir = dataset_dir / "pointnet" / "labels_3d"

pc_files = sorted(list(pointcloud_dir.glob("*.npy")))[:5]  # Primeiros 5 frames

print(f"\n Exportando {len(pc_files)} frames...")

for pc_file in pc_files:
    frame_name = pc_file.stem
    print(f"\n Processando {frame_name}...")

    # Load the point cloud
    point_cloud = np.load(pc_file)

    # Subsample to keep it light
    if len(point_cloud) > 50000:
        indices = np.random.choice(len(point_cloud), 50000, replace=False)
        point_cloud = point_cloud[indices]

    # Load the 3D labels
    labels_file = labels_3d_dir / f"{frame_name}.json"
    bbox_points = []

    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_3d = json.load(f)

        # Generate the points of the bounding boxes
        for label in labels_3d:
            box_wireframe = create_box_wireframe(
                label['center'],
                label['size'],
                num_points=200
            )
            bbox_points.append(box_wireframe)

    # Combine the point cloud with the boxes
    if bbox_points:
        all_bbox_points = np.vstack(bbox_points)

        # Point cloud em cinza/azul
        pc_colors = np.ones((len(point_cloud), 3)) * 150  # Cinza
        pc_colors[:, 2] = 200  # bluer

        # Bboxes em vermelho
        bbox_colors = np.zeros((len(all_bbox_points), 3))
        bbox_colors[:, 0] = 255  # Vermelho

        # Combine everything
        all_points = np.vstack([point_cloud, all_bbox_points])
        all_colors = np.vstack([pc_colors, bbox_colors]).astype(np.uint8)
    else:
        all_points = point_cloud
        all_colors = np.ones((len(point_cloud), 3)) * 150
        all_colors[:, 2] = 200
        all_colors = all_colors.astype(np.uint8)

    # Save as PLY
    output_file = export_dir / f"{frame_name}.ply"
    numpy_to_ply(all_points, all_colors, output_file)
    print(f"   Saved: {output_file}")

# Also write a plain XYZ file
print("\nWriting the XYZ file...")
sample_pc = np.load(pc_files[0])
if len(sample_pc) > 10000:
    indices = np.random.choice(len(sample_pc), 10000, replace=False)
    sample_pc = sample_pc[indices]

xyz_file = export_dir / "sample_pointcloud.xyz"
np.savetxt(xyz_file, sample_pc, fmt='%.6f')
print(f"   Saved: {xyz_file}")

# Write the instructions file
instructions = """
====================================================================
 HOW TO VIEW THE POINT CLOUDS
====================================================================

1. CLOUDCOMPARE (Recomendado):
   - Download: https://www.cloudcompare.org/
   - Abra o CloudCompare
   - drag the .ply files into the window
   - Point cloud aparece em cinza/azul
   - Bounding boxes aparecem em vermelho

2. MESHLAB:
   - Download: https://www.meshlab.net/
   - File > Import Mesh
   - select the .ply file

3. WINDOWS 3D VIEWER:
   - right-click the .ply file
   - Open with > 3D Viewer

4. ONLINE (nothing to install):
   - https://3dviewer.net/
   - drag the .ply file in

FILES GENERATED:
- frame_XXXXXX.ply: Point cloud + bounding boxes
- sample_pointcloud.xyz: simple format (points only)

COLOURS:
- Cinza/Azul: Point cloud do ambiente
- red: the drones' bounding boxes

====================================================================
"""

with open(export_dir / "LEIA_ME.txt", 'w', encoding='utf-8') as f:
    f.write(instructions)

print("\n" + "="*70)
print("EXPORT COMPLETE")
print("="*70)
print(f"\nFiles saved to: {export_dir.absolute()}")
print("\nPara visualizar:")
print("1. copy the 'export_for_viewer' folder to Windows")
print("2. open the .ply files in CloudCompare or MeshLab")
print("3. As bounding boxes aparecem em VERMELHO")
print("4. A point cloud aparece em CINZA/AZUL")