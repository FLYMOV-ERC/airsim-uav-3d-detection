#!/usr/bin/env python3
"""ARCHIVED. Standalone frustum extractor for the Frustum-PointNet pipeline.

Uses 2D (YOLO) detections to crop frustums out of the point cloud. The earliest
statement of the frustum idea in this project; later folded into the dataset
builders.
"""

import numpy as np
import json
from pathlib import Path
import cv2

def extract_frustum_from_bbox_2d(point_cloud, bbox_2d, K, img_width, img_height):
    """
    Extrai frustum da point cloud baseado na bounding box 2D

    Args:
        point_cloud: the complete point cloud (N, 3)
        bbox_2d: Bounding box 2D (x_min, y_min, x_max, y_max)
        K: the camera intrinsic matrix
        img_width, img_height: image dimensions

    Returns:
        frustum_points: the points inside the frustum
        frustum_mask: boolean mask of the selected points
    """

    x_min, y_min, x_max, y_max = bbox_2d

    # Add a 10% margin to the box
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

    # For each 3D point, project to 2D and test whether it falls inside the box
    frustum_mask = np.zeros(len(point_cloud), dtype=bool)

    for i, point in enumerate(point_cloud):
        x, y, z = point

        if z > 0:  # point in front of the camera
            # Project to 2D
            u = fx * x / z + cx
            v = fy * y / z + cy

            # Check whether it falls inside the bounding box
            if x_min <= u <= x_max and y_min <= v <= y_max:
                frustum_mask[i] = True

    frustum_points = point_cloud[frustum_mask]

    return frustum_points, frustum_mask

def normalize_frustum(frustum_points, center_method='median'):
    """
    Normalise the frustum so it is centred on the object

    Args:
        frustum_points: the frustum points
        center_method: 'median' or 'mean', for computing the centre

    Returns:
        normalized_points: the normalised points
        center: the centre used for the normalization
    """

    if len(frustum_points) == 0:
        return frustum_points, np.zeros(3)

    # Compute the centre
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
    Process the whole dataset to extract the frustums
    """

    dataset_path = Path(dataset_path)

    # Directories
    metadata_dir = dataset_path / "metadata"
    pointcloud_dir = dataset_path / "pointnet_dataset" / "point_clouds"
    frustum_dir = dataset_path / "pointnet_dataset" / "frustums"
    frustum_dir.mkdir(exist_ok=True)

    # Process each frame
    metadata_files = sorted(metadata_dir.glob("*.json"))

    print(f"Processando {len(metadata_files)} frames...")

    stats = {
        'total_frames': 0,
        'total_frustums': 0,
        'avg_points_per_frustum': []
    }

    for meta_file in metadata_files:
        frame_name = meta_file.stem

        # Load the metadata
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        # Load the point cloud
        pc_file = pointcloud_dir / f"{frame_name}.npy"
        if not pc_file.exists():
            continue

        point_cloud = np.load(pc_file)

        # Camera matrix
        K = np.array(metadata['camera_matrix'])
        img_width, img_height = metadata['image_size']

        # Process each 2D bounding box
        frustums = []
        for i, bbox_2d_info in enumerate(metadata.get('bboxes_2d', [])):
            bbox_2d = bbox_2d_info['bbox']

            # Extrai frustum
            frustum_points, frustum_mask = extract_frustum_from_bbox_2d(
                point_cloud, bbox_2d, K, img_width, img_height
            )

            if len(frustum_points) > 10:  # minimum number of points
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

        # Save the frustums
        if frustums:
            frustum_file = frustum_dir / f"{frame_name}.npz"

            # Save several frustums in one file
            np.savez(frustum_file,
                     frustums=[f['points'] for f in frustums],
                     labels=[f['class'] for f in frustums],
                     bboxes_3d=[f['bbox_3d'] for f in frustums])

            stats['total_frustums'] += len(frustums)

        stats['total_frames'] += 1

        if stats['total_frames'] % 50 == 0:
            print(f"   Processados {stats['total_frames']} frames, "
                  f"{stats['total_frustums']} frustums extracted")

    # Final statistics
    print("\n" + "="*60)
    print("FRUSTUM EXTRACTION COMPLETE")
    print("="*60)
    print(f"Frames processados: {stats['total_frames']}")
    print(f"Total de frustums: {stats['total_frustums']}")
    if stats['avg_points_per_frustum']:
        avg_points = np.mean(stats['avg_points_per_frustum'])
        print(f"Mean points per frustum: {avg_points:.1f}")

    # Build the train/val split
    create_train_val_split(frustum_dir)

def create_train_val_split(frustum_dir, train_ratio=0.8):
    """
    Create the train/validation split
    """

    frustum_files = list(frustum_dir.glob("*.npz"))
    np.random.shuffle(frustum_files)

    n_train = int(len(frustum_files) * train_ratio)

    train_files = frustum_files[:n_train]
    val_files = frustum_files[n_train:]

    # Save the lists
    with open(frustum_dir.parent / "train.txt", 'w') as f:
        for file in train_files:
            f.write(f"{file.stem}\n")

    with open(frustum_dir.parent / "val.txt", 'w') as f:
        for file in val_files:
            f.write(f"{file.stem}\n")

    print(f"\nSplit criado:")
    print(f"   training: {len(train_files)} files")
    print(f"   validation: {len(val_files)} files")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "dataset_frustum_pointnet"

    process_dataset(dataset_path)