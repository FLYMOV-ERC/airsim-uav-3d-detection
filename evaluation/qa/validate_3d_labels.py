#!/usr/bin/env python3
"""
Validate the 3D labels of a collected dataset against its point clouds.

Prints, for the first N frames, the 3D box centre/size of every labelled drone
together with point-cloud extents, flags implausible values, and writes one
figure combining a 3D scatter with boxes and a top-down (XY) view.

Usage:
    python evaluation/qa/validate_3d_labels.py <dataset_dir> [--frames N] [--out FILE]

<dataset_dir> is a dataset root containing ``pointnet/point_clouds/*.npy``,
``pointnet/labels_3d/*.json`` and (optionally) ``visualizations/*.jpg``.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import cv2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Validate a dataset's 3D labels against its point clouds.")
    ap.add_argument("dataset_dir", type=Path,
                    help="dataset root containing pointnet/point_clouds and "
                         "pointnet/labels_3d")
    ap.add_argument("--frames", type=int, default=5,
                    help="how many frames to check (default: 5)")
    ap.add_argument("--out", type=Path, default=Path("validation_3d.png"),
                    help="where to write the figure (default: ./validation_3d.png)")
    ap.add_argument("--max-points", type=int, default=5000,
                    help="points subsampled for plotting (default: 5000)")
    return ap.parse_args(argv)


def report_frame(pc_file, labels_3d_dir):
    """Print the label/point-cloud report for one frame."""
    frame_name = pc_file.stem
    print(f"\nFrame: {frame_name}")
    print("-" * 50)

    point_cloud = np.load(pc_file)
    print(f"   Point cloud: {point_cloud.shape[0]:,} points")

    labels_file = labels_3d_dir / f"{frame_name}.json"
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_3d = json.load(f)

        print(f"   3D labels: {len(labels_3d)} drones detected")

        for i, label in enumerate(labels_3d):
            print(f"\n   Drone {i+1}:")
            print(f"      Name: {label.get('name', 'unknown')}")
            print(f"      3D centre: ({label['center'][0]:.2f}, {label['center'][1]:.2f}, {label['center'][2]:.2f})")
            print(f"      size: ({label['size'][0]:.2f}, {label['size'][1]:.2f}, {label['size'][2]:.2f})")
            print(f"      BBox 2D: {label['bbox_2d']}")

            center = label['center']

            # Does the position make sense?
            if abs(center[0]) > 100 or abs(center[1]) > 100 or abs(center[2]) > 100:
                print("      WARNING: position very far away (>100 m)")

            if center[2] < 1:  # Z far too small
                print("      WARNING: drone very close (Z < 1 m)")

            # Does the size make sense for a drone?
            size = label['size']
            if any(s > 5 for s in size):
                print("      WARNING: size too large for a drone (>5 m)")
            if any(s < 0.1 for s in size):
                print("      WARNING: size too small (<0.1 m)")
    else:
        print("   no 3D labels")

    # Point-cloud statistics
    if len(point_cloud) > 0:
        print("\n   Point-cloud statistics:")
        print(f"      X: min={point_cloud[:, 0].min():.2f}, max={point_cloud[:, 0].max():.2f}")
        print(f"      Y: min={point_cloud[:, 1].min():.2f}, max={point_cloud[:, 1].max():.2f}")
        print(f"      Z: min={point_cloud[:, 2].min():.2f}, max={point_cloud[:, 2].max():.2f}")


def plot_frame(pc_file, labels_3d, out_path, max_points):
    """Write the 3D-scatter + top-down figure for one frame."""
    sample_frame = pc_file.stem
    point_cloud = np.load(pc_file)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: the point cloud with 3D boxes
    ax1 = fig.add_subplot(121, projection='3d')

    # Subsample the points for plotting (too many points is slow)
    sample_size = min(max_points, len(point_cloud))
    indices = np.random.choice(len(point_cloud), sample_size, replace=False)
    sampled_points = point_cloud[indices]

    ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                c=sampled_points[:, 2], cmap='viridis', s=0.1, alpha=0.5)

    for label in labels_3d:
        center = label['center']
        size = label['size']

        # The 8 vertices of the bounding box
        x_range = [center[0] - size[0]/2, center[0] + size[0]/2]
        y_range = [center[1] - size[1]/2, center[1] + size[1]/2]
        z_range = [center[2] - size[2]/2, center[2] + size[2]/2]

        # Draw the edges of the box
        for x in x_range:
            for y in y_range:
                ax1.plot([x, x], [y, y], z_range, 'r-', linewidth=2)

        for x in x_range:
            for z in z_range:
                ax1.plot([x, x], y_range, [z, z], 'r-', linewidth=2)

        for y in y_range:
            for z in z_range:
                ax1.plot(x_range, [y, y], [z, z], 'r-', linewidth=2)

        # Mark the centre
        ax1.scatter([center[0]], [center[1]], [center[2]],
                    c='red', s=100, marker='x')

        # Add the name as text
        ax1.text(center[0], center[1], center[2],
                 label.get('name', 'drone'), fontsize=8)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'Point cloud with 3D bounding boxes\n{sample_frame}')

    # Plot 2: top-down view (XY)
    ax2 = fig.add_subplot(122)
    ax2.scatter(sampled_points[:, 0], sampled_points[:, 1],
                c=sampled_points[:, 2], cmap='viridis', s=0.1, alpha=0.5)

    for label in labels_3d:
        center = label['center']
        size = label['size']

        rect = plt.Rectangle((center[0] - size[0]/2, center[1] - size[1]/2),
                             size[0], size[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)

        ax2.plot(center[0], center[1], 'rx', markersize=10)
        ax2.text(center[0], center[1], label.get('name', 'drone'), fontsize=8)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-down view (XY)')
    ax2.axis('equal')
    ax2.grid(True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    print(f"\nVisualisation saved: {out_path}")


def check_2d_3d_consistency(dataset_dir, sample_frame, labels_3d):
    """Check that each 3D centre falls on the correct side of the image."""
    print("\nVALIDATING THE 2D-3D CORRESPONDENCE")
    print("-" * 50)

    img_path = dataset_dir / "visualizations" / f"{sample_frame}.jpg"
    if not img_path.exists():
        print(f"   no reference image at {img_path}; skipped")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"   could not read {img_path}; skipped")
        return

    for i, label in enumerate(labels_3d):
        bbox_2d = label['bbox_2d']
        center_3d = label['center']

        print(f"\n   Drone {i+1} ({label.get('name', 'unknown')}):")
        print(f"      BBox 2D: [{bbox_2d[0]}, {bbox_2d[1]}] to [{bbox_2d[2]}, {bbox_2d[3]}]")
        print(f"      3D centre: ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})")

        center_2d_x = (bbox_2d[0] + bbox_2d[2]) / 2
        center_2d_y = (bbox_2d[1] + bbox_2d[3]) / 2

        img_height, img_width = img.shape[:2]
        rel_x = center_2d_x / img_width
        rel_y = center_2d_y / img_height

        print(f"      relative 2D centre: ({rel_x:.2%}, {rel_y:.2%})")

        if center_3d[1] > 0 and rel_x < 0.5:
            print("      Inconsistent: drone to the right (Y>0) but box to the left")
        elif center_3d[1] < 0 and rel_x > 0.5:
            print("      Inconsistent: drone to the left (Y<0) but box to the right")
        else:
            print("      2D-3D position consistent")


def main(argv=None):
    args = parse_args(argv)

    print("\n" + "=" * 70)
    print("VALIDATION OF THE 3D LABELS")
    print("=" * 70)

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"dataset not found: {dataset_dir}")

    pointcloud_dir = dataset_dir / "pointnet" / "point_clouds"
    labels_3d_dir = dataset_dir / "pointnet" / "labels_3d"

    pointcloud_files = sorted(pointcloud_dir.glob("*.npy"))[:args.frames]
    if not pointcloud_files:
        raise SystemExit(f"no point cloud found under {pointcloud_dir}")

    print(f"Validating {len(pointcloud_files)} frames...")

    for pc_file in pointcloud_files:
        report_frame(pc_file, labels_3d_dir)

    print("\n" + "=" * 70)
    print("BUILDING THE 3D VISUALISATION")
    print("=" * 70)

    # Take one frame for a detailed view
    sample_pc = pointcloud_files[0]
    sample_frame = sample_pc.stem
    labels_file = labels_3d_dir / f"{sample_frame}.json"

    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_3d = json.load(f)
        plot_frame(sample_pc, labels_3d, args.out, args.max_points)
        check_2d_3d_consistency(dataset_dir, sample_frame, labels_3d)
    else:
        print(f"no 3D labels for {sample_frame}; nothing plotted")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
