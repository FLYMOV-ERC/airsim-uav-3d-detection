#!/usr/bin/env python3
"""
analyze_dataset.py
Analyse and visualise a collected dataset.
"""

import json
import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import random


def analyze_metadata(dataset_path):
    """Analyse the dataset metadata."""
    meta_dir = Path(dataset_path) / "meta"

    if not meta_dir.exists():
        print(f"Metadata directory not found: {meta_dir}")
        return None

    # Statistics
    stats = {
        'total_frames': 0,
        'vehicles_per_frame': defaultdict(int),
        'weather_conditions': defaultdict(int),
        'time_conditions': defaultdict(int),
        'movement_patterns': defaultdict(int),
        'drone_positions': defaultdict(list),
        'has_lidar': 0,
        'cameras': set()
    }

    # Analyse each metadata file
    meta_files = sorted(meta_dir.glob("*.json"))
    stats['total_frames'] = len(meta_files)

    print(f"Analysing {stats['total_frames']} frames...")

    for meta_file in meta_files[:min(100, len(meta_files))]:  # analyse the first 100 frames
        with open(meta_file, 'r') as f:
            data = json.load(f)

        # Count the vehicles
        num_vehicles = len(data.get('vehicles', {}))
        stats['vehicles_per_frame'][num_vehicles] += 1

        # Drone positions
        for vehicle, info in data.get('vehicles', {}).items():
            if 'pose' in info and 'position' in info['pose']:
                pos = info['pose']['position']
                stats['drone_positions'][vehicle].append({
                    'x': pos['x'],
                    'y': pos['y'],
                    'z': pos['z']
                })

        # Environmental conditions (when available)
        if 'conditions' in data:
            conditions = data['conditions']
            if 'weather' in conditions:
                stats['weather_conditions'][conditions['weather']] += 1
            if 'time' in conditions:
                stats['time_conditions'][conditions['time']] += 1

        # Motion patterns (when available)
        if 'movement_pattern' in data:
            stats['movement_patterns'][data['movement_pattern']] += 1

        # LiDAR
        if 'lidar_points' in data and data['lidar_points'] > 0:
            stats['has_lidar'] += 1

        # Cameras
        if 'cams' in data:
            stats['cameras'].update(data['cams'])

    return stats


def visualize_drone_trajectories(stats, output_file='trajectories.png'):
    """Plot the drone trajectories."""
    if not stats or not stats['drone_positions']:
        print("No position data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drone trajectory analysis', fontsize=16)

    # One colour per drone
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    # Vista superior (XY)
    ax = axes[0, 0]
    for i, (drone, positions) in enumerate(stats['drone_positions'].items()):
        if positions:
            x = [p['x'] for p in positions]
            y = [p['y'] for p in positions]
            ax.scatter(x, y, c=colors[i % len(colors)], label=drone, alpha=0.6, s=20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Vista Superior (XY)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Vista lateral (XZ)
    ax = axes[0, 1]
    for i, (drone, positions) in enumerate(stats['drone_positions'].items()):
        if positions:
            x = [p['x'] for p in positions]
            z = [p['z'] for p in positions]
            ax.scatter(x, z, c=colors[i % len(colors)], label=drone, alpha=0.6, s=20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Vista Lateral (XZ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Altitude distribution
    ax = axes[1, 0]
    for i, (drone, positions) in enumerate(stats['drone_positions'].items()):
        if positions:
            z = [p['z'] for p in positions]
            ax.hist(z, bins=20, alpha=0.5, label=drone, color=colors[i % len(colors)])
    ax.set_xlabel('Altitude Z (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Altitude distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overall statistics
    ax = axes[1, 1]
    ax.axis('off')

    info_text = f"Dataset statistics\n\n"
    info_text += f"Total de frames: {stats['total_frames']}\n"
    info_text += f"Cameras: {', '.join(stats['cameras']) if stats['cameras'] else 'N/A'}\n"
    info_text += f"Frames with LiDAR: {stats['has_lidar']}\n\n"

    info_text += "Vehicles per frame:\n"
    for num_vehicles, count in sorted(stats['vehicles_per_frame'].items()):
        info_text += f"  {num_vehicles} vehicles: {count} frames\n"

    if stats['weather_conditions']:
        info_text += "\nWeather conditions:\n"
        for weather, count in stats['weather_conditions'].items():
            info_text += f"  {weather}: {count} frames\n"

    if stats['time_conditions']:
        info_text += "\nTimes of day:\n"
        for time_cond, count in list(stats['time_conditions'].items())[:5]:
            info_text += f"  {time_cond}: {count} frames\n"

    if stats['movement_patterns']:
        info_text += "\nMotion patterns:\n"
        for pattern, count in stats['movement_patterns'].items():
            info_text += f"  {pattern}: {count} frames\n"

    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def check_images_samples(dataset_path, num_samples=5):
    """Inspect sample RGB and segmentation images."""
    images_dir = Path(dataset_path) / "images"
    seg_dir = Path(dataset_path) / "seg"

    if not images_dir.exists():
        print(f"Image directory not found: {images_dir}")
        return

    # Available cameras
    cameras = [d.name for d in images_dir.iterdir() if d.is_dir()]

    print(f"\nInspecting images from {len(cameras)} camera(s)...")

    for camera in cameras[:1]:  # inspect the first camera only
        img_cam_dir = images_dir / camera
        seg_cam_dir = seg_dir / camera

        img_files = sorted(img_cam_dir.glob("*.png"))
        print(f"\nCamera '{camera}': {len(img_files)} images")

        if not img_files:
            continue

        # Build the sample mosaic
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        fig.suptitle(f'Samples - camera: {camera}', fontsize=14)

        # Select random frames
        sample_indices = sorted(random.sample(range(len(img_files)), min(num_samples, len(img_files))))

        for i, idx in enumerate(sample_indices):
            # RGB image
            img_path = img_files[idx]
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Frame {idx+1}')
                axes[0, i].axis('off')

                # Check the size
                if i == 0:
                    print(f"  Resolution: {img.shape[1]}x{img.shape[0]}")

            # Segmentation image
            seg_path = seg_cam_dir / img_path.name
            if seg_path.exists():
                seg = cv2.imread(str(seg_path))
                if seg is not None:
                    axes[1, i].imshow(seg)
                    axes[1, i].set_title('Segmentation')
                    axes[1, i].axis('off')

        plt.tight_layout()
        output_file = f'samples_{camera}.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Samples saved to: {output_file}")
        plt.close()


def check_lidar_data(dataset_path, sample_frame=1):
    """Inspect the LiDAR data."""
    lidar_dir = Path(dataset_path) / "lidar"

    if not lidar_dir.exists():
        print(f"LiDAR directory not found: {lidar_dir}")
        return

    lidar_files = sorted(lidar_dir.glob("*.npy"))
    print(f"\nLiDAR: {len(lidar_files)} files")

    if lidar_files:
        # Load a sample
        sample_file = lidar_files[min(sample_frame-1, len(lidar_files)-1)]
        points = np.load(sample_file)

        print(f"  sample: {sample_file.name}")
        print(f"  points: {points.shape}")
        print(f"  Range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Range Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        # Plot the point cloud
        if len(points) > 0:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Subsample for plotting
            step = max(1, len(points) // 5000)
            points_vis = points[::step]

            scatter = ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                               c=points_vis[:, 2], cmap='viridis', s=0.5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'LiDAR point cloud - {len(points_vis)} points (subsampled)')

            plt.colorbar(scatter, ax=ax, label='Z (m)')
            plt.tight_layout()
            plt.savefig('lidar_sample.png', dpi=100, bbox_inches='tight')
            print(f"LiDAR visualisation saved to: lidar_sample.png")
            plt.close()


def analyze_calibration(dataset_path):
    """Analyse the calibration file."""
    calib_file = Path(dataset_path) / "calibration.json"

    if not calib_file.exists():
        print(f"Calibration file not found: {calib_file}")
        return

    with open(calib_file, 'r') as f:
        calib = json.load(f)

    print("\nCalibration:")
    print(f"  Main vehicle: {calib.get('vehicle', 'N/A')}")

    if 'intruders' in calib:
        print(f"  Intruders: {', '.join(calib['intruders'])}")

    if 'image_size' in calib:
        print(f"  Image size: {calib['image_size'][0]}x{calib['image_size'][1]}")

    if 'cams' in calib:
        print(f"  Cameras configured:")
        for cam_name, cam_info in calib['cams'].items():
            print(f"    - {cam_name}: FOV={cam_info.get('fov_deg', 'N/A')}°")
            if 'intrinsics' in cam_info:
                intr = cam_info['intrinsics']
                print(f"      fx={intr['fx']:.2f}, fy={intr['fy']:.2f}")
                print(f"      cx={intr['cx']:.2f}, cy={intr['cy']:.2f}")


def main():
    """Entry point of the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyse a dataset collected from AirSim")
    parser.add_argument('--dataset', default='dataset', help='Caminho do dataset')
    parser.add_argument('--samples', type=int, default=5, help='number of image samples')
    parser.add_argument('--max_frames', type=int, default=100, help='maximum number of frames to analyse')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print("="*60)
    print(f"DATASET ANALYSIS: {dataset_path}")
    print("="*60)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # 1. Analyse the calibration
    analyze_calibration(dataset_path)

    # 2. Analyse the metadata
    stats = analyze_metadata(dataset_path)

    # 3. Plot the trajectories
    if stats:
        visualize_drone_trajectories(stats)

    # 4. Inspect the image samples
    check_images_samples(dataset_path, num_samples=args.samples)

    # 5. Inspect the LiDAR
    check_lidar_data(dataset_path)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nFiles generated:")
    print("  - trajectories.png: drone trajectories")
    print("  - samples_*.png: image samples")
    print("  - lidar_sample.png: LiDAR visualisation")


if __name__ == "__main__":
    main()