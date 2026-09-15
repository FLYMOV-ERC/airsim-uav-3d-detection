#!/usr/bin/env python3
"""ARCHIVED. LiDAR-camera synchronization analyser.

Quantifies the drift between the channels -- the measurement behind Section
7.2.1's "about one meter in the half-second spanned by the successive RPC
queries", and behind the decision to freeze the scene with simPause.
"""

import numpy as np
import cv2
from pathlib import Path
import json

def analyze_frame(frame_idx):
    """Analyse one specific frame."""
    dataset_dir = Path("dataset_300_fusion_fixed")

    # Load the images
    rgb_path = dataset_dir / "images_rgb" / f"frame_{frame_idx:04d}.png"
    fusion_path = dataset_dir / "images_fusion" / f"frame_{frame_idx:04d}.png"
    metadata_path = dataset_dir / "metadata" / f"frame_{frame_idx:04d}.json"
    lidar_path = dataset_dir / "lidar_points" / f"frame_{frame_idx:04d}.npy"

    if not all([rgb_path.exists(), fusion_path.exists(), metadata_path.exists(), lidar_path.exists()]):
        print(f"Frame {frame_idx:04d} is incomplete")
        return None

    # Load the data
    img_rgb = cv2.imread(str(rgb_path))
    img_fusion = cv2.imread(str(fusion_path))
    points = np.load(lidar_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n{'='*60}")
    print(f"Frame {frame_idx:04d} - {metadata['scenario']}")
    print(f"{'='*60}")
    print(f"Ego altitude: {metadata['ego_altitude_meters']}m")
    print(f"Yaw: {metadata['yaw']}°")
    print(f"Total points: {metadata['total_points']}")
    print(f"Front points: {metadata['front_points']}")
    print(f"Anomalies detected: {metadata['anomalies']}")
    print(f"Detections: {metadata['detections']}")

    print("\nDrone positions:")
    for drone in metadata['drone_positions']:
        print(f"  {drone['name']}: x={drone['x']}, y={drone['y']}, altitude={drone['altitude_meters']}m")

    # Analyse the distribution of the LiDAR points
    if len(points) > 0:
        x_range = (points[:, 0].min(), points[:, 0].max())
        y_range = (points[:, 1].min(), points[:, 1].max())
        z_range = (points[:, 2].min(), points[:, 2].max())

        print(f"\nLiDAR point cloud analysis:")
        print(f"  X range: {x_range[0]:.2f} to {x_range[1]:.2f}")
        print(f"  Y range: {y_range[0]:.2f} to {y_range[1]:.2f}")
        print(f"  Z range: {z_range[0]:.2f} to {z_range[1]:.2f}")

        # Keep the forward points
        front_mask = points[:, 0] > 0.5
        front_points = points[front_mask]

        if len(front_points) > 0:
            z_mean = front_points[:, 2].mean()
            z_std = front_points[:, 2].std()
            print(f"  Front points Z mean: {z_mean:.2f}, std: {z_std:.2f}")

            # Detecta anomalias
            anomalies_mask = np.abs(front_points[:, 2] - z_mean) > 2
            anomaly_points = front_points[anomalies_mask]

            if len(anomaly_points) > 0:
                print(f"\nAnomaly points (potential drones):")
                for i, pt in enumerate(anomaly_points[:5]):  # show up to five points
                    print(f"  Point {i}: x={pt[0]:.2f}, y={pt[1]:.2f}, z={pt[2]:.2f}")

    return {
        'rgb': img_rgb,
        'fusion': img_fusion,
        'metadata': metadata,
        'points': points
    }

def visualize_sync_issue():
    """Visualise the synchronization problem."""
    print("Analysing the LiDAR-camera synchronization problem")
    print("="*60)

    # Analyse a few specific frames
    frames_to_check = [0, 50, 100, 150, 200, 250, 299]

    for frame_idx in frames_to_check:
        data = analyze_frame(frame_idx)

        if data is not None:
            # Show the visual comparison
            comparison = np.hstack([data['rgb'], data['fusion']])

            # Adiciona texto informativo
            cv2.putText(comparison, f"Frame {frame_idx:04d}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(comparison, "RGB",
                       (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "FUSION",
                       (50 + 1280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.imshow("Sync Analysis", comparison)

            print("\nPressione:")
            print("  'n' for the next frame")
            print("  'q' to quit")

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_sync_issue()