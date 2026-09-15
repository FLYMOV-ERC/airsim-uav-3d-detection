#!/usr/bin/env python3
"""ARCHIVED. Analyse drone visibility in the collected frames.

Part of the investigation that concluded the native multirotor mesh was too small
to be detected at range, which led to the VisQuad 4x visual mesh (Section 7.2.2).
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse

def analyze_segmentation(seg_path):
    """Analyse a segmentation image to find the drones."""
    seg = cv2.imread(str(seg_path))

    if seg is None:
        return None

    # Colours typically taken by drones in the segmentation
    drone_colors = [
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
    ]

    results = {}
    total_pixels = seg.shape[0] * seg.shape[1]

    for color in drone_colors:
        # Build a mask for this colour
        mask = cv2.inRange(seg, color, color)
        pixel_count = np.sum(mask > 0)

        if pixel_count > 0:
            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if area > 50:  # ignore very small objects
                    objects.append({
                        'bbox': [x, y, w, h],
                        'area': area,
                        'percentage': (area / total_pixels) * 100
                    })

            if objects:
                results[str(color)] = {
                    'pixel_count': int(pixel_count),
                    'percentage': (pixel_count / total_pixels) * 100,
                    'objects': objects,
                    'num_objects': len(objects)
                }

    return results

def create_visualization(rgb_path, seg_path, output_path):
    """Render a visualisation with the bounding boxes."""
    rgb = cv2.imread(str(rgb_path))
    seg = cv2.imread(str(seg_path))

    if rgb is None or seg is None:
        return False

    vis = rgb.copy()

    # Analyse the segmentation
    analysis = analyze_segmentation(seg_path)

    if analysis:
        for color_str, data in analysis.items():
            for obj in data['objects']:
                x, y, w, h = obj['bbox']

                # Desenha bounding box
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Adiciona label
                label = f"Drone {obj['percentage']:.2f}%"
                cv2.putText(vis, label, (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the visualisation
    cv2.imwrite(str(output_path), vis)
    return True

def analyze_dataset(dataset_dir):
    """Analisa dataset completo"""
    dataset_path = Path(dataset_dir)

    # Find the images
    rgb_dir = dataset_path / 'images' / 'front_center'
    seg_dir = dataset_path / 'seg' / 'front_center'

    if not rgb_dir.exists():
        rgb_dir = dataset_path / 'rgb'
        seg_dir = dataset_path / 'segmentation'

    if not rgb_dir.exists():
        print(f"No image directory found in {dataset_path}")
        return

    rgb_files = sorted(rgb_dir.glob('*.png'))
    print(f"Analysing {len(rgb_files)} images in {dataset_path}")

    # Create the output directory
    output_dir = dataset_path / 'analysis'
    output_dir.mkdir(exist_ok=True)

    # Overall statistics
    stats = {
        'total_frames': len(rgb_files),
        'frames_with_drones': 0,
        'total_drone_detections': 0,
        'avg_drone_size': [],
        'drone_visibility': []
    }

    print("\nAnalysing the frames...")

    for i, rgb_path in enumerate(rgb_files[:50]):  # Analisa primeiros 50 frames
        seg_path = seg_dir / rgb_path.name

        if not seg_path.exists():
            continue

        # Analyse the segmentation
        analysis = analyze_segmentation(seg_path)

        if analysis:
            stats['frames_with_drones'] += 1

            for color, data in analysis.items():
                stats['total_drone_detections'] += data['num_objects']

                for obj in data['objects']:
                    stats['avg_drone_size'].append(obj['area'])
                    stats['drone_visibility'].append(obj['percentage'])

            # Build a visualisation for a few frames
            if i % 10 == 0:
                vis_path = output_dir / f"vis_{rgb_path.stem}.png"
                create_visualization(rgb_path, seg_path, vis_path)
                print(f"   Frame {i}: {len(analysis)} drone colours detected")

    # Compute the final statistics
    if stats['avg_drone_size']:
        stats['avg_drone_size'] = np.mean(stats['avg_drone_size'])
        stats['avg_visibility'] = np.mean(stats['drone_visibility'])
        stats['max_visibility'] = np.max(stats['drone_visibility'])
        stats['min_visibility'] = np.min(stats['drone_visibility'])
    else:
        stats['avg_drone_size'] = 0
        stats['avg_visibility'] = 0
        stats['max_visibility'] = 0
        stats['min_visibility'] = 0

    # Save the statistics
    stats_file = output_dir / 'visibility_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Imprime resumo
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total de frames analisados: {stats['total_frames']}")
    print(f"Frames with visible drones: {stats['frames_with_drones']} ({stats['frames_with_drones']/stats['total_frames']*100:.1f}%)")
    print(f"Total detections: {stats['total_drone_detections']}")

    if stats['avg_drone_size'] > 0:
        print(f"\nMean drone size: {stats['avg_drone_size']:.0f} px")
        print(f"Mean visibility: {stats['avg_visibility']:.3f}%")
        print(f"Maximum visibility: {stats['max_visibility']:.3f}%")
        print(f"Minimum visibility: {stats['min_visibility']:.3f}%")

    print(f"\nStatistics saved to: {stats_file}")
    print(f"Visualisations saved to: {output_dir}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if stats['frames_with_drones'] < stats['total_frames'] * 0.5:
        print("   Few frames contain visible drones.")
        print("   - move the drones closer to the camera (15-25 m)")
        print("   - use formations that keep the drones inside the field of view")

    if stats['avg_visibility'] < 0.1:
        print("   The drones are very small in the images.")
        print("   - reduce the distance between the drones and the camera")
        print("   - consider enlarging the drones in the simulator")

    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='dataset',
                       help='dataset directory to analyse')
    args = parser.parse_args()

    analyze_dataset(args.dataset)