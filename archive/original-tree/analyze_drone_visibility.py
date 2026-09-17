#!/usr/bin/env python3
"""
Script para analisar visibilidade de drones em imagens de dataset
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse

def analyze_segmentation(seg_path):
    """Analisa imagem de segmentação para encontrar drones"""
    seg = cv2.imread(str(seg_path))

    if seg is None:
        return None

    # Cores típicas de drones na segmentação
    drone_colors = [
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
    ]

    results = {}
    total_pixels = seg.shape[0] * seg.shape[1]

    for color in drone_colors:
        # Cria máscara para esta cor
        mask = cv2.inRange(seg, color, color)
        pixel_count = np.sum(mask > 0)

        if pixel_count > 0:
            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if area > 50:  # Ignora objetos muito pequenos
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
    """Cria visualização com bounding boxes"""
    rgb = cv2.imread(str(rgb_path))
    seg = cv2.imread(str(seg_path))

    if rgb is None or seg is None:
        return False

    vis = rgb.copy()

    # Analisa segmentação
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

    # Salva visualização
    cv2.imwrite(str(output_path), vis)
    return True

def analyze_dataset(dataset_dir):
    """Analisa dataset completo"""
    dataset_path = Path(dataset_dir)

    # Encontra imagens
    rgb_dir = dataset_path / 'images' / 'front_center'
    seg_dir = dataset_path / 'seg' / 'front_center'

    if not rgb_dir.exists():
        rgb_dir = dataset_path / 'rgb'
        seg_dir = dataset_path / 'segmentation'

    if not rgb_dir.exists():
        print(f"❌ Não encontrado diretório de imagens em {dataset_path}")
        return

    rgb_files = sorted(rgb_dir.glob('*.png'))
    print(f"📁 Analisando {len(rgb_files)} imagens em {dataset_path}")

    # Cria diretório de saída
    output_dir = dataset_path / 'analysis'
    output_dir.mkdir(exist_ok=True)

    # Estatísticas gerais
    stats = {
        'total_frames': len(rgb_files),
        'frames_with_drones': 0,
        'total_drone_detections': 0,
        'avg_drone_size': [],
        'drone_visibility': []
    }

    print("\n🔍 Analisando frames...")

    for i, rgb_path in enumerate(rgb_files[:50]):  # Analisa primeiros 50 frames
        seg_path = seg_dir / rgb_path.name

        if not seg_path.exists():
            continue

        # Analisa segmentação
        analysis = analyze_segmentation(seg_path)

        if analysis:
            stats['frames_with_drones'] += 1

            for color, data in analysis.items():
                stats['total_drone_detections'] += data['num_objects']

                for obj in data['objects']:
                    stats['avg_drone_size'].append(obj['area'])
                    stats['drone_visibility'].append(obj['percentage'])

            # Cria visualização para alguns frames
            if i % 10 == 0:
                vis_path = output_dir / f"vis_{rgb_path.stem}.png"
                create_visualization(rgb_path, seg_path, vis_path)
                print(f"   Frame {i}: {len(analysis)} cores de drone detectadas")

    # Calcula estatísticas finais
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

    # Salva estatísticas
    stats_file = output_dir / 'visibility_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Imprime resumo
    print("\n" + "="*60)
    print("📊 RESUMO DA ANÁLISE")
    print("="*60)
    print(f"Total de frames analisados: {stats['total_frames']}")
    print(f"Frames com drones visíveis: {stats['frames_with_drones']} ({stats['frames_with_drones']/stats['total_frames']*100:.1f}%)")
    print(f"Total de detecções: {stats['total_drone_detections']}")

    if stats['avg_drone_size'] > 0:
        print(f"\n📐 Tamanho médio do drone: {stats['avg_drone_size']:.0f} pixels")
        print(f"📊 Visibilidade média: {stats['avg_visibility']:.3f}%")
        print(f"📈 Visibilidade máxima: {stats['max_visibility']:.3f}%")
        print(f"📉 Visibilidade mínima: {stats['min_visibility']:.3f}%")

    print(f"\n💾 Estatísticas salvas em: {stats_file}")
    print(f"🖼️ Visualizações salvas em: {output_dir}")

    # Recomendações
    print("\n💡 RECOMENDAÇÕES:")
    if stats['frames_with_drones'] < stats['total_frames'] * 0.5:
        print("   ⚠️ Poucos frames com drones visíveis!")
        print("   • Aproxime os drones da câmera (15-25m)")
        print("   • Use formações que mantêm drones no campo de visão")

    if stats['avg_visibility'] < 0.1:
        print("   ⚠️ Drones muito pequenos nas imagens!")
        print("   • Reduza distância entre drones e câmera")
        print("   • Considere aumentar o tamanho dos drones no simulador")

    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='dataset',
                       help='Diretório do dataset a analisar')
    args = parser.parse_args()

    analyze_dataset(args.dataset)