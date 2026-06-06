#!/usr/bin/env python3
"""
analyze_dataset.py
Script para analisar e visualizar o dataset gerado
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
    """Analisa os metadados do dataset"""
    meta_dir = Path(dataset_path) / "meta"

    if not meta_dir.exists():
        print(f"❌ Diretório de metadados não encontrado: {meta_dir}")
        return None

    # Estatísticas
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

    # Analisar cada arquivo de metadados
    meta_files = sorted(meta_dir.glob("*.json"))
    stats['total_frames'] = len(meta_files)

    print(f"📊 Analisando {stats['total_frames']} frames...")

    for meta_file in meta_files[:min(100, len(meta_files))]:  # Analisar primeiros 100 frames
        with open(meta_file, 'r') as f:
            data = json.load(f)

        # Contar veículos
        num_vehicles = len(data.get('vehicles', {}))
        stats['vehicles_per_frame'][num_vehicles] += 1

        # Posições dos drones
        for vehicle, info in data.get('vehicles', {}).items():
            if 'pose' in info and 'position' in info['pose']:
                pos = info['pose']['position']
                stats['drone_positions'][vehicle].append({
                    'x': pos['x'],
                    'y': pos['y'],
                    'z': pos['z']
                })

        # Condições ambientais (se disponível)
        if 'conditions' in data:
            conditions = data['conditions']
            if 'weather' in conditions:
                stats['weather_conditions'][conditions['weather']] += 1
            if 'time' in conditions:
                stats['time_conditions'][conditions['time']] += 1

        # Padrões de movimento (se disponível)
        if 'movement_pattern' in data:
            stats['movement_patterns'][data['movement_pattern']] += 1

        # LiDAR
        if 'lidar_points' in data and data['lidar_points'] > 0:
            stats['has_lidar'] += 1

        # Câmeras
        if 'cams' in data:
            stats['cameras'].update(data['cams'])

    return stats


def visualize_drone_trajectories(stats, output_file='trajectories.png'):
    """Visualiza as trajetórias dos drones"""
    if not stats or not stats['drone_positions']:
        print("❌ Sem dados de posição para visualizar")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análise de Trajetórias dos Drones', fontsize=16)

    # Cores para cada drone
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

    # Distribuição de altitudes
    ax = axes[1, 0]
    for i, (drone, positions) in enumerate(stats['drone_positions'].items()):
        if positions:
            z = [p['z'] for p in positions]
            ax.hist(z, bins=20, alpha=0.5, label=drone, color=colors[i % len(colors)])
    ax.set_xlabel('Altitude Z (m)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição de Altitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Estatísticas gerais
    ax = axes[1, 1]
    ax.axis('off')

    info_text = f"📊 Estatísticas do Dataset\n\n"
    info_text += f"Total de frames: {stats['total_frames']}\n"
    info_text += f"Câmeras: {', '.join(stats['cameras']) if stats['cameras'] else 'N/A'}\n"
    info_text += f"Frames com LiDAR: {stats['has_lidar']}\n\n"

    info_text += "Veículos por frame:\n"
    for num_vehicles, count in sorted(stats['vehicles_per_frame'].items()):
        info_text += f"  {num_vehicles} veículos: {count} frames\n"

    if stats['weather_conditions']:
        info_text += "\n🌤️ Condições climáticas:\n"
        for weather, count in stats['weather_conditions'].items():
            info_text += f"  {weather}: {count} frames\n"

    if stats['time_conditions']:
        info_text += "\n🕐 Horários do dia:\n"
        for time_cond, count in list(stats['time_conditions'].items())[:5]:
            info_text += f"  {time_cond}: {count} frames\n"

    if stats['movement_patterns']:
        info_text += "\n🚁 Padrões de movimento:\n"
        for pattern, count in stats['movement_patterns'].items():
            info_text += f"  {pattern}: {count} frames\n"

    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {output_file}")
    plt.close()


def check_images_samples(dataset_path, num_samples=5):
    """Verifica amostras de imagens RGB e segmentação"""
    images_dir = Path(dataset_path) / "images"
    seg_dir = Path(dataset_path) / "seg"

    if not images_dir.exists():
        print(f"❌ Diretório de imagens não encontrado: {images_dir}")
        return

    # Pegar câmeras disponíveis
    cameras = [d.name for d in images_dir.iterdir() if d.is_dir()]

    print(f"\n📷 Verificando imagens de {len(cameras)} câmera(s)...")

    for camera in cameras[:1]:  # Verificar primeira câmera
        img_cam_dir = images_dir / camera
        seg_cam_dir = seg_dir / camera

        img_files = sorted(img_cam_dir.glob("*.png"))
        print(f"\nCâmera '{camera}': {len(img_files)} imagens")

        if not img_files:
            continue

        # Criar mosaico de amostras
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        fig.suptitle(f'Amostras - Câmera: {camera}', fontsize=14)

        # Selecionar frames aleatórios
        sample_indices = sorted(random.sample(range(len(img_files)), min(num_samples, len(img_files))))

        for i, idx in enumerate(sample_indices):
            # Imagem RGB
            img_path = img_files[idx]
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Frame {idx+1}')
                axes[0, i].axis('off')

                # Verificar tamanho
                if i == 0:
                    print(f"  Resolução: {img.shape[1]}x{img.shape[0]}")

            # Imagem de segmentação
            seg_path = seg_cam_dir / img_path.name
            if seg_path.exists():
                seg = cv2.imread(str(seg_path))
                if seg is not None:
                    axes[1, i].imshow(seg)
                    axes[1, i].set_title('Segmentação')
                    axes[1, i].axis('off')

        plt.tight_layout()
        output_file = f'samples_{camera}.png'
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"✅ Amostras salvas em: {output_file}")
        plt.close()


def check_lidar_data(dataset_path, sample_frame=1):
    """Verifica dados LiDAR"""
    lidar_dir = Path(dataset_path) / "lidar"

    if not lidar_dir.exists():
        print(f"❌ Diretório LiDAR não encontrado: {lidar_dir}")
        return

    lidar_files = sorted(lidar_dir.glob("*.npy"))
    print(f"\n📡 LiDAR: {len(lidar_files)} arquivos")

    if lidar_files:
        # Carregar amostra
        sample_file = lidar_files[min(sample_frame-1, len(lidar_files)-1)]
        points = np.load(sample_file)

        print(f"  Amostra: {sample_file.name}")
        print(f"  Pontos: {points.shape}")
        print(f"  Range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Range Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        # Visualizar nuvem de pontos
        if len(points) > 0:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Subsample para visualização
            step = max(1, len(points) // 5000)
            points_vis = points[::step]

            scatter = ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                               c=points_vis[:, 2], cmap='viridis', s=0.5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Nuvem de Pontos LiDAR - {len(points_vis)} pontos (subsampled)')

            plt.colorbar(scatter, ax=ax, label='Z (m)')
            plt.tight_layout()
            plt.savefig('lidar_sample.png', dpi=100, bbox_inches='tight')
            print(f"✅ Visualização LiDAR salva em: lidar_sample.png")
            plt.close()


def analyze_calibration(dataset_path):
    """Analisa arquivo de calibração"""
    calib_file = Path(dataset_path) / "calibration.json"

    if not calib_file.exists():
        print(f"❌ Arquivo de calibração não encontrado: {calib_file}")
        return

    with open(calib_file, 'r') as f:
        calib = json.load(f)

    print("\n🎯 Calibração:")
    print(f"  Veículo principal: {calib.get('vehicle', 'N/A')}")

    if 'intruders' in calib:
        print(f"  Intruders: {', '.join(calib['intruders'])}")

    if 'image_size' in calib:
        print(f"  Tamanho da imagem: {calib['image_size'][0]}x{calib['image_size'][1]}")

    if 'cams' in calib:
        print(f"  Câmeras configuradas:")
        for cam_name, cam_info in calib['cams'].items():
            print(f"    - {cam_name}: FOV={cam_info.get('fov_deg', 'N/A')}°")
            if 'intrinsics' in cam_info:
                intr = cam_info['intrinsics']
                print(f"      fx={intr['fx']:.2f}, fy={intr['fy']:.2f}")
                print(f"      cx={intr['cx']:.2f}, cy={intr['cy']:.2f}")


def main():
    """Função principal de análise"""
    import argparse

    parser = argparse.ArgumentParser(description="Analisa dataset gerado pelo AirSim")
    parser.add_argument('--dataset', default='dataset', help='Caminho do dataset')
    parser.add_argument('--samples', type=int, default=5, help='Número de amostras de imagem')
    parser.add_argument('--max_frames', type=int, default=100, help='Máximo de frames para analisar')
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print("="*60)
    print(f"🔍 ANÁLISE DO DATASET: {dataset_path}")
    print("="*60)

    if not dataset_path.exists():
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return

    # 1. Analisar calibração
    analyze_calibration(dataset_path)

    # 2. Analisar metadados
    stats = analyze_metadata(dataset_path)

    # 3. Visualizar trajetórias
    if stats:
        visualize_drone_trajectories(stats)

    # 4. Verificar amostras de imagens
    check_images_samples(dataset_path, num_samples=args.samples)

    # 5. Verificar LiDAR
    check_lidar_data(dataset_path)

    print("\n" + "="*60)
    print("✅ ANÁLISE COMPLETA!")
    print("="*60)
    print("\nArquivos gerados:")
    print("  - trajectories.png: Trajetórias dos drones")
    print("  - samples_*.png: Amostras de imagens")
    print("  - lidar_sample.png: Visualização LiDAR")


if __name__ == "__main__":
    main()