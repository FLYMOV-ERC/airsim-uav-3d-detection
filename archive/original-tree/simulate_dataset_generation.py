#!/usr/bin/env python3
"""
simulate_dataset_generation.py
Simula a geração de dataset para verificar funcionalidades
Cria dados sintéticos para testar análise de bounding boxes e variações
"""

import json
import numpy as np
import cv2
from pathlib import Path
import random
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_synthetic_scene(width=1280, height=720, scenario="desert", weather="clear", time_of_day="noon"):
    """Cria uma cena sintética com variações"""

    # Base da imagem dependendo do cenário
    if scenario == "desert":
        # Céu e areia
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Céu (parte superior)
        sky_color = {
            "noon": [135, 206, 235],      # Azul claro
            "sunset": [255, 150, 100],    # Laranja
            "night": [25, 25, 50],        # Azul escuro
            "dawn": [150, 130, 180]       # Roxo claro
        }.get(time_of_day, [135, 206, 235])

        img[:height//2, :] = sky_color

        # Areia (parte inferior)
        sand_color = {
            "noon": [238, 203, 173],
            "sunset": [200, 150, 100],
            "night": [100, 80, 60],
            "dawn": [180, 160, 140]
        }.get(time_of_day, [238, 203, 173])

        img[height//2:, :] = sand_color

    elif scenario == "city":
        # Cidade com prédios
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Céu
        sky_color = [100, 150, 200] if time_of_day == "noon" else [50, 50, 80]
        img[:height//2, :] = sky_color

        # Asfalto
        img[height//2:, :] = [80, 80, 80]

        # Adicionar alguns "prédios"
        for i in range(5):
            x = random.randint(0, width-100)
            h = random.randint(100, 300)
            cv2.rectangle(img, (x, height-h), (x+80, height), (120, 120, 130), -1)

    elif scenario == "forest":
        # Floresta
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Céu verde-azulado
        img[:height//3, :] = [100, 150, 100]
        # Vegetação
        img[height//3:, :] = [34, 139, 34]

    else:  # mountains
        # Montanhas
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:height//2, :] = [135, 206, 235]
        img[height//2:, :] = [139, 137, 137]

    # Aplicar efeitos de clima
    if weather == "fog":
        # Adicionar neblina
        fog = np.ones((height, width, 3), dtype=np.uint8) * 200
        img = cv2.addWeighted(img, 0.6, fog, 0.4, 0)
    elif weather == "rain":
        # Adicionar linhas de chuva
        for _ in range(100):
            x = random.randint(0, width)
            y = random.randint(0, height)
            cv2.line(img, (x, y), (x+5, y+20), (200, 200, 255), 1)
    elif weather == "snow":
        # Adicionar pontos de neve
        for _ in range(200):
            x = random.randint(0, width)
            y = random.randint(0, height)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)

    return img


def add_drone_to_image(img, drone_pos, drone_size=30, drone_color=(255, 0, 0), drone_id="Intruder1"):
    """Adiciona um drone à imagem e retorna a bounding box"""

    x, y = drone_pos
    h, w = img.shape[:2]

    # Garantir que o drone está dentro da imagem
    x = max(drone_size, min(x, w - drone_size))
    y = max(drone_size, min(y, h - drone_size))

    # Desenhar o drone (representação simplificada)
    # Corpo
    cv2.circle(img, (x, y), drone_size//2, drone_color, -1)

    # Braços
    cv2.line(img, (x-drone_size, y), (x+drone_size, y), drone_color, 3)
    cv2.line(img, (x, y-drone_size), (x, y+drone_size), drone_color, 3)

    # Rotores
    for dx, dy in [(-drone_size, -drone_size), (drone_size, -drone_size),
                   (-drone_size, drone_size), (drone_size, drone_size)]:
        cv2.circle(img, (x+dx, y+dy), 5, (100, 100, 100), -1)

    # Calcular bounding box
    bbox = {
        "drone_id": drone_id,
        "x": x - drone_size,
        "y": y - drone_size,
        "width": drone_size * 2,
        "height": drone_size * 2,
        "center_x": x,
        "center_y": y,
        "confidence": 0.95
    }

    return img, bbox


def create_segmentation_mask(img_shape, bboxes):
    """Cria máscara de segmentação para os drones"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    for i, bbox in enumerate(bboxes):
        color_id = (i + 1) * 50  # IDs diferentes para cada drone
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cv2.rectangle(mask, (x, y), (x+w, y+h), color_id, -1)

    return mask


def generate_synthetic_dataset(output_dir="synthetic_dataset", num_frames=50):
    """Gera dataset sintético completo com variações"""

    output_path = Path(output_dir)

    # Criar estrutura de diretórios
    for subdir in ["images/front_center", "seg/front_center", "meta", "analysis"]:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    # Configurações de variação
    scenarios = ["desert", "city", "forest", "mountains"]
    weather_conditions = ["clear", "fog", "rain", "snow"]
    times_of_day = ["dawn", "noon", "sunset", "night"]

    # Lista para armazenar metadados
    all_metadata = []
    all_bboxes = []

    print(f"🎬 Gerando {num_frames} frames sintéticos...")

    for frame_idx in range(1, num_frames + 1):
        # Variar cenário a cada 10 frames
        scenario = scenarios[(frame_idx // 10) % len(scenarios)]

        # Variar clima a cada 5 frames
        weather = weather_conditions[(frame_idx // 5) % len(weather_conditions)]

        # Variar horário a cada 7 frames
        time_of_day = times_of_day[(frame_idx // 7) % len(times_of_day)]

        # Criar cena base
        img = create_synthetic_scene(scenario=scenario, weather=weather, time_of_day=time_of_day)

        # Adicionar múltiplos drones
        bboxes = []
        drone_configs = [
            {"id": "Intruder1", "color": (255, 0, 0), "base_pos": (400, 300)},
            {"id": "Intruder2", "color": (0, 255, 0), "base_pos": (800, 400)},
            {"id": "Intruder3", "color": (0, 0, 255), "base_pos": (600, 200)},
            {"id": "Intruder4", "color": (255, 255, 0), "base_pos": (1000, 350)}
        ]

        for drone_cfg in drone_configs:
            # Adicionar movimento variado
            offset_x = int(100 * np.sin(frame_idx * 0.1 + hash(drone_cfg["id"]) % 10))
            offset_y = int(50 * np.cos(frame_idx * 0.15 + hash(drone_cfg["id"]) % 10))

            drone_pos = (
                drone_cfg["base_pos"][0] + offset_x,
                drone_cfg["base_pos"][1] + offset_y
            )

            # Adicionar drone à imagem
            img, bbox = add_drone_to_image(
                img, drone_pos,
                drone_size=25 + (frame_idx % 3) * 5,  # Variar tamanho
                drone_color=drone_cfg["color"],
                drone_id=drone_cfg["id"]
            )

            bbox["frame"] = frame_idx
            bbox["scenario"] = scenario
            bbox["weather"] = weather
            bbox["time_of_day"] = time_of_day
            bboxes.append(bbox)
            all_bboxes.append(bbox)

        # Criar máscara de segmentação
        seg_mask = create_segmentation_mask(img.shape, bboxes)

        # Salvar imagem
        img_path = output_path / f"images/front_center/{frame_idx:06d}.png"
        cv2.imwrite(str(img_path), img)

        # Salvar segmentação
        seg_path = output_path / f"seg/front_center/{frame_idx:06d}.png"
        cv2.imwrite(str(seg_path), seg_mask)

        # Criar metadados
        metadata = {
            "frame": frame_idx,
            "timestamp": time.time(),
            "scenario": scenario,
            "weather": weather,
            "time_of_day": time_of_day,
            "image_size": {"width": 1280, "height": 720},
            "vehicles": {
                "Ego": {
                    "pose": {
                        "position": {"x": 0, "y": 0, "z": -10},
                        "orientation": [0, 0, 0, 1]
                    }
                }
            },
            "bounding_boxes": bboxes,
            "num_drones_visible": len(bboxes)
        }

        # Adicionar posições dos drones aos metadados
        for bbox in bboxes:
            metadata["vehicles"][bbox["drone_id"]] = {
                "pose": {
                    "position": {
                        "x": bbox["center_x"] / 10,  # Conversão para metros
                        "y": bbox["center_y"] / 10,
                        "z": -10 + random.uniform(-5, 5)
                    },
                    "orientation": [0, 0, 0, 1]
                }
            }

        # Salvar metadados
        meta_path = output_path / f"meta/{frame_idx:06d}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        all_metadata.append(metadata)

        if frame_idx % 10 == 0:
            print(f"  Processado: {frame_idx}/{num_frames} frames")

    print(f"✅ Dataset sintético gerado em: {output_path}")

    # Criar arquivo de resumo
    summary = {
        "total_frames": num_frames,
        "scenarios_used": list(set(m["scenario"] for m in all_metadata)),
        "weather_conditions_used": list(set(m["weather"] for m in all_metadata)),
        "times_of_day_used": list(set(m["time_of_day"] for m in all_metadata)),
        "total_bboxes": len(all_bboxes),
        "avg_drones_per_frame": len(all_bboxes) / num_frames,
        "output_path": str(output_path)
    }

    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return output_path, summary


def analyze_synthetic_dataset(dataset_path):
    """Analisa o dataset sintético gerado"""

    dataset_path = Path(dataset_path)
    analysis_dir = dataset_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("🔍 ANÁLISE DO DATASET SINTÉTICO")
    print("="*60)

    # Carregar resumo
    with open(dataset_path / "summary.json", 'r') as f:
        summary = json.load(f)

    print(f"\n📊 Estatísticas Gerais:")
    print(f"  Total de frames: {summary['total_frames']}")
    print(f"  Cenários: {', '.join(summary['scenarios_used'])}")
    print(f"  Condições climáticas: {', '.join(summary['weather_conditions_used'])}")
    print(f"  Horários: {', '.join(summary['times_of_day_used'])}")
    print(f"  Total de bounding boxes: {summary['total_bboxes']}")
    print(f"  Média de drones por frame: {summary['avg_drones_per_frame']:.1f}")

    # Analisar distribuição de cenários
    meta_files = sorted((dataset_path / "meta").glob("*.json"))

    scenario_count = {}
    weather_count = {}
    time_count = {}
    drones_per_frame = []

    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        scenario_count[meta['scenario']] = scenario_count.get(meta['scenario'], 0) + 1
        weather_count[meta['weather']] = weather_count.get(meta['weather'], 0) + 1
        time_count[meta['time_of_day']] = time_count.get(meta['time_of_day'], 0) + 1
        drones_per_frame.append(meta['num_drones_visible'])

    # Criar visualizações
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Análise do Dataset Sintético Multi-Drone', fontsize=16)

    # 1. Distribuição de cenários
    ax = axes[0, 0]
    ax.bar(scenario_count.keys(), scenario_count.values(), color='skyblue')
    ax.set_title('Distribuição de Cenários')
    ax.set_xlabel('Cenário')
    ax.set_ylabel('Número de Frames')
    ax.tick_params(axis='x', rotation=45)

    # 2. Distribuição de clima
    ax = axes[0, 1]
    ax.bar(weather_count.keys(), weather_count.values(), color='lightcoral')
    ax.set_title('Distribuição de Condições Climáticas')
    ax.set_xlabel('Clima')
    ax.set_ylabel('Número de Frames')
    ax.tick_params(axis='x', rotation=45)

    # 3. Distribuição de horário
    ax = axes[0, 2]
    ax.bar(time_count.keys(), time_count.values(), color='lightgreen')
    ax.set_title('Distribuição de Horários')
    ax.set_xlabel('Horário')
    ax.set_ylabel('Número de Frames')
    ax.tick_params(axis='x', rotation=45)

    # 4. Amostra de imagem com bboxes
    ax = axes[1, 0]
    sample_img_path = dataset_path / "images/front_center/000025.png"
    if sample_img_path.exists():
        img = cv2.imread(str(sample_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        # Carregar e desenhar bboxes
        with open(dataset_path / "meta/000025.json", 'r') as f:
            meta = json.load(f)

        for bbox in meta.get('bounding_boxes', []):
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']),
                bbox['width'], bbox['height'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(bbox['x'], bbox['y']-5, bbox['drone_id'],
                   color='yellow', fontsize=8, weight='bold')

        ax.set_title(f"Frame 25: {meta['scenario']} - {meta['weather']} - {meta['time_of_day']}")
    ax.axis('off')

    # 5. Outra amostra
    ax = axes[1, 1]
    sample_img_path = dataset_path / "images/front_center/000040.png"
    if sample_img_path.exists():
        img = cv2.imread(str(sample_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        with open(dataset_path / "meta/000040.json", 'r') as f:
            meta = json.load(f)

        for bbox in meta.get('bounding_boxes', []):
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']),
                bbox['width'], bbox['height'],
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

        ax.set_title(f"Frame 40: {meta['scenario']} - {meta['weather']} - {meta['time_of_day']}")
    ax.axis('off')

    # 6. Histograma de drones por frame
    ax = axes[1, 2]
    ax.hist(drones_per_frame, bins=range(0, 6), color='purple', alpha=0.7)
    ax.set_title('Drones Visíveis por Frame')
    ax.set_xlabel('Número de Drones')
    ax.set_ylabel('Frequência')
    ax.set_xticks(range(0, 5))

    plt.tight_layout()
    analysis_path = analysis_dir / 'dataset_analysis.png'
    plt.savefig(analysis_path, dpi=100, bbox_inches='tight')
    print(f"\n✅ Análise salva em: {analysis_path}")
    plt.close()

    return True


def create_bbox_visualization(dataset_path, num_samples=6):
    """Cria visualização detalhada de bounding boxes"""

    dataset_path = Path(dataset_path)
    analysis_dir = dataset_path / "analysis"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Verificação de Bounding Boxes em Diferentes Cenários', fontsize=16)

    # Selecionar frames diversos
    sample_frames = [5, 15, 25, 35, 45, 49]

    for idx, frame_num in enumerate(sample_frames):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        img_path = dataset_path / f"images/front_center/{frame_num:06d}.png"
        meta_path = dataset_path / f"meta/{frame_num:06d}.json"

        if img_path.exists() and meta_path.exists():
            # Carregar imagem
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Carregar metadados
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Desenhar bounding boxes diretamente na imagem
            for bbox in meta.get('bounding_boxes', []):
                color = {
                    "Intruder1": (255, 0, 0),
                    "Intruder2": (0, 255, 0),
                    "Intruder3": (0, 0, 255),
                    "Intruder4": (255, 255, 0)
                }.get(bbox['drone_id'], (255, 255, 255))

                cv2.rectangle(img,
                            (bbox['x'], bbox['y']),
                            (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                            color, 2)

                # Label
                cv2.putText(img, bbox['drone_id'],
                          (bbox['x'], bbox['y']-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ax.imshow(img)
            title = f"Frame {frame_num}\n{meta['scenario']} | {meta['weather']} | {meta['time_of_day']}"
            title += f"\n{meta['num_drones_visible']} drones"
            ax.set_title(title, fontsize=9)

        ax.axis('off')

    plt.tight_layout()
    bbox_path = analysis_dir / 'bbox_verification.png'
    plt.savefig(bbox_path, dpi=100, bbox_inches='tight')
    print(f"✅ Verificação de bboxes salva em: {bbox_path}")
    plt.close()


def main():
    """Função principal"""
    print("\n" + "="*70)
    print(" "*15 + "🚁 GERAÇÃO E ANÁLISE DE DATASET MULTI-DRONE")
    print("="*70)

    # 1. Gerar dataset sintético
    print("\n📦 Etapa 1: Gerando dataset sintético...")
    dataset_path, summary = generate_synthetic_dataset(
        output_dir="synthetic_dataset",
        num_frames=50
    )

    # 2. Analisar dataset
    print("\n📊 Etapa 2: Analisando dataset gerado...")
    analyze_synthetic_dataset(dataset_path)

    # 3. Verificar bounding boxes
    print("\n🎯 Etapa 3: Verificando bounding boxes...")
    create_bbox_visualization(dataset_path)

    print("\n" + "="*70)
    print(" "*20 + "✅ ANÁLISE COMPLETA!")
    print("="*70)

    print("\n📋 RESUMO DA VERIFICAÇÃO:")
    print("✅ Múltiplos drones: 4 drones em cada frame")
    print("✅ Cenários variados: desert, city, forest, mountains")
    print("✅ Condições climáticas: clear, fog, rain, snow")
    print("✅ Horários do dia: dawn, noon, sunset, night")
    print("✅ Bounding boxes: Geradas para cada drone")
    print("✅ Segmentação: Máscaras criadas")

    print(f"\n📁 Arquivos gerados em: {dataset_path}/")
    print("  - images/: Imagens RGB com drones")
    print("  - seg/: Máscaras de segmentação")
    print("  - meta/: Metadados com bboxes")
    print("  - analysis/: Visualizações")


if __name__ == "__main__":
    main()