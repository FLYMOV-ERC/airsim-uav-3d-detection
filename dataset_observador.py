#!/usr/bin/env python3
"""
Dataset com drone observador capturando imagens de outros drones
Ego = Observador (câmera)
Outros = Alvos em movimento
"""

import airsim
import numpy as np
import time
import json
import math
from pathlib import Path
from datetime import datetime

print("📸 DATASET: DRONE OBSERVADOR")
print("="*60)
print("Configuração:")
print("  • Ego: Drone observador com câmera")
print("  • Outros: Drones alvos em movimento")
print("="*60)

# Setup
output_dir = Path("dataset_observador")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
(output_dir / "annotations").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()
print(f"\n✅ Conectado! Drones: {drones}")

# Separa observador dos alvos
OBSERVER = "Ego"  # Drone com câmera
TARGETS = [d for d in drones if d != OBSERVER]  # Drones a serem observados

print(f"📷 Observador: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}\n")

# Prepara todos os drones
print("🚁 Preparando drones...")
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola todos
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)

# Posiciona o observador
print("📍 Posicionando observador...")
client.moveToPositionAsync(0, 0, -30, 5, vehicle_name=OBSERVER).join()
print("✅ Observador em posição!\n")

# Cenários
scenarios = [
    {"name": "Formação Linha", "weather": "clear", "time": 12},
    {"name": "Círculo", "weather": "foggy", "time": 8},
    {"name": "Triângulo", "weather": "clear", "time": 18},
    {"name": "Aleatório", "weather": "clear", "time": 22},
]

FRAMES_PER_SCENARIO = 30
frame_count = 0
start_time = time.time()

print(f"🎬 Iniciando coleta: {len(scenarios)} cenários × {FRAMES_PER_SCENARIO} frames\n")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"📍 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Configura ambiente
    if scenario['weather'] == 'foggy':
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.5)
    else:
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)

    client.simSetTimeOfDay(True, f"2024-01-01 {scenario['time']:02d}:00:00")
    time.sleep(1)

    for frame in range(FRAMES_PER_SCENARIO):
        # Move alvos em diferentes padrões
        t = frame / FRAMES_PER_SCENARIO * 2 * math.pi

        if scenario['name'] == "Formação Linha":
            # Alvos em linha horizontal
            for i, target in enumerate(TARGETS):
                x = (i - len(TARGETS)/2) * 15
                y = 30 + math.sin(t) * 10
                z = -25 + math.cos(t) * 5
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        elif scenario['name'] == "Círculo":
            # Alvos em círculo
            for i, target in enumerate(TARGETS):
                angle = (2 * math.pi * i / len(TARGETS)) + t
                radius = 25
                x = radius * math.cos(angle)
                y = 30 + radius * math.sin(angle)
                z = -30 + math.sin(t) * 5
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        elif scenario['name'] == "Triângulo":
            # Alvos em triângulo
            positions = [
                (0, 40, -25),
                (-20, 20, -30),
                (20, 20, -30)
            ]
            for i, target in enumerate(TARGETS[:3]):
                if i < len(positions):
                    x, y, z = positions[i]
                    x += math.sin(t) * 5
                    y += math.cos(t) * 5
                    client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        else:  # Aleatório
            # Movimento aleatório suave
            for i, target in enumerate(TARGETS):
                x = 20 * math.sin(t + i * 1.5)
                y = 30 + 20 * math.cos(t * 1.2 + i)
                z = -25 + 10 * math.sin(t * 0.8 + i * 2)
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        time.sleep(0.5)  # Aguarda movimento

        # CAPTURA IMAGEM DO OBSERVADOR
        try:
            # Captura RGB
            png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=OBSERVER)

            if png and len(png) > 1000:
                # Salva imagem
                img_filename = f"frame_{frame_count:05d}.png"
                img_path = output_dir / "images" / img_filename
                with open(img_path, 'wb') as f:
                    f.write(png)

                # Coleta posições dos alvos (para anotações)
                annotations = {
                    "frame": frame_count,
                    "image": img_filename,
                    "scenario": scenario['name'],
                    "observer_position": {
                        "x": 0, "y": 0, "z": -30
                    },
                    "targets": {}
                }

                # Posições dos alvos
                for target in TARGETS:
                    state = client.getMultirotorState(vehicle_name=target)
                    pos = state.kinematics_estimated.position
                    annotations["targets"][target] = {
                        "x": pos.x_val,
                        "y": pos.y_val,
                        "z": pos.z_val
                    }

                # Salva anotações
                anno_filename = f"frame_{frame_count:05d}.json"
                anno_path = output_dir / "annotations" / anno_filename
                with open(anno_path, 'w') as f:
                    json.dump(annotations, f, indent=2)

                frame_count += 1

        except Exception as e:
            print(f"   ⚠️ Erro na captura: {e}")

        # Progresso
        total_progress = scenario_idx * FRAMES_PER_SCENARIO + frame + 1
        total_frames = len(scenarios) * FRAMES_PER_SCENARIO
        pct = (total_progress / total_frames) * 100
        print(f"   📸 {total_progress}/{total_frames} ({pct:.0f}%)", end='\r')

    print()  # Nova linha após cenário

# Pousa todos
print("\n🛬 Pousando drones...")
for drone in drones:
    client.landAsync(vehicle_name=drone)
time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

# Estatísticas
elapsed = time.time() - start_time
num_images = len(list((output_dir / "images").glob("*.png")))

print("\n" + "="*60)
print("✅ DATASET DO OBSERVADOR COMPLETO!")
print("="*60)
print(f"📸 Imagens capturadas: {num_images}")
print(f"📊 Anotações salvas: {num_images}")
print(f"🎯 Drones observados: {', '.join(TARGETS)}")
print(f"⏱️ Tempo total: {elapsed:.1f} segundos")
print(f"📁 Local: {output_dir.absolute()}")
print("="*60)

# Resumo
summary = {
    "observer": OBSERVER,
    "targets": TARGETS,
    "total_frames": num_images,
    "scenarios": scenarios,
    "generation_time": elapsed,
    "timestamp": str(datetime.now())
}

with open(output_dir / "dataset_info.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n💡 Dica: As imagens mostram os drones alvos do ponto de vista do observador!")