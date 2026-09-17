#!/usr/bin/env python3
"""
Dataset com drone observador capturando outros drones VISÍVEIS
Ego observa de posição estratégica para sempre ver os outros drones
"""

import airsim
import numpy as np
import time
import json
import math
from pathlib import Path
from datetime import datetime

print("📸 DATASET: OBSERVADOR COM DRONES VISÍVEIS")
print("="*60)

# Setup
output_dir = Path("dataset_observador_visivel")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
(output_dir / "annotations").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()
print(f"✅ Conectado! Drones: {drones}")

# Separa observador dos alvos
OBSERVER = "Ego"
TARGETS = [d for d in drones if d != OBSERVER]

print(f"📷 Observador: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}\n")

# Prepara drones
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)

# IMPORTANTE: Posiciona observador ATRÁS e ACIMA para ver os outros
print("📍 Posicionando observador para visão clara...")
client.moveToPositionAsync(-30, 0, -20, 5, vehicle_name=OBSERVER).join()

# Rotaciona observador para olhar para frente (onde estarão os alvos)
client.simSetVehicleOrientation(
    airsim.Quaternionr(0, 0, 0, 1),  # Olhando para frente
    vehicle_name=OBSERVER
)
print("✅ Observador posicionado!\n")

# Cenários
scenarios = [
    {"name": "Linha Frontal", "weather": "clear", "time": 12},
    {"name": "Triângulo", "weather": "foggy", "time": 8},
    {"name": "Dança Circular", "weather": "clear", "time": 18},
    {"name": "Subida e Descida", "weather": "clear", "time": 10},
]

FRAMES_PER_SCENARIO = 25
frame_count = 0
start_time = time.time()

print(f"🎬 Coletando {len(scenarios) * FRAMES_PER_SCENARIO} frames\n")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"📍 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Configura ambiente
    fog_value = 0.3 if scenario['weather'] == 'foggy' else 0
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog_value)
    client.simSetTimeOfDay(True, f"2024-01-01 {scenario['time']:02d}:00:00")
    time.sleep(1)

    for frame in range(FRAMES_PER_SCENARIO):
        t = frame / FRAMES_PER_SCENARIO * 2 * math.pi

        # POSICIONA ALVOS SEMPRE NA FRENTE DO OBSERVADOR
        if scenario['name'] == "Linha Frontal":
            # Alvos em linha horizontal na frente
            for i, target in enumerate(TARGETS):
                x = 10 + (i - len(TARGETS)/2) * 8  # À frente
                y = (i - len(TARGETS)/2) * 10       # Espaçados lateralmente
                z = -18 + math.sin(t) * 3           # Próximo da altura do observador
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        elif scenario['name'] == "Triângulo":
            # Formação triangular na frente
            positions = [
                (15, 0, -15),    # Topo
                (15, -8, -22),   # Base esquerda
                (15, 8, -22)     # Base direita
            ]
            for i, target in enumerate(TARGETS[:3]):
                if i < len(positions):
                    x, y, z = positions[i]
                    x += math.sin(t) * 3
                    z += math.cos(t) * 2
                    client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        elif scenario['name'] == "Dança Circular":
            # Círculo vertical na frente do observador
            for i, target in enumerate(TARGETS):
                angle = (2 * math.pi * i / len(TARGETS)) + t
                x = 20  # Distância frontal fixa
                y = 10 * math.cos(angle)
                z = -18 + 10 * math.sin(angle)
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        else:  # Subida e Descida
            # Movimento vertical sincronizado
            for i, target in enumerate(TARGETS):
                x = 15 + i * 3  # Escalonados à frente
                y = (i - len(TARGETS)/2) * 8
                z = -18 + math.sin(t + i * 0.5) * 8
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

        time.sleep(0.5)

        # CAPTURA IMAGEM
        try:
            png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=OBSERVER)

            if png and len(png) > 1000:
                # Salva imagem
                img_filename = f"frame_{frame_count:05d}.png"
                img_path = output_dir / "images" / img_filename
                with open(img_path, 'wb') as f:
                    f.write(png)

                # Anotações
                annotations = {
                    "frame": frame_count,
                    "image": img_filename,
                    "scenario": scenario['name'],
                    "observer": {
                        "name": OBSERVER,
                        "x": -30, "y": 0, "z": -20
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
                        "z": pos.z_val,
                        # Calcula distância do observador
                        "distance": math.sqrt(
                            (pos.x_val + 30)**2 +
                            pos.y_val**2 +
                            (pos.z_val + 20)**2
                        )
                    }

                # Salva JSON
                anno_path = output_dir / "annotations" / f"frame_{frame_count:05d}.json"
                with open(anno_path, 'w') as f:
                    json.dump(annotations, f, indent=2)

                frame_count += 1

        except Exception as e:
            print(f"   ⚠️ Erro: {e}")

        # Progresso
        progress = (scenario_idx * FRAMES_PER_SCENARIO + frame + 1)
        total = len(scenarios) * FRAMES_PER_SCENARIO
        print(f"   📸 {progress}/{total} ({100*progress/total:.0f}%)", end='\r')

    print()

# Pousa
print("\n🛬 Pousando...")
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
print("✅ DATASET COMPLETO!")
print("="*60)
print(f"📸 Imagens: {num_images}")
print(f"🎯 Alvos observados: {', '.join(TARGETS)}")
print(f"⏱️ Tempo: {elapsed:.1f}s")
print(f"📁 Local: {output_dir.absolute()}")
print("\n💡 Os drones agora devem estar VISÍVEIS nas imagens!")
print("   Observador em (-30,0,-20) olhando para frente")
print("   Alvos sempre posicionados à frente (X > 0)")