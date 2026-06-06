#!/usr/bin/env python3
"""
Dataset FINAL - Observador captura drones visíveis
"""

import airsim
import time
import json
import math
from pathlib import Path

print("📸 GERANDO DATASET - OBSERVADOR")
print("="*60)

# Setup
output_dir = Path("dataset_final")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()

OBSERVER = "Ego"
TARGETS = [d for d in drones if d != OBSERVER]

print(f"📷 Câmera: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}\n")

# Prepara
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)

# Posiciona observador
print("📍 Posicionando observador...")
client.moveToPositionAsync(-40, 0, -25, 5, vehicle_name=OBSERVER).join()
print("✅ Pronto!\n")

frame_count = 0
TOTAL_FRAMES = 100

print(f"🎬 Capturando {TOTAL_FRAMES} frames\n")

for frame in range(TOTAL_FRAMES):
    t = frame / TOTAL_FRAMES * 2 * math.pi

    # Move alvos em padrões VISÍVEIS
    for i, target in enumerate(TARGETS):
        # Padrão circular à frente do observador
        angle = (2 * math.pi * i / len(TARGETS)) + t

        # Posições À FRENTE do observador (X positivo)
        x = 15 + 10 * math.cos(angle)  # Entre 5 e 25 metros à frente
        y = 15 * math.sin(angle)        # Movimento lateral
        z = -25 + 8 * math.sin(angle)   # Variação de altura

        client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

    time.sleep(0.3)  # Aguarda movimento

    # Captura imagem do observador
    try:
        png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=OBSERVER)

        if png and len(png) > 1000:
            # Salva imagem
            img_file = f"frame_{frame_count:05d}.png"
            with open(output_dir / "images" / img_file, 'wb') as f:
                f.write(png)

            # Metadata
            meta = {
                "frame": frame_count,
                "observer_pos": {"x": -40, "y": 0, "z": -25},
                "targets": {}
            }

            for target in TARGETS:
                state = client.getMultirotorState(vehicle_name=target)
                pos = state.kinematics_estimated.position
                meta["targets"][target] = {
                    "x": pos.x_val,
                    "y": pos.y_val,
                    "z": pos.z_val
                }

            # Salva metadata
            with open(output_dir / "metadata" / f"frame_{frame_count:05d}.json", 'w') as f:
                json.dump(meta, f)

            frame_count += 1

    except Exception as e:
        print(f"Erro: {e}")

    # Progresso
    print(f"📸 {frame_count}/{TOTAL_FRAMES} ({100*frame_count/TOTAL_FRAMES:.0f}%)", end='\r')

# Pousa
print("\n\n🛬 Pousando...")
for drone in drones:
    client.landAsync(vehicle_name=drone)
time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

print("\n" + "="*60)
print(f"✅ COMPLETO! {frame_count} frames")
print(f"📁 {output_dir.absolute()}")
print("="*60)