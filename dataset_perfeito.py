#!/usr/bin/env python3
"""
Dataset PERFEITO - Usa configuração completa com 5 drones e 3 câmeras
"""

import airsim
import time
import json
import math
from pathlib import Path

print("🎯 DATASET COM 5 DRONES - CONFIGURAÇÃO COMPLETA")
print("="*60)

output_dir = Path("dataset_5_drones")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()
print(f"✅ Conectado! Drones: {drones}\n")

# Setup
OBSERVER = "Ego"
TARGETS = ["Intruder1", "Intruder2", "Intruder3", "Intruder4"]

# Prepara todos
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone)

time.sleep(5)
print("✅ Todos no ar!\n")

# Posiciona observador
client.moveToPositionAsync(0, -30, -25, 5, vehicle_name=OBSERVER).join()

TOTAL_FRAMES = 100
frame_count = 0

print(f"🎬 Capturando {TOTAL_FRAMES} frames com 3 câmeras\n")

for frame in range(TOTAL_FRAMES):
    t = frame / TOTAL_FRAMES * 2 * math.pi

    # Move 4 intruders em formação dinâmica
    formations = {
        "Intruder1": (20 * math.cos(t), 20 * math.sin(t), -20),
        "Intruder2": (20 * math.cos(t + 1.57), 20 * math.sin(t + 1.57), -25),
        "Intruder3": (20 * math.cos(t + 3.14), 20 * math.sin(t + 3.14), -30),
        "Intruder4": (20 * math.cos(t - 1.57), 20 * math.sin(t - 1.57), -15)
    }

    for drone, pos in formations.items():
        client.moveToPositionAsync(pos[0], pos[1], pos[2], 5, vehicle_name=drone)

    time.sleep(0.3)

    # Captura de MÚLTIPLAS câmeras
    cameras = ["front_center", "back_center", "bottom_center"]

    for camera in cameras:
        try:
            png = client.simGetImage(camera, airsim.ImageType.Scene, vehicle_name=OBSERVER)

            if png and len(png) > 1000:
                img_file = f"frame_{frame_count:04d}_{camera}.png"
                with open(output_dir / "images" / img_file, 'wb') as f:
                    f.write(png)

        except:
            pass

    frame_count += 1
    print(f"📸 {frame_count}/{TOTAL_FRAMES} - Capturando 3 câmeras", end='\r')

# Pousa
print("\n\n🛬 Pousando 5 drones...")
for drone in drones:
    client.landAsync(vehicle_name=drone)

time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

num_images = len(list((output_dir / "images").glob("*.png")))

print("\n" + "="*60)
print("✅ DATASET COMPLETO!")
print(f"📸 Imagens: {num_images}")
print(f"🚁 5 drones utilizados")
print(f"📷 3 câmeras (front, back, bottom)")
print(f"📁 {output_dir.absolute()}")
print("="*60)