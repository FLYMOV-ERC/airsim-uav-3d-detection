#!/usr/bin/env python3
"""
SOLUÇÃO: Usar Intruder1 como observador (talvez tenha câmera melhor)
Os outros 3 drones serão os alvos
"""

import airsim
import time
import json
import math
from pathlib import Path

print("📸 DATASET: INTRUDER1 COMO OBSERVADOR")
print("="*60)

# Setup
output_dir = Path("dataset_intruder_observa")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()

# MUDANÇA: Intruder1 é o observador
OBSERVER = "Intruder1"
TARGETS = ["Ego", "Drone3", "Drone4"]

print(f"📷 Observador: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}")
print("="*60)

# Prepara todos
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone)

time.sleep(5)

# Posiciona observador em posição estratégica
print("\n📍 Posicionando Intruder1 como observador...")
client.moveToPositionAsync(0, -40, -20, 5, vehicle_name=OBSERVER).join()
print("   Observador posicionado em (0, -40, -20)")

# Gera dataset
FRAMES = 60
frame_count = 0

print(f"\n🎬 Capturando {FRAMES} frames\n")

for frame in range(FRAMES):
    t = frame / FRAMES * 2 * math.pi

    # Move os 3 alvos em formação triangular
    positions = [
        (20 + 10*math.sin(t), 0, -15 + 5*math.cos(t)),      # Topo
        (20 + 10*math.sin(t), -15, -25 + 5*math.cos(t)),    # Base esquerda
        (20 + 10*math.sin(t), 15, -25 + 5*math.cos(t))      # Base direita
    ]

    for target, pos in zip(TARGETS, positions):
        client.moveToPositionAsync(pos[0], pos[1], pos[2], 5, vehicle_name=target)

    time.sleep(0.5)

    # Captura do INTRUDER1
    try:
        png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=OBSERVER)

        if png and len(png) > 1000:
            img_file = f"frame_{frame_count:04d}.png"
            with open(output_dir / "images" / img_file, 'wb') as f:
                f.write(png)

            frame_count += 1

    except:
        pass

    print(f"📸 {frame_count}/{FRAMES}", end='\r')

# Pousa
print("\n\n🛬 Pousando...")
for drone in drones:
    client.landAsync(vehicle_name=drone)
time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

print(f"\n✅ {frame_count} frames salvos!")
print(f"📁 {output_dir.absolute()}")