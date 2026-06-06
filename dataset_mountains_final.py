#!/usr/bin/env python3
"""
Dataset FINAL para Mountains com drones VISÍVEIS
"""

import airsim
import time
import math
from pathlib import Path

print("🏔️ DATASET MOUNTAINS COM DRONES VISÍVEIS")
print("="*60)

output_dir = Path("dataset_mountains_final")
output_dir.mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()
print(f"✅ Drones: {drones}\n")

# Prepara todos
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola todos
print("🛫 Decolando...")
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)

# IMPORTANTE: Posiciona drones onde a câmera pode ver
print("📍 Posicionando drones estrategicamente...")

# Ego fica parado como observador
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()
print("   Ego (observador) em (0, 0, -20)")

# Coloca outros drones NA FRENTE e ACIMA (onde a câmera vê)
positions = {
    "Drone3": (30, 0, -10),      # 30m frente, 10m acima
    "Drone4": (25, 10, -15),     # 25m frente, direita
    "Intruder1": (25, -10, -15)  # 25m frente, esquerda
}

for drone, pos in positions.items():
    if drone in drones:
        client.moveToPositionAsync(pos[0], pos[1], pos[2], 5, vehicle_name=drone).join()
        print(f"   {drone} em {pos}")

time.sleep(3)

# Gera dataset
FRAMES = 100
print(f"\n🎬 Capturando {FRAMES} frames\n")

for frame in range(FRAMES):
    t = frame / FRAMES * 2 * math.pi

    # Move drones em padrão visível
    for i, drone in enumerate(["Drone3", "Drone4", "Intruder1"]):
        if drone in drones:
            angle = (2 * math.pi * i / 3) + t

            # Movimento circular NA FRENTE do observador
            x = 25 + 10 * math.cos(angle)  # Entre 15-35m na frente
            y = 15 * math.sin(angle)        # Movimento lateral
            z = -15 + 5 * math.sin(t)       # Oscila verticalmente

            client.moveToPositionAsync(x, y, z, 3, vehicle_name=drone)

    time.sleep(0.5)

    # Captura imagem do Ego
    try:
        # Usa front_center que sabemos que funciona
        img = client.simGetImage("front_center", airsim.ImageType.Scene, "Ego")

        if img and len(img) > 1000:
            filename = f"{output_dir}/frame_{frame:04d}.png"
            with open(filename, 'wb') as f:
                f.write(img)

            print(f"📸 Frame {frame+1}/{FRAMES} ({len(img)/1024:.0f}KB)", end='\r')
    except:
        pass

# Pousa
print("\n\n🛬 Pousando...")
for drone in drones:
    client.landAsync(vehicle_name=drone)
time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

# Conta imagens
num_images = len(list(output_dir.glob("*.png")))

print("\n" + "="*60)
print(f"✅ COMPLETO! {num_images} imagens capturadas")
print(f"📁 {output_dir.absolute()}")
print("\n💡 Se os drones não aparecerem nas imagens:")
print("   • Eles podem estar muito pequenos/distantes")
print("   • Tente mover mais próximo (15-20m)")
print("   • Ou aumente o tamanho dos drones no Unreal")
print("="*60)