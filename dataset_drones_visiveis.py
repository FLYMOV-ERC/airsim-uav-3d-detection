#!/usr/bin/env python3
"""
Dataset com drones REALMENTE visíveis
Solução: Colocar alvos na mesma altura ou ACIMA do observador
"""

import airsim
import time
import json
import math
from pathlib import Path

print("🎯 DATASET COM DRONES VISÍVEIS")
print("="*60)

# Setup
output_dir = Path("dataset_drones_visiveis")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()

OBSERVER = "Ego"
TARGETS = [d for d in drones if d != OBSERVER]

print(f"📷 Observador: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}")
print("="*60)

# Prepara e decola
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone)

time.sleep(5)

# IMPORTANTE: Posiciona observador BAIXO e LONGE
print("\n📍 Posicionando observador...")
OBSERVER_POS = (-50, 0, -30)  # Bem atrás e baixo
client.moveToPositionAsync(
    OBSERVER_POS[0], OBSERVER_POS[1], OBSERVER_POS[2],
    5, vehicle_name=OBSERVER
).join()
print(f"   Observador em: {OBSERVER_POS}")

# Posiciona alvos inicialmente À FRENTE e ACIMA
print("📍 Posicionando alvos à frente e acima...")
for i, target in enumerate(TARGETS):
    x = 0  # À frente
    y = (i - len(TARGETS)/2) * 15  # Espaçados lateralmente
    z = -10  # ACIMA do observador
    client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

time.sleep(3)
print("✅ Todos posicionados!\n")

# Gera dataset
TOTAL_FRAMES = 80
frame_count = 0

print(f"🎬 Capturando {TOTAL_FRAMES} frames\n")

for frame in range(TOTAL_FRAMES):
    t = frame / TOTAL_FRAMES * 2 * math.pi

    # Move alvos em padrões SEMPRE À FRENTE E VISÍVEIS
    pattern = frame // 20  # Muda padrão a cada 20 frames

    if pattern == 0:
        # PADRÃO 1: Linha horizontal
        for i, target in enumerate(TARGETS):
            x = 20  # Sempre à frente
            y = (i - len(TARGETS)/2) * 20 + math.sin(t) * 10
            z = -15  # Acima do observador
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

    elif pattern == 1:
        # PADRÃO 2: Círculo vertical
        for i, target in enumerate(TARGETS):
            angle = (2 * math.pi * i / len(TARGETS)) + t
            x = 30  # À frente
            y = 20 * math.cos(angle)
            z = -20 + 15 * math.sin(angle)  # Oscila acima e abaixo
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

    elif pattern == 2:
        # PADRÃO 3: Diagonal ascendente
        for i, target in enumerate(TARGETS):
            x = 15 + i * 10  # Escalonados à frente
            y = (i - len(TARGETS)/2) * 15
            z = -25 + i * 5 + math.sin(t) * 3  # Subindo
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

    else:
        # PADRÃO 4: Formação V
        for i, target in enumerate(TARGETS):
            if i == 0:
                x, y, z = 25, 0, -15  # Líder
            else:
                side = 1 if i % 2 else -1
                x = 25 - i * 5
                y = side * i * 10
                z = -15 - i * 2
            x += math.sin(t) * 5
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

    time.sleep(0.3)

    # Captura imagem
    try:
        png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=OBSERVER)

        if png and len(png) > 1000:
            # Salva imagem
            img_file = f"frame_{frame_count:04d}.png"
            with open(output_dir / "images" / img_file, 'wb') as f:
                f.write(png)

            # Metadata
            meta = {
                "frame": frame_count,
                "pattern": ["linha", "circulo", "diagonal", "formacao_v"][pattern],
                "observer": OBSERVER_POS,
                "targets": {}
            }

            for target in TARGETS:
                state = client.getMultirotorState(vehicle_name=target)
                pos = state.kinematics_estimated.position
                meta["targets"][target] = {
                    "x": pos.x_val,
                    "y": pos.y_val,
                    "z": pos.z_val,
                    "relative_x": pos.x_val - OBSERVER_POS[0],
                    "relative_y": pos.y_val - OBSERVER_POS[1],
                    "relative_z": pos.z_val - OBSERVER_POS[2]
                }

            with open(output_dir / "metadata" / f"frame_{frame_count:04d}.json", 'w') as f:
                json.dump(meta, f, indent=2)

            frame_count += 1

    except Exception as e:
        print(f"Erro: {e}")

    # Progresso
    print(f"📸 Frame {frame_count}/{TOTAL_FRAMES} - Padrão: {['linha', 'círculo', 'diagonal', 'V'][pattern]}", end='\r')

# Pousa
print("\n\n🛬 Pousando todos os drones...")
for drone in drones:
    client.landAsync(vehicle_name=drone)

time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

# Resumo
print("\n" + "="*60)
print("✅ DATASET COMPLETO!")
print("="*60)
print(f"📸 Imagens capturadas: {frame_count}")
print(f"📁 Local: {output_dir.absolute()}")
print("\n💡 Configuração usada:")
print(f"   • Observador: {OBSERVER_POS} (atrás e baixo)")
print(f"   • Alvos: Sempre à frente (X > 0) e visíveis")
print(f"   • 4 padrões de movimento diferentes")
print("="*60)