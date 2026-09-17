#!/usr/bin/env python3
"""
CORREÇÃO FINAL: Usar câmera frontal corretamente
"""

import airsim
import time
import json
import math
from pathlib import Path

print("🎯 DATASET COM CÂMERA CORRIGIDA")
print("="*60)

# Setup
output_dir = Path("dataset_camera_corrigida")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()

OBSERVER = "Ego"
TARGETS = [d for d in drones if d != OBSERVER]

print(f"📷 Observador: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}\n")

# Prepara e decola todos
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone)

time.sleep(5)

# Posiciona observador
print("📍 Posicionando observador...")
client.moveToPositionAsync(-30, 0, -20, 5, vehicle_name=OBSERVER).join()

# CORREÇÃO: Rotaciona o DRONE inteiro para olhar para frente
print("🔄 Rotacionando observador para frente...")
# Pitch negativo = olhar para cima
client.rotateToYawAsync(0, vehicle_name=OBSERVER).join()  # Olhar para frente (norte)

print("✅ Observador configurado!\n")

# Gera dataset
FRAMES = 60
frame_count = 0

# Primeiro, tenta descobrir qual nome de câmera funciona
print("🔍 Testando câmeras disponíveis...")
camera_names = ["0", "front_center", "front", "1", "bottom_center", ""]

working_camera = None
for cam_name in camera_names:
    try:
        test_img = client.simGetImage(cam_name, airsim.ImageType.Scene, vehicle_name=OBSERVER)
        if test_img and len(test_img) > 1000:
            print(f"   ✅ Câmera '{cam_name}' funcionando!")
            working_camera = cam_name
            break
    except:
        continue

if not working_camera:
    working_camera = "0"  # Padrão
    print(f"   ⚠️ Usando câmera padrão '0'")

print(f"\n🎬 Capturando {FRAMES} frames com câmera '{working_camera}'\n")

for frame in range(FRAMES):
    t = frame / FRAMES * 2 * math.pi

    # Move alvos EM VOLTA e ACIMA do observador
    for i, target in enumerate(TARGETS):
        # Padrão circular AO REDOR do observador
        angle = (2 * math.pi * i / len(TARGETS)) + t
        radius = 25

        # Posições ao redor do observador
        x = -30 + radius * math.cos(angle)  # Centro em -30 (posição do observador)
        y = radius * math.sin(angle)
        z = -15 + 10 * math.sin(angle * 2)  # Oscila acima do observador

        client.moveToPositionAsync(x, y, z, 5, vehicle_name=target)

    # A cada 10 frames, rotaciona observador para acompanhar
    if frame % 10 == 0:
        yaw_angle = (frame / FRAMES) * 360  # Gira 360 graus durante a coleta
        client.rotateToYawAsync(yaw_angle, vehicle_name=OBSERVER)

    time.sleep(0.5)

    # Captura imagem
    try:
        # Usa a câmera que funcionou
        png = client.simGetImage(working_camera, airsim.ImageType.Scene, vehicle_name=OBSERVER)

        if png and len(png) > 1000:
            img_file = f"frame_{frame_count:04d}.png"
            with open(output_dir / "images" / img_file, 'wb') as f:
                f.write(png)

            # Metadata
            meta = {
                "frame": frame_count,
                "camera": working_camera,
                "observer_yaw": (frame / FRAMES) * 360,
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

            with open(output_dir / f"frame_{frame_count:04d}.json", 'w') as f:
                json.dump(meta, f)

            frame_count += 1

    except Exception as e:
        print(f"Erro: {e}")

    print(f"📸 {frame_count}/{FRAMES} - Rotação: {(frame/FRAMES*360):.0f}°", end='\r')

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
print(f"📷 Câmera usada: '{working_camera}'")
print(f"📁 {output_dir.absolute()}")
print("\n💡 SOLUÇÃO APLICADA:")
print("   • Observador ROTACIONA 360° para capturar todos os ângulos")
print("   • Alvos voam AO REDOR do observador")
print("   • Testada câmera disponível automaticamente")
print("="*60)