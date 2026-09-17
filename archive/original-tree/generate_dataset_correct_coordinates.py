#!/usr/bin/env python3
"""
Gerador de Dataset - VERSÃO CORRIGIDA COM COORDENADAS
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR COM CORREÇÃO DE COORDENADAS GLOBAIS→LOCAIS!")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\n🛫 Decolando...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass

time.sleep(3)

# Diretórios
output_dir = Path("dataset_correct")
output_dir.mkdir(exist_ok=True)
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset em: {output_dir.absolute()}")

# Parâmetros da câmera
FOV_H = 90
FOV_V = 60
fx = 640 / np.tan(np.radians(FOV_H/2))
fy = 360 / np.tan(np.radians(FOV_V/2))
cx = 640
cy = 360

# Pitch da câmera
camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

# Posiciona Ego em altura conhecida
DRONE_HEIGHT = 15  # metros
print(f"\n📏 Posicionando drone a {DRONE_HEIGHT}m de altura...")
client.moveToPositionAsync(0, 0, -DRONE_HEIGHT, 5, vehicle_name="Ego").join()

# Posiciona outros drones
drone_positions = [
    ("Drone3", 10, -5, -DRONE_HEIGHT + 2),  # 2m acima
    ("Drone4", 15, 5, -DRONE_HEIGHT),       # mesma altura
    ("Intruder1", 20, 0, -DRONE_HEIGHT - 3) # 3m abaixo
]

for name, x, y, z in drone_positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"   {name}: x={x}, y={y}, altura={-z}m")

time.sleep(2)

print("\n🎬 Capturando frames de teste...")

for frame_idx in range(5):
    frame_name = f"frame_{frame_idx:04d}"
    print(f"\nFrame {frame_idx}:")

    # Limpa buffer LiDAR
    _ = client.getLidarData("LidarFront", vehicle_name="Ego")
    time.sleep(0.1)

    # Captura dados
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")
    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    # Pega posição atual do drone
    pose = client.simGetVehiclePose(vehicle_name="Ego")
    drone_z = pose.position.z_val  # Z global do drone

    print(f"   Posição Z do drone (global): {drone_z:.2f}m")
    print(f"   Altura do drone: {-drone_z:.2f}m")

    # Processa imagem
    img_bgr = None
    if responses and responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
        points_global = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # ⚠️ CORREÇÃO CRÍTICA: Converter coordenadas globais para locais
        points_local = points_global.copy()
        points_local[:, 2] = points_global[:, 2] - (-drone_z)  # Subtrai altura do drone

        print(f"   Pontos LiDAR: {len(points_local)}")
        print(f"   Z global: {points_global[:, 2].min():.2f} a {points_global[:, 2].max():.2f}")
        print(f"   Z local:  {points_local[:, 2].min():.2f} a {points_local[:, 2].max():.2f}")

        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Trabalha com coordenadas LOCAIS
        X = points_local[:, 0]
        Y = points_local[:, 1]
        Z = points_local[:, 2]

        # Filtra pontos frontais
        front_mask = X > 0.5
        X_front = X[front_mask]
        Y_front = Y[front_mask]
        Z_front = Z[front_mask]

        if len(X_front) > 0:
            # Aplica rotação do pitch
            X_rot = X_front * np.cos(pitch_rad) - Z_front * np.sin(pitch_rad)
            Z_rot = X_front * np.sin(pitch_rad) + Z_front * np.cos(pitch_rad)

            # Projeta para imagem
            u = (fx * Y_front / X_rot + cx).astype(int)
            v = (fy * Z_rot / X_rot + cy).astype(int)

            # Filtra pontos válidos
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (X_rot > 0)

            u_valid = u[valid]
            v_valid = v[valid]
            z_valid = Z_front[valid]
            x_valid = X_front[valid]

            # Desenha pontos com cores corretas
            for i in range(len(u_valid)):
                z = z_valid[i]
                dist = np.sqrt(x_valid[i]**2 + Y_front[valid][i]**2 + z**2)

                if dist > 80:  # Ignora pontos muito distantes
                    continue

                if z < -10:  # Chão (bem abaixo do drone)
                    color = (0, 255, 0)  # Verde
                    size = 1
                elif -2 < z < 2:  # Nível do drone
                    color = (255, 0, 255)  # Magenta - possível drone
                    size = 3
                elif z > 5:  # Acima do drone
                    color = (0, 0, 255)  # Vermelho
                    size = 2
                else:  # Intermediário
                    intensity = int(255 * (1 - min(1, dist/50)))
                    color = (0, intensity, intensity)  # Ciano
                    size = 1

                cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

        # Informações
        cv2.putText(fusion, f"Frame {frame_idx} | Altura: {-drone_z:.1f}m",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(fusion, "Verde=Chao, Magenta=Drone, Vermelho=Acima",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Salva imagens
        cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
        cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

        comparison = np.hstack([img_bgr, fusion])
        cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

        # Metadados
        metadata = {
            "frame_id": frame_idx,
            "drone_z_global": float(drone_z),
            "drone_height": float(-drone_z),
            "z_range_global": [float(points_global[:, 2].min()), float(points_global[:, 2].max())],
            "z_range_local": [float(points_local[:, 2].min()), float(points_local[:, 2].max())],
            "total_points": len(points_local),
            "coordinate_system": "LOCAL (convertido de global)"
        }

        with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    time.sleep(0.5)

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("✅ SUCESSO! Coordenadas corrigidas!")
print("="*70)
print("\n📌 CORREÇÃO APLICADA:")
print("   • Converte coordenadas GLOBAIS → LOCAIS")
print("   • Z_local = Z_global - altura_do_drone")
print("   • Agora o chão aparece como verde (Z < -10)")
print("   • Drones aparecem como magenta (Z ≈ 0)")
print(f"\n📁 Dataset em: {output_dir.absolute()}")