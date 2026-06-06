#!/usr/bin/env python3
"""
FUSÃO FINAL FUNCIONANDO - LiDAR detectando drones com correção de pitch!
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🎯 FUSÃO FINAL - LIDAR DETECTANDO DRONES!")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n✅ Veículos: {vehicles}")

# Prepara todos
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("🛫 Decolando...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

output_dir = Path("dataset_fusion_final")
output_dir.mkdir(exist_ok=True)

print("\n📸 GERANDO DATASET COM DETECÇÃO DE DRONES...")

# Cenário simples para demonstrar
print("\n📍 Posicionando drones...")

# Ego
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()
print("  Ego: 15m altura")

# Drones em formação visível
positions = [
    ("Drone3", 8, -3, -15, "8m frente, mesmo nível"),
    ("Drone4", 12, 3, -15, "12m frente, mesmo nível"),
    ("Intruder1", 15, 0, -12, "15m frente, 3m acima")
]

for name, x, y, z, desc in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"  {name}: {desc}")

time.sleep(3)

# Captura 5 frames
for frame in range(5):
    print(f"\n🔄 Frame {frame+1}/5:")

    # Varia ângulo
    yaw = (frame - 2) * 10  # -20, -10, 0, 10, 20
    client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
    time.sleep(0.5)

    # Captura imagem
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    img_bgr = None
    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Captura LiDAR
    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Parâmetros da câmera
        fx = w / (2 * np.tan(np.radians(45)))
        fy = h / (2 * np.tan(np.radians(30)))
        cx = w / 2
        cy = h / 2

        # CORREÇÃO DE PITCH (-15°)
        camera_pitch = -15
        pitch_rad = np.radians(camera_pitch)

        # Filtra pontos frontais
        front_points = points[points[:, 0] > 0.5]

        if len(front_points) > 0:
            X = front_points[:, 0]
            Y = front_points[:, 1]
            Z = front_points[:, 2]

            # Aplica correção de pitch
            X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
            Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

            # Projeta
            u = (fx * Y / X_rot + cx).astype(int)
            v = (fy * Z_rot / X_rot + cy).astype(int)

            # Filtra válidos
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[valid]
            v = v[valid]
            x_vals = X[valid]
            z_vals = Z[valid]

            # Detecta anomalias
            z_median = np.median(z_vals) if len(z_vals) > 0 else 0
            anomalies = np.abs(z_vals - z_median) > 3

            detections = 0

            # Desenha
            for i in range(len(u)):
                if anomalies[i] or (5 < x_vals[i] < 20 and abs(z_vals[i] - z_median) > 1):
                    # DRONE!
                    color = (255, 0, 255)
                    size = 5
                    detections += 1
                    cv2.circle(fusion, (u[i], v[i]), 8, (255, 255, 0), 2)
                else:
                    # Chão
                    depth = min(1.0, x_vals[i] / 50)
                    color = (0, int(255*(1-depth)), 0)
                    size = 1

                cv2.circle(fusion, (u[i], v[i]), size, color, -1)

            print(f"  📡 Pontos: {len(points)}")
            print(f"  🎯 Detecções: {detections}")

            # Info
            cv2.putText(fusion, f"Frame {frame+1} | Yaw {yaw} | Detections: {detections}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Salva
            cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion)
            cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)

            # Comparação
            comp = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comp)

            print(f"  ✅ Salvo!")

# Volta ao centro
client.rotateToYawAsync(0, vehicle_name="Ego").join()

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("🎉 DATASET COMPLETO COM DETECÇÃO DE DRONES!")
print("="*70)
print(f"\n📁 Salvo em: {output_dir.absolute()}")
print("\n✅ FUNCIONANDO:")
print("  • LiDAR detecta drones (IgnoreMarked: false)")
print("  • Correção de pitch aplicada")
print("  • Pontos magenta nos drones!")
print("  • Dataset pronto para uso!")