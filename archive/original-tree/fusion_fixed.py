#!/usr/bin/env python3
"""
Fusão CORRIGIDA - usando detecção de anomalias ANTES da filtragem
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🎯 FUSÃO CORRIGIDA - DETECÇÃO FUNCIONANDO!")
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

print("\n📍 Posicionando drones...")

# Mesma posição do teste que funcionou
client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()

positions = [
    ("Drone3", 5, -2, -10),
    ("Drone4", 7, 2, -10),
    ("Intruder1", 10, 0, -8)
]

for name, x, y, z in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"  {name}: {x}m frente")

time.sleep(3)

output_dir = Path("fusion_fixed")
output_dir.mkdir(exist_ok=True)

print("\n📸 Capturando...")

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

    # Correção de pitch
    camera_pitch = -15
    pitch_rad = np.radians(camera_pitch)

    # Filtra pontos frontais
    front_mask = points[:, 0] > 0.5
    front_points = points[front_mask]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        # IMPORTANTE: Detecta anomalias ANTES de filtrar!
        z_mean = Z.mean()
        z_std = Z.std()
        anomalies_mask = np.abs(Z - z_mean) > 2  # Anomalias nos dados originais

        print(f"\n📊 Análise PRÉ-filtragem:")
        print(f"  Total pontos frontais: {len(front_points)}")
        print(f"  Z médio: {z_mean:.2f} ± {z_std:.2f}")
        print(f"  Anomalias detectadas: {anomalies_mask.sum()}")

        # Aplica correção de pitch
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        # Projeta
        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        # Filtra válidos
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        # Aplica filtros mantendo correspondência
        u_valid = u[valid]
        v_valid = v[valid]
        x_valid = X[valid]
        z_valid = Z[valid]
        anomalies_valid = anomalies_mask[valid]  # Mantém info de anomalia

        print(f"\n📊 Análise PÓS-filtragem:")
        print(f"  Pontos válidos: {len(u_valid)}")
        print(f"  Anomalias válidas: {anomalies_valid.sum()}")

        detections = 0

        # Desenha
        for i in range(len(u_valid)):
            # Usa a flag de anomalia calculada ANTES
            if anomalies_valid[i]:
                # ANOMALIA = POSSÍVEL DRONE!
                color = (255, 0, 255)  # Magenta
                size = 5
                detections += 1
                # Círculo de destaque
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (255, 255, 0), 2)
            else:
                # Normal
                depth = min(1.0, x_valid[i] / 50)
                color = (0, int(255*(1-depth)), 0)
                size = 1

            cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

        print(f"  🎯 Detecções projetadas: {detections}")

        # Info
        cv2.putText(fusion, f"FIXED: Anomaly detection BEFORE filtering", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(fusion, f"Detections: {detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(fusion, f"Total anomalies: {anomalies_mask.sum()}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Salva
        cv2.imwrite(str(output_dir / "fusion.png"), fusion)
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)

        # Comparação
        comp = np.hstack([img_bgr, fusion])
        cv2.imwrite(str(output_dir / "comparison.png"), comp)

        print(f"\n💾 Salvo em {output_dir}")

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
print("✅ CORREÇÃO APLICADA!")
print("="*70)
print("\nDIFERENÇA CHAVE:")
print("  ❌ ERRADO: Calcular anomalias após filtrar (perde dados)")
print("  ✅ CERTO: Calcular anomalias antes de filtrar (preserva detecções)")
print("="*70)