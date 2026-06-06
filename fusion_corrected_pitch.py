#!/usr/bin/env python3
"""
Fusão com correção de pitch da câmera
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 FUSÃO CORRIGIDA - COMPENSANDO PITCH DA CÂMERA")
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

# Ego
client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()

# Drones na frente
positions = [
    ("Drone3", 5, -2, -10, "5m frente, mesmo nível"),
    ("Drone4", 7, 2, -10, "7m frente, mesmo nível"),
    ("Intruder1", 10, 0, -8, "10m frente, 2m acima")
]

for name, x, y, z, desc in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"  {name}: {desc}")

time.sleep(3)

output_dir = Path("fusion_corrected")
output_dir.mkdir(exist_ok=True)

print("\n📸 CAPTURANDO COM CORREÇÃO DE PITCH...")

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

    print(f"\n📊 Análise:")
    print(f"  Total pontos: {len(points):,}")

    h, w = img_bgr.shape[:2]

    # Parâmetros da câmera
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    # CORREÇÃO CRÍTICA: Câmera tem pitch de -15° (olhando para baixo)
    camera_pitch = -15  # graus
    pitch_rad = np.radians(camera_pitch)

    print(f"  📷 Câmera com pitch: {camera_pitch}°")

    # Cria 3 fusões para comparar
    fusion_original = img_bgr.copy()
    fusion_corrected = img_bgr.copy()
    fusion_inverted = img_bgr.copy()

    # Filtra pontos frontais
    front_mask = points[:, 0] > 1
    front_points = points[front_mask]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        # Identifica anomalias (possíveis drones)
        z_mean = Z.mean()
        anomalies = np.abs(Z - z_mean) > 2

        # PROJEÇÃO 1: Original (incorreta)
        u = (fx * Y / X + cx).astype(int)
        v_original = (fy * Z / X + cy).astype(int)

        # PROJEÇÃO 2: Com correção de pitch
        # Rotaciona pontos para compensar a inclinação da câmera
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        v_corrected = (fy * Z_rot / X_rot + cy).astype(int)

        # PROJEÇÃO 3: Z invertido (teste)
        v_inverted = (fy * (-Z) / X + cy).astype(int)

        # Desenha em cada fusão
        for i in range(len(u)):
            # Cor baseada em anomalia
            if anomalies[i]:
                color = (255, 0, 255)  # Magenta = anomalia
                size = 4
            else:
                color = (0, 255, 0)  # Verde = normal
                size = 1

            # Original
            if 0 <= u[i] < w and 0 <= v_original[i] < h:
                cv2.circle(fusion_original, (u[i], v_original[i]), size, color, -1)

            # Corrigida
            if 0 <= u[i] < w and 0 <= v_corrected[i] < h:
                cv2.circle(fusion_corrected, (u[i], v_corrected[i]), size, color, -1)

            # Invertida
            if 0 <= u[i] < w and 0 <= v_inverted[i] < h:
                cv2.circle(fusion_inverted, (u[i], v_inverted[i]), size, color, -1)

        print(f"  🎯 Anomalias detectadas: {anomalies.sum()}")

        # Adiciona labels
        cv2.putText(fusion_original, "ORIGINAL (sem correcao)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(fusion_original, "Pontos aparecem ABAIXO", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(fusion_corrected, "CORRIGIDA (pitch -15)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(fusion_corrected, "Pontos devem alinhar com drones", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(fusion_inverted, "Z INVERTIDO (teste)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(fusion_inverted, "Se funcionar, Z esta invertido", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Salva todas
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)
        cv2.imwrite(str(output_dir / "fusion_original.png"), fusion_original)
        cv2.imwrite(str(output_dir / "fusion_corrected.png"), fusion_corrected)
        cv2.imwrite(str(output_dir / "fusion_inverted.png"), fusion_inverted)

        # Comparação 2x2
        top = np.hstack([img_bgr, fusion_original])
        bottom = np.hstack([fusion_corrected, fusion_inverted])
        comparison = np.vstack([top, bottom])

        cv2.imwrite(str(output_dir / "comparison_all.png"), comparison)

        print(f"\n💾 Salvo em {output_dir}")
        print("\n📋 Verifique qual projeção alinha melhor com os drones!")

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n✅ TESTE COMPLETO!")
print("\n🔍 Compare as 4 imagens:")
print("  1. Original (sem LiDAR)")
print("  2. Fusão original (pontos abaixo)")
print("  3. Fusão corrigida (compensando pitch)")
print("  4. Fusão invertida (testando inversão de Z)")