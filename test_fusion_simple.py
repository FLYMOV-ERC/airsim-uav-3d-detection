#!/usr/bin/env python3
"""
Teste simples e rápido da fusão
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 TESTE RÁPIDO DE FUSÃO")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n✅ Veículos: {vehicles}")

# Prepara e decola apenas Ego
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")
client.takeoffAsync(vehicle_name="Ego").join()

# Posiciona
print("📍 Posicionando Ego...")
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()
time.sleep(2)

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

    print(f"\n📊 ANÁLISE:")
    print(f"  Total pontos: {len(points):,}")
    print(f"  Z min: {points[:,2].min():.2f}m")
    print(f"  Z max: {points[:,2].max():.2f}m")

    # Pontos acima
    above = points[points[:,2] < -1]
    print(f"  Pontos ACIMA (Z<-1): {len(above)}")

    # Cria fusão simples
    h, w = img_bgr.shape[:2]
    fusion = img_bgr.copy()

    # Parâmetros câmera
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    # Filtra pontos frontais
    front = points[points[:, 0] > 1]

    if len(front) > 0:
        X = front[:, 0]
        Y = front[:, 1]
        Z = front[:, 2]

        # Projeta
        u = (fx * Y / X + cx).astype(int)
        v = (fy * Z / X + cy).astype(int)

        # Filtra válidos
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid]
        v = v[valid]
        z_vals = Z[valid]

        print(f"  Pontos projetados: {len(u):,}")

        # Desenha com cores por altura
        for i in range(len(u)):
            if z_vals[i] < 0:  # Acima
                color = (255, 0, 255)  # Magenta
                size = 4
            elif z_vals[i] < 5:  # Próximo
                color = (0, 255, 255)  # Amarelo
                size = 2
            else:  # Chão
                color = (0, 255, 0)  # Verde
                size = 1

            cv2.circle(fusion, (u[i], v[i]), size, color, -1)

        # Info
        cv2.putText(fusion, f"LiDAR: {len(points)} pts", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(fusion, f"DrawDebugPoints: ON", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Salva
        output_dir = Path("test_fusion_debug")
        output_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(output_dir / "fusion.png"), fusion)
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)

        # Comparação
        comp = np.hstack([img_bgr, fusion])
        cv2.imwrite(str(output_dir / "comparison.png"), comp)

        print(f"\n💾 Salvo em {output_dir}")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

print("\n✅ COMPLETO!")
print("\n💡 Com DrawDebugPoints ativo, você deve ver:")
print("   • Pontos verdes no ambiente 3D = hits do LiDAR")
print("   • Se não há pontos nos drones = drones não têm colisão LiDAR")