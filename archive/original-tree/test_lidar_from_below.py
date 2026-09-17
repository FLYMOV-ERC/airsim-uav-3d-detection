#!/usr/bin/env python3
"""
Testa LiDAR olhando de baixo para cima
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🔍 TESTE LIDAR OLHANDO PARA CIMA")
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

# Decola todos
print("🛫 Decolando...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

print("\n📍 TESTE 1: Ego BAIXO, outros drones ACIMA")
print("="*50)

# Ego bem baixo
client.moveToPositionAsync(0, 0, -5, 5, vehicle_name="Ego").join()
print("   Ego: 5m altura (BEM BAIXO)")

# Outros drones ACIMA
if "Drone3" in vehicles:
    client.moveToPositionAsync(10, 0, -15, 5, vehicle_name="Drone3").join()
    print("   Drone3: 10m frente, 15m altura (10m ACIMA)")

if "Drone4" in vehicles:
    client.moveToPositionAsync(15, 5, -20, 5, vehicle_name="Drone4").join()
    print("   Drone4: 15m frente, 20m altura (15m ACIMA)")

if "Intruder1" in vehicles:
    client.moveToPositionAsync(20, -5, -25, 5, vehicle_name="Intruder1").join()
    print("   Intruder1: 20m frente, 25m altura (20m ACIMA)")

time.sleep(3)

# Captura e analisa
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")
if lidar_data:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    z_neg = points[points[:, 2] < 0]
    print(f"\n   📡 Pontos totais: {len(points)}")
    print(f"   📊 Pontos ACIMA (Z<0): {len(z_neg)}")

    if len(z_neg) > 0:
        print(f"      Z range: {z_neg[:,2].min():.1f} a {z_neg[:,2].max():.1f}m")
        print(f"      X range: {z_neg[:,0].min():.1f} a {z_neg[:,0].max():.1f}m")

print("\n" + "="*50)
print("📍 TESTE 2: Ego ALTO, outros drones ABAIXO")
print("="*50)

# Ego bem alto
client.moveToPositionAsync(0, 0, -30, 5, vehicle_name="Ego").join()
print("   Ego: 30m altura (BEM ALTO)")

# Outros drones ABAIXO
if "Drone3" in vehicles:
    client.moveToPositionAsync(10, 0, -20, 5, vehicle_name="Drone3").join()
    print("   Drone3: 10m frente, 20m altura (10m ABAIXO)")

if "Drone4" in vehicles:
    client.moveToPositionAsync(15, 5, -15, 5, vehicle_name="Drone4").join()
    print("   Drone4: 15m frente, 15m altura (15m ABAIXO)")

if "Intruder1" in vehicles:
    client.moveToPositionAsync(20, -5, -10, 5, vehicle_name="Intruder1").join()
    print("   Intruder1: 20m frente, 10m altura (20m ABAIXO)")

time.sleep(3)

# Captura e analisa
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")
if lidar_data:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    z_pos_low = points[(points[:, 2] > 5) & (points[:, 2] < 25)]
    print(f"\n   📡 Pontos totais: {len(points)}")
    print(f"   📊 Pontos 5-25m ABAIXO: {len(z_pos_low)}")

    if len(z_pos_low) > 0:
        # Analisa clusters
        for dist in [10, 15, 20]:
            cluster = z_pos_low[(z_pos_low[:,0] > dist-2) & (z_pos_low[:,0] < dist+2)]
            if len(cluster) > 0:
                print(f"      Cluster ~{dist}m: {len(cluster)} pontos")

# Captura imagem final para fusão
print("\n📸 Capturando fusão final...")

responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")

if responses[0].image_data_uint8 and lidar_data:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    fusion = img_bgr.copy()

    # Projeta pontos
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    front = points[points[:, 0] > 1]
    if len(front) > 0:
        X = front[:, 0]
        Y = front[:, 1]
        Z = front[:, 2]

        u = (fx * Y / X + cx).astype(int)
        v = (fy * Z / X + cy).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid]
        v = v[valid]
        z_vals = Z[valid]

        # Desenha com destaque para não-chão
        for i in range(len(u)):
            if z_vals[i] < 25:  # Não é chão distante
                if z_vals[i] < 0:
                    color = (255, 0, 255)  # Magenta = acima
                    size = 5
                elif z_vals[i] < 15:
                    color = (0, 255, 255)  # Amarelo = possível drone
                    size = 3
                else:
                    color = (0, 255, 0)  # Verde = chão próximo
                    size = 1
            else:
                color = (100, 100, 100)  # Cinza = chão distante
                size = 1

            cv2.circle(fusion, (u[i], v[i]), size, color, -1)

        # Info
        cv2.putText(fusion, f"Vista de 30m altura", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(fusion, f"Drones devem aparecer amarelos", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Salva
        output_dir = Path("lidar_test_vertical")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "fusion_from_above.png"), fusion)
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)
        print(f"\n💾 Imagens salvas em {output_dir}")

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
print("📊 CONCLUSÃO:")
print("="*70)
if 'z_neg' in locals() and len(z_neg) > 100:
    print("✅ LiDAR detecta objetos acima quando posicionado baixo!")
else:
    print("⚠️ Detecção limitada de objetos aéreos")
    print("\nPOSSÍVEIS CAUSAS:")
    print("1. Drones podem não ter superfície reflexiva no AirSim")
    print("2. Configuração do LiDAR ainda pode estar limitada")
    print("3. Pode ser necessário aumentar DrawDebugPoints")
print("="*70)