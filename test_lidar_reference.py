#!/usr/bin/env python3
"""
Teste para entender o referencial do LiDAR
"""

import cosysairsim as airsim
import numpy as np
import time
import sys

print("\n" + "="*70)
print("TESTE DE REFERENCIAL DO LIDAR")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado!")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Habilita e decola Ego
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")
client.takeoffAsync(vehicle_name="Ego").join()
time.sleep(3)

print("\n📊 Testando em diferentes alturas...")

alturas = [5, 10, 20]

for altura in alturas:
    print(f"\n{'='*50}")
    print(f"Teste na altura: {altura}m")

    # Move para altura
    client.moveToPositionAsync(0, 0, -altura, 3, vehicle_name="Ego").join()
    time.sleep(2)

    # Pega pose do drone
    pose = client.simGetVehiclePose(vehicle_name="Ego")
    drone_x = pose.position.x_val
    drone_y = pose.position.y_val
    drone_z = pose.position.z_val

    print(f"Posição do drone (API):")
    print(f"  X: {drone_x:.2f}, Y: {drone_y:.2f}, Z: {drone_z:.2f}")
    print(f"  Altura: {-drone_z:.2f}m")

    # Captura LiDAR
    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    if lidar_data and len(lidar_data.point_cloud) > 3:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Analisa pontos
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        print(f"\nPontos LiDAR ({len(points)} total):")
        print(f"  X range: {X.min():.2f} to {X.max():.2f}")
        print(f"  Y range: {Y.min():.2f} to {Y.max():.2f}")
        print(f"  Z range: {Z.min():.2f} to {Z.max():.2f}")

        # Pontos diretamente abaixo (chão)
        below_mask = (np.abs(X) < 5) & (np.abs(Y) < 5) & (X > 0.1)
        if np.any(below_mask):
            Z_below = Z[below_mask]
            print(f"\nPontos abaixo do drone (X,Y < 5m):")
            print(f"  Quantidade: {len(Z_below)}")
            print(f"  Z médio: {Z_below.mean():.2f}")
            print(f"  Z range: {Z_below.min():.2f} to {Z_below.max():.2f}")

            # TESTE CRÍTICO
            print(f"\n🔍 TESTE DE REFERENCIAL:")
            print(f"  Se Z médio ≈ 0: pontos estão em coord GLOBAL (chão = 0)")
            print(f"  Se Z médio ≈ -{altura}: pontos estão em coord LOCAL (chão abaixo)")

            if abs(Z_below.mean()) < 2:
                print("  ✅ CONCLUSÃO: Coordenadas GLOBAIS (Z=0 é o solo)")
            elif abs(Z_below.mean() + altura) < 2:
                print("  ✅ CONCLUSÃO: Coordenadas LOCAIS (Z=0 é o drone)")
            else:
                print("  ❓ CONCLUSÃO: Sistema misto ou outro referencial")

        # Pontos no horizonte
        horizon_mask = (X > 50) & (np.abs(Z) < 5)
        if np.any(horizon_mask):
            print(f"\nPontos no horizonte (X > 50m, |Z| < 5):")
            print(f"  Quantidade: {np.sum(horizon_mask)}")
            print(f"  Isso explica a faixa horizontal na imagem!")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

print("\n" + "="*70)
print("CONCLUSÃO DO TESTE")
print("="*70)