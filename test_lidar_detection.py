#!/usr/bin/env python3
"""
Teste de detecção do LiDAR nos drones
"""

import cosysairsim as airsim
import numpy as np
import time
import sys

print("\n" + "="*70)
print("TESTE DE DETECÇÃO LIDAR")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except:
    print("❌ Erro ao conectar")
    sys.exit(1)

# Prepara e decola apenas Ego e Drone3
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")
client.enableApiControl(True, "Drone3")
client.armDisarm(True, "Drone3")

print("\n🛫 Decolando...")
client.takeoffAsync(vehicle_name="Ego").join()
client.takeoffAsync(vehicle_name="Drone3").join()
time.sleep(3)

# Coloca Drone3 bem na frente do Ego
print("\n📍 Posicionando Drone3 diretamente na frente...")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(5, 0, -10, 3, vehicle_name="Drone3").join()  # 5m na frente, mesma altura
time.sleep(2)

print("   Ego em: x=0, y=0, z=-10 (10m altura)")
print("   Drone3 em: x=5, y=0, z=-10 (5m na frente, mesma altura)")

# Captura LiDAR
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Procura pontos na região do Drone3 (4-6m frente, ±1m lateral)
    drone_region = (X > 4) & (X < 6) & (np.abs(Y) < 1)
    points_in_drone = np.sum(drone_region)

    print(f"\n📊 Análise da região do Drone3 (4-6m frente):")
    print(f"   Pontos encontrados: {points_in_drone}")

    if points_in_drone > 0:
        Z_drone = Z[drone_region]
        X_drone = X[drone_region]

        print(f"   Z range: {Z_drone.min():.2f} to {Z_drone.max():.2f}")
        print(f"   Z médio: {Z_drone.mean():.2f}")

        # Se Z ≈ 10 (altura do drone em coord globais), está detectando
        z_at_drone_level = np.sum((Z_drone > 9) & (Z_drone < 11))

        if z_at_drone_level > 10:
            print(f"   ✅ {z_at_drone_level} pontos no nível do drone!")
            print("   O LiDAR ESTÁ detectando o drone!")
        else:
            print(f"   ❌ Apenas {z_at_drone_level} pontos no nível do drone")
            print("   O LiDAR NÃO está detectando o drone propriamente")

            # Verifica se está vendo através
            if np.all(Z_drone > 11):
                print("   🔴 Todos os pontos têm Z > 11")
                print("   O LiDAR está ATRAVESSANDO o drone!")
                print("\n   PROBLEMA: O drone não está configurado como obstáculo")
                print("   SOLUÇÃO: Verificar configuração do modelo no Unreal")

# Move Drone3 para diferentes distâncias
distances = [3, 7, 10, 15]
print("\n📊 Testando diferentes distâncias...")

for dist in distances:
    client.moveToPositionAsync(dist, 0, -10, 3, vehicle_name="Drone3").join()
    time.sleep(1)

    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")
    if lidar_data:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        X = points[:, 0]
        Z = points[:, 2]

        # Pontos na distância do drone ±1m
        at_drone = (X > dist-1) & (X < dist+1) & (np.abs(points[:, 1]) < 1)

        if np.any(at_drone):
            Z_at = Z[at_drone]
            z_drone_level = np.sum((Z_at > 9) & (Z_at < 11))
            print(f"   {dist}m: {z_drone_level} pontos no nível do drone (Z≈10)")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.landAsync(vehicle_name="Drone3").join()
client.armDisarm(False, "Ego")
client.armDisarm(False, "Drone3")
client.enableApiControl(False, "Ego")
client.enableApiControl(False, "Drone3")

print("\n" + "="*70)
print("CONCLUSÃO")
print("="*70)
print("""
Se o LiDAR não detecta os drones, possíveis soluções:

1. NO UNREAL ENGINE:
   - Verificar se os drones têm collision mesh
   - Adicionar tag "Lidar" aos drones
   - Verificar material/textura dos drones

2. NO SETTINGS.JSON:
   - Tentar "IgnoreMarked": true (contraintuitivo)
   - Adicionar "DrawDebugPoints": false

3. WORKAROUND:
   - Usar segmentação de imagem ao invés de LiDAR
   - Usar depth camera ao invés de LiDAR
   - Adicionar objetos especiais para detecção
""")