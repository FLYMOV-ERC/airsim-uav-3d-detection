#!/usr/bin/env python3
"""
Debug detalhado do LiDAR
"""

import cosysairsim as airsim
import numpy as np
import time

print("\n" + "="*70)
print("🔍 DEBUG DO LIDAR")
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

# Posiciona Ego
print("\n📍 Posicionando Ego a 20m de altura...")
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()
time.sleep(3)

# Posiciona outros drones ACIMA do Ego para teste
print("\n📍 Posicionando outros drones ACIMA do Ego...")

# Drone3 ACIMA do Ego
if "Drone3" in vehicles:
    client.enableApiControl(True, "Drone3")
    client.armDisarm(True, "Drone3")
    client.takeoffAsync(vehicle_name="Drone3").join()
    client.moveToPositionAsync(10, 0, -10, 5, vehicle_name="Drone3").join()
    print("   Drone3: 10m frente, 10m ACIMA do Ego")

# Drone4 no mesmo nível
if "Drone4" in vehicles:
    client.enableApiControl(True, "Drone4")
    client.armDisarm(True, "Drone4")
    client.takeoffAsync(vehicle_name="Drone4").join()
    client.moveToPositionAsync(15, 5, -20, 5, vehicle_name="Drone4").join()
    print("   Drone4: 15m frente, MESMO nível do Ego")

time.sleep(3)

# Captura LiDAR
print("\n📡 Capturando LiDAR...")
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    print(f"\n📊 ANÁLISE DETALHADA:")
    print(f"Total de pontos: {len(points):,}")

    # Análise de Z
    z_values = points[:, 2]
    print(f"\nDistribuição Z (vertical):")
    print(f"  Min Z: {z_values.min():.2f}m")
    print(f"  Max Z: {z_values.max():.2f}m")
    print(f"  Média Z: {z_values.mean():.2f}m")

    # Histograma de Z
    print(f"\nHistograma Z:")
    for i in range(-50, 51, 10):
        count = np.sum((z_values >= i) & (z_values < i+10))
        if count > 0:
            bar = "█" * min(50, int(count/100))
            print(f"  {i:3d} a {i+10:3d}m: {count:5d} pts {bar}")

    # Pontos negativos (acima)
    z_neg = z_values[z_values < 0]
    if len(z_neg) > 0:
        print(f"\n✅ DETECTOU {len(z_neg)} PONTOS ACIMA (Z negativo)!")
        print(f"   Range Z negativo: {z_neg.min():.2f} a {z_neg.max():.2f}m")

        # Analisa pontos acima
        above_points = points[z_values < 0]
        print(f"\n   Pontos acima - análise X,Y:")
        print(f"   X range: {above_points[:,0].min():.1f} a {above_points[:,0].max():.1f}m")
        print(f"   Y range: {above_points[:,1].min():.1f} a {above_points[:,1].max():.1f}m")

        # Verifica se está na posição esperada do Drone3
        drone3_x = above_points[(above_points[:,0] > 8) & (above_points[:,0] < 12)]
        if len(drone3_x) > 0:
            print(f"\n   🚁 POSSÍVEL DRONE3 DETECTADO! {len(drone3_x)} pontos na região X=10m")
    else:
        print(f"\n❌ NENHUM ponto com Z negativo (nada acima detectado)")

    # Pontos frontais
    front = points[points[:, 0] > 5]
    print(f"\n📍 Pontos frontais (X > 5m): {len(front):,}")

    if len(front) > 0:
        # Agrupa por distância
        for dist in [10, 15, 20, 25, 30]:
            region = front[(front[:,0] > dist-2) & (front[:,0] < dist+2)]
            if len(region) > 0:
                z_mean = region[:,2].mean()
                z_min = region[:,2].min()
                z_max = region[:,2].max()
                print(f"   X≈{dist}m: {len(region)} pts, Z={z_min:.1f} a {z_max:.1f}m (média {z_mean:.1f}m)")

                # Verifica se há pontos acima do chão nesta distância
                above_ground = region[region[:,2] < -1]
                if len(above_ground) > 0:
                    print(f"      → {len(above_ground)} pontos ACIMA do chão!")

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
if 'z_neg' in locals() and len(z_neg) > 0:
    print("✅ LIDAR DETECTANDO OBJETOS ACIMA!")
else:
    print("❌ LIDAR AINDA NÃO DETECTA OBJETOS ACIMA")
    print("   Possíveis causas:")
    print("   1. Configuração não foi carregada (reinicie o AirSim)")
    print("   2. FOV ainda limitado no servidor")
    print("   3. Drones não têm pontos LiDAR refletivos")
print("="*70)