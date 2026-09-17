#!/usr/bin/env python3
"""
Debug profundo do problema do LiDAR
"""

import cosysairsim as airsim
import numpy as np
import time
import sys

print("\n" + "="*70)
print("🔍 DEBUG PROFUNDO DO LIDAR")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado!")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Prepara Ego
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")

# NÃO decola - fica no chão
print("\n📍 Teste 1: DRONE NO CHÃO (sem decolar)")
time.sleep(2)

# Captura LiDAR no chão
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    Z = points[:, 2]

    print(f"Pontos capturados: {len(points)}")
    print(f"Z range: {Z.min():.2f} to {Z.max():.2f}")
    print(f"Z médio: {Z.mean():.2f}")

    # Se o drone está no chão, deveria ver:
    # - Chão em Z ≈ 0 ou ligeiramente negativo
    # - Nada muito abaixo (não há chão abaixo do chão!)

    print("\n📊 Análise:")
    if Z.min() > 5:
        print("  ❌ PROBLEMA CONFIRMADO!")
        print("  Mesmo no chão, Z mínimo > 5")
        print("  O LiDAR está claramente mal configurado ou com bug")
    else:
        print("  ✅ LiDAR funcionando corretamente no chão")

# Agora decola e testa
print("\n📍 Teste 2: DECOLANDO para 10m")
client.takeoffAsync(vehicle_name="Ego").join()
time.sleep(3)
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
time.sleep(2)

# Captura no ar
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    print(f"\nPontos capturados: {len(points)}")
    print(f"Z range: {Z.min():.2f} to {Z.max():.2f}")

    # Pontos próximos
    near = (X > 0.1) & (X < 20) & (np.abs(Y) < 10)
    if np.any(near):
        Z_near = Z[near]
        print(f"\nPontos próximos (X<20m):")
        print(f"  Quantidade: {len(Z_near)}")
        print(f"  Z range: {Z_near.min():.2f} to {Z_near.max():.2f}")
        print(f"  Z médio: {Z_near.mean():.2f}")

    # Calcula ângulos
    angles = np.degrees(np.arctan2(Z, np.sqrt(X**2 + Y**2)))
    print(f"\nÂngulos verticais:")
    print(f"  Min: {angles.min():.1f}°")
    print(f"  Max: {angles.max():.1f}°")

    # Histograma de ângulos
    hist, bins = np.histogram(angles, bins=10)
    print(f"\nDistribuição angular:")
    for i in range(len(hist)):
        print(f"  {bins[i]:.1f}° a {bins[i+1]:.1f}°: {hist[i]} pontos")

print("\n" + "="*70)
print("DIAGNÓSTICO FINAL")
print("="*70)

print("""
POSSÍVEIS CAUSAS DO PROBLEMA:

1. BUG NO AIRSIM: O LiDAR pode estar com bug e ignorando
   as configurações de FOV vertical

2. COORDENADAS INVERTIDAS: Z positivo pode significar
   "para baixo" no referencial do AirSim

3. SENSOR MAL POSICIONADO: O sensor pode estar fisicamente
   apontado para cima no modelo do drone

4. LIMITAÇÃO DO SIMULADOR: O terreno plano pode não estar
   gerando retornos do LiDAR

SOLUÇÕES POSSÍVEIS:

A. Tentar inverter o sinal de Z na projeção
B. Usar um mapa com terreno mais complexo
C. Reportar bug para os desenvolvedores do AirSim
D. Usar uma versão diferente do AirSim
""")

# Pousa
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

print("\n✅ Debug concluído!")