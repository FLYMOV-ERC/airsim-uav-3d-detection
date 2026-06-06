#!/usr/bin/env python3
"""
Teste agressivo de detecção - verifica se há QUALQUER ponto não-chão
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🔍 TESTE AGRESSIVO DE DETECÇÃO")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n✅ Veículos: {vehicles}")

# Info sobre configuração
print("\n📋 Configuração esperada do LiDAR:")
print("  • IgnoreMarked: false (deve detectar veículos)")
print("  • 64 canais")
print("  • FOV: -45° a +45°")
print("  • DrawDebugPoints: true")

# Prepara todos
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\n🛫 Decolando...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

print("\n📍 Teste 1: Drones MUITO próximos e alinhados")
print("-"*50)

# Ego
client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()

# Coloca drone BEM na frente
if "Drone3" in vehicles:
    client.moveToPositionAsync(3, 0, -10, 5, vehicle_name="Drone3").join()
    print("  Drone3: Apenas 3m na frente (MUITO PERTO)")

time.sleep(2)

# Captura 1
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")
if lidar_data:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    print(f"\n  Total pontos: {len(points)}")

    # Analisa região 2-4m frente
    front = points[(points[:, 0] > 2) & (points[:, 0] < 4)]
    print(f"  Pontos 2-4m frente: {len(front)}")

    if len(front) > 0:
        z_unique = np.unique(np.round(front[:, 2], 1))
        print(f"  Valores Z únicos: {z_unique}")

        # Qualquer coisa diferente do padrão é drone
        non_standard = front[np.abs(front[:, 2] - 11.73) > 0.5]
        if len(non_standard) > 0:
            print(f"  ✅ {len(non_standard)} pontos ANORMAIS - possível drone!")
        else:
            print(f"  ❌ Todos os pontos são chão (Z≈11.73m)")

print("\n" + "-"*50)
print("📍 Teste 2: Drone ACIMA do sensor")
print("-"*50)

# Move Drone3 para CIMA
if "Drone3" in vehicles:
    client.moveToPositionAsync(5, 0, -5, 5, vehicle_name="Drone3").join()
    print("  Drone3: 5m frente, 5m ACIMA do Ego")

time.sleep(2)

# Captura 2
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")
if lidar_data:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    # Procura por QUALQUER ponto com Z negativo
    negative_z = points[points[:, 2] < 0]
    print(f"\n  Pontos com Z < 0 (acima): {len(negative_z)}")

    if len(negative_z) > 0:
        print(f"  ✅ DETECTOU OBJETO ACIMA!")
        print(f"     X range: {negative_z[:, 0].min():.1f} - {negative_z[:, 0].max():.1f}")
        print(f"     Z range: {negative_z[:, 2].min():.1f} - {negative_z[:, 2].max():.1f}")

print("\n" + "-"*50)
print("📍 Teste 3: Análise estatística completa")
print("-"*50)

# Move todos para frente do Ego
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

positions = [
    ("Drone3", 5, -2, -15),
    ("Drone4", 5, 2, -15),
    ("Intruder1", 8, 0, -12)
]

for name, x, y, z in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()

time.sleep(3)

# Captura final com análise completa
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    print(f"\n  📊 ANÁLISE ESTATÍSTICA COMPLETA:")
    print(f"  Total: {len(points)} pontos")

    # Histograma de Z
    z_values = points[:, 2]
    z_bins = np.histogram(z_values, bins=50)

    # Encontra valores Z anormais (outliers)
    z_mean = z_values.mean()
    z_std = z_values.std()
    outliers = points[np.abs(z_values - z_mean) > 2 * z_std]

    print(f"  Z médio: {z_mean:.2f}m ± {z_std:.2f}")
    print(f"  Outliers (>2σ): {len(outliers)} pontos")

    if len(outliers) > 0:
        print(f"  ✅ POSSÍVEIS OBJETOS NÃO-CHÃO DETECTADOS!")

    # Verifica descontinuidades
    front_5_10 = points[(points[:, 0] > 4) & (points[:, 0] < 10)]
    if len(front_5_10) > 0:
        z_front = front_5_10[:, 2]
        z_variance = np.var(z_front)
        print(f"  Variância Z (5-10m frente): {z_variance:.3f}")

        if z_variance > 0.1:
            print(f"  ✅ Alta variância - possíveis objetos!")
        else:
            print(f"  ❌ Baixa variância - apenas superfície plana")

    # Salva visualização
    img_data = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if img_data[0].image_data_uint8:
        img_1d = np.frombuffer(img_data[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(img_data[0].height, img_data[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Marca onde drones deveriam estar
        h, w = img_bgr.shape[:2]
        cv2.putText(img_bgr, "Drones should be visible here", (w//4, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.circle(img_bgr, (w//2, h//2), 50, (0, 255, 255), 3)

        output_dir = Path("detection_test")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "test.png"), img_bgr)
        print(f"\n  💾 Imagem salva em {output_dir}")

# Pousa
print("\n🛬 Finalizando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("📌 DIAGNÓSTICO FINAL:")
print("="*70)

has_detection = False
if 'outliers' in locals() and len(outliers) > 0:
    has_detection = True
if 'negative_z' in locals() and len(negative_z) > 0:
    has_detection = True
if 'non_standard' in locals() and len(non_standard) > 0:
    has_detection = True

if has_detection:
    print("✅ ALGUMA DETECÇÃO ANORMAL ENCONTRADA!")
else:
    print("❌ NENHUMA DETECÇÃO DE DRONES")
    print("\n🔧 SOLUÇÕES ALTERNATIVAS:")
    print("1. Os drones SimpleFlight não têm colisão no Blocks")
    print("2. Tente usar o ambiente CityEnviron ou AirSimNH")
    print("3. Ou adicione objetos customizados no Unreal Engine")
    print("4. Ou use simulação de obstáculos virtuais")

print("="*70)