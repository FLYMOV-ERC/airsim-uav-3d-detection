#!/usr/bin/env python3
"""
Teste rápido - Verifica se drones estão voando na altura correta
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🔍 TESTE RÁPIDO - VERIFICAÇÃO DE ALTURA DOS DRONES")
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

# Decola e AGUARDA
print("\n🛫 Decolando todos os drones...")
takeoff_tasks = []
for v in vehicles:
    try:
        task = client.takeoffAsync(vehicle_name=v)
        takeoff_tasks.append(task)
        print(f"   {v}: decolando...")
    except:
        pass

# Aguarda todas as decolagens
for task in takeoff_tasks:
    task.join()

print("✅ Todos decolaram! Aguardando estabilização...")
time.sleep(3)

# Verifica posições iniciais
print("\n📍 POSIÇÕES APÓS DECOLAGEM:")
for v in vehicles:
    try:
        pose = client.simGetVehiclePose(vehicle_name=v)
        z = pose.position.z_val
        print(f"   {v}: z={z:.2f} (altura={-z:.2f}m)")
    except:
        pass

# Move para alturas específicas
print("\n🎯 Movendo drones para alturas de teste...")
test_positions = [
    ("Ego", 0, 0, -15, "15m de altura"),
    ("Drone3", 10, -3, -15, "15m, 10m frente"),
    ("Drone4", 15, 3, -18, "18m, 15m frente"),
    ("Intruder1", 20, 0, -12, "12m, 20m frente")
]

for name, x, y, z, desc in test_positions:
    if name in vehicles:
        print(f"   {name}: Movendo para {desc}...")
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()

time.sleep(2)

# Verifica posições finais
print("\n📍 POSIÇÕES FINAIS:")
for v in vehicles:
    try:
        pose = client.simGetVehiclePose(vehicle_name=v)
        x = pose.position.x_val
        y = pose.position.y_val
        z = pose.position.z_val
        print(f"   {v}: x={x:.1f}, y={y:.1f}, z={z:.1f} (altura={-z:.1f}m)")
    except:
        pass

# Captura uma imagem para verificação visual
print("\n📸 Capturando imagem de teste...")

output_dir = Path("test_altura")
output_dir.mkdir(exist_ok=True)

# Imagem RGB
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")

if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Adiciona texto com informações
    cv2.putText(img_bgr, "TESTE: Drones devem estar VOANDO", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img_bgr, "Ego: 15m altura | Drones: 12-18m", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(str(output_dir / "teste_altura.png"), img_bgr)
    print(f"   ✅ Imagem salva em {output_dir}/teste_altura.png")

# Captura com LiDAR para verificar detecção
print("\n📡 Testando fusão LiDAR-câmera...")
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    # Analisa distribuição de alturas
    z_values = points[:, 2]
    z_mean = z_values.mean()
    z_std = z_values.std()
    z_min = z_values.min()
    z_max = z_values.max()

    print(f"\n📊 ANÁLISE LIDAR:")
    print(f"   Total pontos: {len(points)}")
    print(f"   Z médio: {z_mean:.2f} ± {z_std:.2f}")
    print(f"   Z mín/máx: {z_min:.2f} / {z_max:.2f}")

    # Detecta anomalias (possíveis drones)
    anomalies = np.abs(z_values - z_mean) > 2 * z_std
    print(f"   Anomalias detectadas: {anomalies.sum()}")

    if anomalies.sum() > 0:
        print("   ✅ LiDAR detectando objetos no ar (drones)!")
    else:
        print("   ⚠️  Nenhuma anomalia - verificar se drones estão visíveis")

# Pousa
print("\n🛬 Pousando todos...")
for v in vehicles:
    try:
        client.moveToPositionAsync(0, 0, -3, 3, vehicle_name=v).join()
    except:
        pass

for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("✅ TESTE COMPLETO!")
print("="*70)
print("\n🔍 VERIFICAR:")
print("   1. Posições Z devem ser NEGATIVAS (ex: -15)")
print("   2. Alturas devem estar entre 10-20m")
print("   3. Imagem deve mostrar drones no AR")
print("   4. LiDAR deve detectar anomalias (drones)")
print("="*70)