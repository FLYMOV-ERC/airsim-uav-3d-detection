#!/usr/bin/env python3
"""
Gerador de Dataset com Sincronização Real de LiDAR
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR COM SINCRONIZAÇÃO REAL DE LIDAR!")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
    sys.exit(1)

# Prepara todos os drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola todos os drones
print("\n🛫 Decolando todos os drones...")
takeoff_tasks = []
for v in vehicles:
    try:
        task = client.takeoffAsync(vehicle_name=v)
        takeoff_tasks.append(task)
    except:
        pass

for task in takeoff_tasks:
    try:
        task.join()
    except:
        pass

print("✅ Todos os drones decolaram!")
time.sleep(3)

# Cria diretório
output_dir = Path("dataset_lidar_sync")
output_dir.mkdir(exist_ok=True)

(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "lidar_points").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset em: {output_dir.absolute()}")

# Parâmetros da câmera - AJUSTADOS para FOV correto
FOV_H = 90  # graus
FOV_V = 60  # estimado para 16:9

fx = 640 / np.tan(np.radians(FOV_H/2))
fy = 360 / np.tan(np.radians(FOV_V/2))
cx = 640
cy = 360

# Pitch da câmera (do settings.json)
camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

print(f"\n⚙️  Configurações:")
print(f"   • FOV: {FOV_H}° horizontal, {FOV_V}° vertical")
print(f"   • Pitch da câmera: {camera_pitch}°")
print(f"   • Método: Aguardar rotação completa do LiDAR")

# Cenários de teste
scenarios = [
    {
        "name": "test_sync",
        "ego_height": -12,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -12},
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -14},
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -10}
        ]
    }
]

total_frames = 0
max_frames = 10

print(f"\n📊 Gerando {max_frames} frames de teste...")

for scenario in scenarios:
    print(f"\n🎬 Cenário: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Ego: altura {-ego_height}m")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        vehicle_name = cfg["vehicle"]
        if vehicle_name in vehicles:
            x, y, z = cfg["x"], cfg["y"], cfg["z"]
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle_name).join()
            print(f"   {vehicle_name}: x={x}, y={y}, altura={-z}m")

    time.sleep(3)  # Estabilização completa

    for frame_idx in range(max_frames):
        frame_name = f"frame_{total_frames:04d}"
        print(f"\n   Frame {total_frames + 1}/{max_frames}")

        # === MÉTODO 1: Limpar buffer do LiDAR ===
        # Descarta dados antigos
        _ = client.getLidarData("LidarFront", vehicle_name="Ego")
        time.sleep(0.1)  # Aguarda 100ms para nova varredura

        # === CAPTURA SINCRONIZADA ===
        t0 = time.time()

        # Captura imagem
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        # Captura LiDAR fresco
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

        capture_time = (time.time() - t0) * 1000

        # Processa imagem
        img_bgr = None
        if responses and responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            # Analisa timestamp do LiDAR
            lidar_timestamp = lidar_data.time_stamp
            print(f"      LiDAR timestamp: {lidar_timestamp}")
            print(f"      Pontos capturados: {len(points)}")
            print(f"      Tempo de captura: {capture_time:.1f}ms")

            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            # Filtra pontos frontais
            front_mask = points[:, 0] > 0.5
            front_points = points[front_mask]

            if len(front_points) > 0:
                X = front_points[:, 0]
                Y = front_points[:, 1]
                Z = front_points[:, 2]

                # Detecta anomalias (drones)
                z_mean = Z.mean()
                z_std = Z.std()
                anomalies_mask = np.abs(Z - z_mean) > 1.5

                # Correção: Aplica pitch tanto na câmera quanto no LiDAR
                # já que ambos têm o mesmo pitch (-15°)
                X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
                Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

                # Projeta para imagem
                u = (fx * Y / X_rot + cx).astype(int)
                v = (fy * Z_rot / X_rot + cy).astype(int)

                # Filtra pontos válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (X_rot > 0)

                u_valid = u[valid]
                v_valid = v[valid]
                x_valid = X[valid]
                anomalies_valid = anomalies_mask[valid]

                detections = 0

                # Desenha pontos
                for i in range(len(u_valid)):
                    if anomalies_valid[i]:
                        # Drone detectado
                        color = (255, 0, 255)
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 5, color, -1)
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (255, 255, 0), 2)
                        detections += 1
                    else:
                        # Ponto normal
                        depth = min(1.0, x_valid[i] / 50)
                        color = (0, int(255*(1-depth)), 0)
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, color, -1)

                print(f"      Detecções: {detections}")

            # Info na imagem
            cv2.putText(fusion, f"Frame {total_frames} | Alt: {-ego_height}m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"LiDAR pts: {len(points)} | Front: {len(front_points)}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(fusion, f"Sync time: {capture_time:.1f}ms",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Salva imagens
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            # Salva pontos
            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points)

            # Metadados
            metadata = {
                "frame_id": total_frames,
                "lidar_timestamp": lidar_timestamp,
                "capture_time_ms": capture_time,
                "total_points": len(points),
                "front_points": len(front_points)
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

        # Pequeno delay entre frames
        time.sleep(0.5)

# Pousa drones
print("\n🛬 Pousando drones...")
for v in vehicles:
    try:
        client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=v).join()
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
print(f"📁 Dataset em: {output_dir.absolute()}")
print("\n💡 IMPORTANTE:")
print("   1. Copie settings_fixed.json para o local correto do AirSim")
print("   2. O LiDAR deve ter o mesmo pitch que a câmera")
print("   3. Verifique se os pontos estão alinhados nas imagens de fusão")