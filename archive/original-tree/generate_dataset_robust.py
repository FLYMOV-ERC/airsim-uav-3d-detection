#!/usr/bin/env python3
"""
Gerador de Dataset Robusto - 300 imagens com salvamento incremental
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR DE DATASET ROBUSTO - 300 IMAGENS")
print("="*70)

# Conecta com timeout
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"\n✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro na conexão: {e}")
    sys.exit(1)

# Prepara drones
print("🚀 Preparando drones...")
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        print(f"   ⚠️ Falha ao preparar {v}")

# Decola
print("🛫 Decolando...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v)
    except:
        pass
time.sleep(5)

# Cria diretórios
output_dir = Path("dataset_300_robust")
output_dir.mkdir(exist_ok=True)
(output_dir / "rgb").mkdir(exist_ok=True)
(output_dir / "fusion").mkdir(exist_ok=True)
(output_dir / "meta").mkdir(exist_ok=True)

# Verifica frames já existentes
existing_frames = list((output_dir / "rgb").glob("*.png"))
start_frame = len(existing_frames)
print(f"\n📊 Frames já existentes: {start_frame}")

# Parâmetros fixos
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360
pitch_rad = np.radians(-15)

# Loop principal simplificado
frame_count = start_frame
target_frames = 300
batch_size = 10  # Salva status a cada 10 frames

print(f"\n🎯 Meta: {target_frames} frames (começando do frame {start_frame})")

while frame_count < target_frames:
    try:
        # Varia posições aleatoriamente para diversidade
        batch_start = frame_count

        # Nova posição aleatória a cada batch
        ego_height = np.random.randint(-25, -10)
        client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

        # Posiciona outros drones aleatoriamente
        for idx, v in enumerate(vehicles):
            if v != "Ego":
                x = np.random.randint(5, 20)
                y = np.random.randint(-10, 10)
                z = ego_height + np.random.randint(-5, 5)
                try:
                    client.moveToPositionAsync(x, y, z, 5, vehicle_name=v).join()
                except:
                    pass

        time.sleep(2)

        # Captura batch de frames
        for i in range(min(batch_size, target_frames - frame_count)):
            # Varia ângulo
            yaw = np.random.randint(-30, 30)
            client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
            time.sleep(0.3)

            # Captura RGB
            responses = client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ], vehicle_name="Ego")

            img_bgr = None
            if responses and responses[0].image_data_uint8:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(720, 1280, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Captura LiDAR
            lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

            if lidar_data and img_bgr is not None:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

                # Processa fusão
                fusion = img_bgr.copy()
                front = points[points[:, 0] > 0.5]
                detections = 0

                if len(front) > 0:
                    X, Y, Z = front[:, 0], front[:, 1], front[:, 2]

                    # Detecta anomalias
                    z_mean = Z.mean()
                    anomalies = np.abs(Z - z_mean) > 2 * Z.std()

                    # Projeta com correção de pitch
                    X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
                    Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

                    u = (fx * Y / X_rot + cx).astype(int)
                    v = (fy * Z_rot / X_rot + cy).astype(int)

                    valid = (u >= 0) & (u < 1280) & (v >= 0) & (v < 720)

                    # Desenha
                    for j in np.where(valid)[0]:
                        if anomalies[j]:
                            cv2.circle(fusion, (u[j], v[j]), 4, (255, 0, 255), -1)
                            cv2.circle(fusion, (u[j], v[j]), 7, (255, 255, 0), 2)
                            detections += 1
                        else:
                            cv2.circle(fusion, (u[j], v[j]), 1, (0, 200, 0), -1)

                # Info
                cv2.putText(fusion, f"Frame {frame_count} | Det: {detections}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Salva
                frame_name = f"frame_{frame_count:04d}"
                cv2.imwrite(str(output_dir / "rgb" / f"{frame_name}.png"), img_bgr)
                cv2.imwrite(str(output_dir / "fusion" / f"{frame_name}.png"), fusion)

                # Metadata
                meta = {
                    "frame": frame_count,
                    "yaw": int(yaw),
                    "height": int(-ego_height),
                    "points": len(points),
                    "detections": detections
                }
                with open(output_dir / "meta" / f"{frame_name}.json", 'w') as f:
                    json.dump(meta, f)

                frame_count += 1

                # Print progress
                if frame_count % 10 == 0:
                    print(f"   📊 Progresso: {frame_count}/{target_frames} frames")

        # Salva progresso
        print(f"✅ Batch completo: frames {batch_start} a {frame_count-1}")

    except Exception as e:
        print(f"⚠️ Erro no frame {frame_count}: {e}")
        print("   Tentando reconectar...")

        # Tenta reconectar
        try:
            client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
            client.confirmConnection()
            print("   ✅ Reconectado!")
        except:
            print("   ❌ Falha na reconexão. Salvando progresso...")
            break

# Pousa
print("\n🛬 Finalizando...")
try:
    for v in vehicles:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
except:
    pass

# Estatísticas
print("\n" + "="*70)
print("📊 RESULTADOS:")
print("="*70)
print(f"  • Frames gerados: {frame_count}")
print(f"  • Dataset em: {output_dir.absolute()}")

if frame_count >= target_frames:
    print("\n🎉 DATASET COMPLETO COM 300 IMAGENS!")
else:
    print(f"\n⚠️ Dataset parcial: {frame_count}/{target_frames} frames")
    print("   Execute novamente para continuar de onde parou.")