#!/usr/bin/env python3
"""
Gerador de Dataset com Compensação Manual de Delay
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR DE DATASET - COMPENSAÇÃO DE DELAY!")
print("="*70)

# Tenta conectar com timeout
print("\n🔌 Tentando conectar ao AirSim...")
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
    print("\n⚠️  Certifique-se que o AirSim está rodando!")
    print("    Execute o simulador Unreal Engine primeiro.")
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

# Cria diretório para o dataset
output_dir = Path("dataset_sync_compensated")
output_dir.mkdir(exist_ok=True)

# Subdiretórios
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "lidar_points").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset será salvo em: {output_dir.absolute()}")

# Parâmetros da câmera
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360
camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

# COMPENSAÇÃO DE DELAY ESTIMADA (em pixels)
# Ajuste este valor se os pontos ainda estiverem desalinhados
DELAY_COMPENSATION_X = 0  # pixels horizontais
DELAY_COMPENSATION_Y = 0  # pixels verticais

print(f"\n⚙️  Configurações de compensação:")
print(f"   • Compensação X: {DELAY_COMPENSATION_X} pixels")
print(f"   • Compensação Y: {DELAY_COMPENSATION_Y} pixels")

# Cenários para teste
scenarios = [
    {
        "name": "low_altitude",
        "ego_height": -10,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 5, "y": -2, "z": -10},
            {"vehicle": "Drone4", "x": 7, "y": 2, "z": -12},
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z": -8}
        ]
    },
    {
        "name": "medium_altitude",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": -5, "z": -18},
            {"vehicle": "Drone4", "x": 15, "y": 5, "z": -15},
            {"vehicle": "Intruder1", "x": 20, "y": 0, "z": -12}
        ]
    }
]

yaw_angles = [-10, 0, 10]
total_frames = 0
total_detections = 0
start_time = time.time()
max_frames = 20  # Limitado para teste

print(f"\n📊 Config: {len(scenarios)} cenários, {len(yaw_angles)} ângulos, máx {max_frames} frames")

# Gera o dataset
for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Movendo Ego para {-ego_height}m de altura...")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for drone_cfg in scenario["drone_configs"]:
        vehicle_name = drone_cfg["vehicle"]
        if vehicle_name in vehicles:
            x, y, z = drone_cfg["x"], drone_cfg["y"], drone_cfg["z"]
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle_name).join()
            print(f"   {vehicle_name}: x={x}m, y={y}m, altura={-z}m")

    # Aguarda estabilização
    time.sleep(2)

    for yaw_idx, yaw in enumerate(yaw_angles):
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.8)  # Aguarda rotação estabilizar

        frame_name = f"frame_{total_frames:04d}"

        # === CAPTURA SINCRONIZADA ===
        # Captura RGB e LiDAR com mínimo delay possível
        t0 = time.time()

        # Captura simultânea
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

        capture_delay = (time.time() - t0) * 1000  # em ms

        img_bgr = None
        if responses and responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            # Filtra pontos frontais
            front_mask = points[:, 0] > 0.5
            front_points = points[front_mask]

            frame_detections = 0

            if len(front_points) > 0:
                X = front_points[:, 0]
                Y = front_points[:, 1]
                Z = front_points[:, 2]

                # Detecta anomalias
                z_mean = Z.mean()
                z_std = Z.std()
                anomalies_mask = np.abs(Z - z_mean) > 2

                # Aplica correção de pitch
                X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
                Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

                # Projeta para pixels
                u = (fx * Y / X_rot + cx).astype(int)
                v = (fy * Z_rot / X_rot + cy).astype(int)

                # === APLICA COMPENSAÇÃO DE DELAY ===
                u = u + DELAY_COMPENSATION_X
                v = v + DELAY_COMPENSATION_Y

                # Filtra válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                x_valid = X[valid]
                anomalies_valid = anomalies_mask[valid]

                # Desenha pontos
                for i in range(len(u_valid)):
                    if anomalies_valid[i]:
                        # Drone detectado
                        color = (255, 0, 255)
                        size = 5
                        frame_detections += 1
                        # Círculo maior ao redor
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (255, 255, 0), 2)
                    else:
                        # Normal
                        depth = min(1.0, x_valid[i] / 50)
                        color = (0, int(255*(1-depth)), 0)
                        size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Info na imagem
            info_text = f"{scenario['name']} | Alt:{-ego_height}m | Yaw:{yaw}°"
            cv2.putText(fusion, info_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Detections: {frame_detections}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, f"Delay: {capture_delay:.1f}ms | Comp: X={DELAY_COMPENSATION_X} Y={DELAY_COMPENSATION_Y}",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Salva arquivos
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            # Comparação
            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            # Salva pontos
            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points)

            # Metadados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "yaw": yaw,
                "ego_height": ego_height,
                "total_points": len(points),
                "front_points": len(front_points),
                "detections": frame_detections,
                "capture_delay_ms": capture_delay,
                "compensation_x": DELAY_COMPENSATION_X,
                "compensation_y": DELAY_COMPENSATION_Y,
                "drone_positions": scenario["drone_configs"]
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_detections += frame_detections
            total_frames += 1

            print(f"   Frame {total_frames:03d}: {frame_detections} detecções, delay {capture_delay:.1f}ms")

    if total_frames >= max_frames:
        break

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

elapsed = time.time() - start_time

print("\n" + "="*70)
print("✅ DATASET GERADO COM SUCESSO!")
print("="*70)
print(f"\n📊 Estatísticas:")
print(f"   • Frames: {total_frames}")
print(f"   • Detecções: {total_detections}")
print(f"   • Tempo: {elapsed:.1f}s")
print(f"   • FPS médio: {total_frames/elapsed:.2f}")
print(f"\n📁 Dataset em: {output_dir.absolute()}")
print("\n💡 Dica: Se os pontos ainda estiverem desalinhados,")
print("   ajuste DELAY_COMPENSATION_X e DELAY_COMPENSATION_Y no código.")