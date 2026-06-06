#!/usr/bin/env python3
"""
VERSÃO CORRETA FINAL - Transformação de Coordenadas Globais para Locais
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR CORRETO - TRANSFORMAÇÃO GLOBAL→LOCAL!")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\n🛫 Decolando...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Diretórios
output_dir = Path("dataset_correto")
output_dir.mkdir(exist_ok=True)
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "lidar_points").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset em: {output_dir.absolute()}")

# Parâmetros da câmera
FOV_H = 90
FOV_V = 60
fx = 640 / np.tan(np.radians(FOV_H/2))
fy = 360 / np.tan(np.radians(FOV_V/2))
cx = 640
cy = 360

# Cenários
scenarios = [
    {
        "name": "low_altitude",
        "ego_height": -10,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -10},
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -8},
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -12}
        ]
    },
    {
        "name": "medium_altitude",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 12, "y": -5, "z": -15},
            {"vehicle": "Drone4", "x": 18, "y": 5, "z": -13},
            {"vehicle": "Intruder1", "x": 25, "y": 0, "z": -17}
        ]
    }
]

total_frames = 0
max_frames = 50

print(f"\n📊 Gerando {max_frames} frames de teste...")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Ego: altura {-ego_height}m")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        if cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(cfg["x"], cfg["y"], cfg["z"], 5,
                                      vehicle_name=cfg["vehicle"]).join()

    time.sleep(2)

    for frame_idx in range(min(25, max_frames - total_frames)):
        if total_frames >= max_frames:
            break

        frame_name = f"frame_{total_frames:04d}"

        # Pega posição do drone
        pose = client.simGetVehiclePose(vehicle_name="Ego")
        drone_z_global = pose.position.z_val  # Z global do drone (negativo)
        drone_altitude = -drone_z_global  # Altura real

        # Limpa buffer
        _ = client.getLidarData("LidarFront", vehicle_name="Ego")
        time.sleep(0.1)

        # Captura
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

        # Processa imagem
        img_bgr = None
        if responses and responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
            points_raw = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            # TRANSFORMAÇÃO CRÍTICA: Global → Local
            # Os pontos estão em coordenadas GLOBAIS onde Z=0 é o solo
            # Precisamos converter para coordenadas LOCAIS do sensor
            X = points_raw[:, 0]
            Y = points_raw[:, 1]
            Z_global = points_raw[:, 2]

            # Converte Z global para local
            Z_local = Z_global - drone_altitude  # Subtrai altura do drone

            # Calcula distâncias
            distances = np.sqrt(X**2 + Y**2 + Z_local**2)

            # Filtra pontos frontais e próximos
            MAX_RANGE = 80
            front_mask = (X > 0.5) & (distances < MAX_RANGE)

            X_front = X[front_mask]
            Y_front = Y[front_mask]
            Z_front = Z_local[front_mask]
            dist_front = distances[front_mask]

            if len(X_front) > 0:
                # Projeta para imagem
                # Nota: Y é lateral, Z é vertical
                u = (fx * Y_front / X_front + cx).astype(int)
                v = (fy * (-Z_front) / X_front + cy).astype(int)  # -Z para tela

                # Filtra válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                z_valid = Z_front[valid]
                dist_valid = dist_front[valid]

                # Desenha pontos com cores corretas
                for i in range(len(u_valid)):
                    z = z_valid[i]
                    dist = dist_valid[i]

                    if z < -5:  # Chão (5m+ abaixo do sensor)
                        color = (0, 255, 0)  # Verde
                        size = 1
                    elif dist < 30 and -2 < z < 2:  # Possível drone
                        color = (255, 0, 255)  # Magenta
                        size = 4
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 8, (255, 255, 0), 2)
                    elif z > 3:  # Acima do sensor
                        color = (0, 100, 255)  # Laranja
                        size = 2
                    else:  # Intermediário
                        intensity = int(150 * (1 - dist/MAX_RANGE))
                        color = (intensity, intensity, intensity)
                        size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Info
            cv2.putText(fusion, f"Frame {total_frames} | Alt: {drone_altitude:.1f}m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, "Verde=Chao, Magenta=Drone, Laranja=Acima",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(fusion, "COORD: Global->Local OK",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Salva
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points_raw)

            # Metadados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "drone_altitude": float(drone_altitude),
                "drone_z_global": float(drone_z_global),
                "total_points": len(points_raw),
                "coordinate_transform": "global_to_local",
                "z_transform": f"Z_local = Z_global - {drone_altitude:.2f}"
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

            if total_frames % 10 == 0:
                print(f"   Progresso: {total_frames}/{max_frames}")

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("✅ DATASET CORRETO GERADO!")
print("="*70)
print("\n📌 Transformação aplicada:")
print("   Z_local = Z_global - altura_do_drone")
print("   Isso converte coordenadas globais (Z=0 é o solo)")
print("   para coordenadas locais (Z=0 é o sensor)")
print(f"\n📁 Dataset em: {output_dir.absolute()}")