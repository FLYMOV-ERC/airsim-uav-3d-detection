#!/usr/bin/env python3
"""
VERSÃO FUNCIONANDO - Dataset com Z Invertido Corrigido
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR DEFINITIVO - Z INVERTIDO CORRIGIDO!")
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
print("\n🛫 Decolando todos os drones...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Diretórios
output_dir = Path("dataset_working")
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

# Cenários variados
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
    },
    {
        "name": "high_altitude",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z": -22},
            {"vehicle": "Drone4", "x": 20, "y": 8, "z": -20},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -18}
        ]
    }
]

yaw_angles = [-20, -10, 0, 10, 20]
total_frames = 0
max_frames = 300

print(f"\n📊 Gerando até {max_frames} frames...")
print("   Z INVERTIDO: Agora o chão aparecerá corretamente!")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Ego: altura {-ego_height}m")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        if cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(cfg["x"], cfg["y"], cfg["z"], 5,
                                      vehicle_name=cfg["vehicle"]).join()
            print(f"   {cfg['vehicle']}: x={cfg['x']}, y={cfg['y']}, altura={-cfg['z']}m")

    time.sleep(2)

    for yaw in yaw_angles:
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # Limpa buffer LiDAR
        _ = client.getLidarData("LidarFront", vehicle_name="Ego")
        time.sleep(0.1)

        # Captura dados
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
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            X = points[:, 0]
            Y = points[:, 1]
            # ⚠️ CORREÇÃO CRÍTICA: Inverter Z!
            Z = -points[:, 2]  # Z INVERTIDO!

            # Calcula distâncias
            distances = np.sqrt(X**2 + Y**2 + points[:, 2]**2)

            # Filtra pontos frontais e próximos
            MAX_RANGE = 100
            front_mask = (X > 0.5) & (distances < MAX_RANGE)

            X_front = X[front_mask]
            Y_front = Y[front_mask]
            Z_front = Z[front_mask]
            dist_front = distances[front_mask]

            detections = 0
            detected_drones = []

            if len(X_front) > 0:
                # Projeta com Z invertido
                u = (fx * Y_front / X_front + cx).astype(int)
                v = (fy * Z_front / X_front + cy).astype(int)

                # Filtra válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                z_valid = Z_front[valid]
                x_valid = X_front[valid]
                y_valid = Y_front[valid]
                dist_valid = dist_front[valid]

                # Desenha pontos
                for i in range(len(u_valid)):
                    z = z_valid[i]
                    dist = dist_valid[i]
                    x = x_valid[i]
                    y = y_valid[i]

                    # Lógica de cores
                    if z < -8:  # Chão (bem abaixo do sensor)
                        color = (0, 200, 0)  # Verde
                        size = 1
                    elif dist < 40 and -3 < z < 3:  # Possível drone próximo
                        color = (255, 0, 255)  # Magenta
                        size = 4
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (255, 255, 0), 2)
                        detections += 1
                        if detections <= 5:
                            detected_drones.append({
                                "x": float(x), "y": float(y), "z": float(z),
                                "distance": float(dist)
                            })
                    elif z > 5:  # Acima do sensor
                        color = (0, 100, 255)  # Laranja
                        size = 2
                    else:  # Intermediário
                        intensity = int(150 * (1 - dist/MAX_RANGE))
                        color = (intensity, intensity, intensity)  # Cinza
                        size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Informações na imagem
            cv2.putText(fusion, f"{scenario['name']} | Alt:{-ego_height}m | Yaw:{yaw} | F:{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Detecções: {detections} | Verde=Chão, Magenta=Drone",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(fusion, "Z INVERTIDO APLICADO",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Salva arquivos
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points)

            # Metadados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "scenario_idx": scenario_idx,
                "yaw": yaw,
                "ego_height": -ego_height,
                "total_points": len(points),
                "front_points": len(X_front),
                "detections": detections,
                "detected_drones": detected_drones,
                "z_inverted": True,
                "max_range": MAX_RANGE
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

            # Status
            if total_frames % 10 == 0:
                print(f"   📊 Progresso: {total_frames}/{max_frames} frames")

    if total_frames >= max_frames:
        break

# Pousa drones
print("\n🛬 Pousando...")
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
print("🎉 DATASET COMPLETO E FUNCIONANDO!")
print("="*70)
print(f"\n📊 Total de frames: {total_frames}")
print("\n✅ CORREÇÃO APLICADA:")
print("   • Z INVERTIDO (-Z) para projeção correta")
print("   • Chão agora aparece como verde")
print("   • Drones detectados em magenta")
print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")