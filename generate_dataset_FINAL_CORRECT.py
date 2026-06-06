#!/usr/bin/env python3
"""
VERSÃO FINAL CORRETA - Dataset com Fusão LiDAR-Câmera
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR FINAL DEFINITIVO - CORREÇÕES APLICADAS!")
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
output_dir = Path("dataset_FINAL")
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
        "ego_height": -12,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -12},
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -10},
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -14}
        ]
    },
    {
        "name": "medium_altitude",
        "ego_height": -18,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 12, "y": -5, "z": -18},
            {"vehicle": "Drone4", "x": 18, "y": 5, "z": -16},
            {"vehicle": "Intruder1", "x": 25, "y": 0, "z": -20}
        ]
    }
]

total_frames = 0
max_frames = 20

print(f"\n📊 Gerando {max_frames} frames de teste...")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        if cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(cfg["x"], cfg["y"], cfg["z"], 5,
                                      vehicle_name=cfg["vehicle"]).join()

    time.sleep(2)

    for frame_idx in range(min(10, max_frames - total_frames)):
        frame_name = f"frame_{total_frames:04d}"
        print(f"  Frame {total_frames}...")

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
            # Pontos já estão em coordenadas LOCAIS do sensor!
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            X = points[:, 0]
            Y = points[:, 1]
            Z = points[:, 2]

            # Calcula distâncias
            distances = np.sqrt(X**2 + Y**2 + Z**2)

            # Filtra pontos frontais e próximos
            MAX_RANGE = 80  # Ignora montanhas distantes
            front_mask = (X > 0.5) & (distances < MAX_RANGE)

            X_front = X[front_mask]
            Y_front = Y[front_mask]
            Z_front = Z[front_mask]
            dist_front = distances[front_mask]

            if len(X_front) > 0:
                # SEM rotação de pitch (teste)
                # Projeta diretamente
                u = (fx * Y_front / X_front + cx).astype(int)
                v = (fy * (-Z_front) / X_front + cy).astype(int)  # Inverte Z para tela

                # Filtra válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                z_valid = Z_front[valid]
                dist_valid = dist_front[valid]

                # Desenha pontos com lógica correta
                for i in range(len(u_valid)):
                    z = z_valid[i]
                    dist = dist_valid[i]

                    # LÓGICA DE CORES CORRIGIDA:
                    if dist < 30:  # Apenas objetos próximos
                        if -2 < z < 2:  # Nível do sensor
                            # Possível drone - magenta
                            color = (255, 0, 255)
                            size = 4
                            cv2.circle(fusion, (u_valid[i], v_valid[i]), 8, (255, 255, 0), 2)
                        elif z < -8:  # Bem abaixo (chão)
                            color = (0, 255, 0)
                            size = 1
                        else:  # Outros objetos próximos
                            color = (0, 200, 200)
                            size = 2
                    else:  # Objetos distantes
                        if z < -5:  # Terreno distante
                            intensity = int(100 * (1 - dist/MAX_RANGE))
                            color = (0, intensity, 0)
                            size = 1
                        else:  # Horizonte/montanhas
                            # NÃO pinta de magenta!
                            intensity = int(150 * (1 - dist/MAX_RANGE))
                            color = (intensity, intensity, intensity)  # Cinza
                            size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Info
            cv2.putText(fusion, f"{scenario['name']} | Frame {total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, "Magenta=Drone(prox), Verde=Chao, Cinza=Horizonte",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Salva
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points)

            # Metadados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "ego_height": -ego_height,
                "total_points": len(points),
                "front_points": len(X_front),
                "max_range_filter": MAX_RANGE
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

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
print("✅ DATASET FINALIZADO COM SUCESSO!")
print("="*70)
print("\n📌 CORREÇÕES APLICADAS:")
print("  • LiDAR já retorna coordenadas LOCAIS (não converter!)")
print("  • Magenta apenas para objetos PRÓXIMOS em Z≈0 (drones)")
print("  • Horizonte em CINZA (não magenta)")
print("  • Filtro de distância máxima (80m)")
print(f"\n📁 Dataset em: {output_dir.absolute()}")