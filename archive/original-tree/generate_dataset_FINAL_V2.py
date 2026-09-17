#!/usr/bin/env python3
"""
VERSÃO FINAL V2 - Com Detecção de Drones Configurados
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys
from sklearn.cluster import DBSCAN

print("\n" + "="*70)
print("🚀 GERADOR FINAL V2 - DRONES COM TAGS LIDAR!")
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
output_dir = Path("dataset_final_v2")
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
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -10},    # mesma altura
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -9},     # 1m acima
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -11}  # 1m abaixo
        ]
    },
    {
        "name": "medium_altitude",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 12, "y": -5, "z": -15},
            {"vehicle": "Drone4", "x": 18, "y": 5, "z": -14},
            {"vehicle": "Intruder1", "x": 25, "y": 0, "z": -16}
        ]
    },
    {
        "name": "high_altitude",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z": -20},
            {"vehicle": "Drone4", "x": 20, "y": 8, "z": -19},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -21}
        ]
    }
]

yaw_angles = [-20, -10, 0, 10, 20]
total_frames = 0
total_detections = 0
max_frames = 300

print(f"\n📊 Gerando até {max_frames} frames...")
print("   ✅ Drones agora com tags Lidar!")

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

        # Rotaciona
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # Pega posição do Ego
        pose = client.simGetVehiclePose(vehicle_name="Ego")
        drone_z_global = pose.position.z_val
        drone_altitude = -drone_z_global

        # Limpa buffer LiDAR
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

            # Coordenadas
            X = points_raw[:, 0]
            Y = points_raw[:, 1]
            Z_global = points_raw[:, 2]

            # Converte para local
            Z_local = Z_global - drone_altitude

            # Filtra pontos frontais
            MAX_RANGE = 100
            distances = np.sqrt(X**2 + Y**2 + Z_local**2)
            front_mask = (X > 0.5) & (distances < MAX_RANGE)

            X_front = X[front_mask]
            Y_front = Y[front_mask]
            Z_front = Z_local[front_mask]
            dist_front = distances[front_mask]

            frame_detections = 0
            detected_clusters = []

            if len(X_front) > 0:
                # DETECÇÃO DE DRONES MELHORADA
                # Procura clusters de pontos "flutuando" (não no chão)
                floating_mask = (Z_front > -8) & (Z_front < 5) & (dist_front < 40)

                if np.sum(floating_mask) > 10:  # Mínimo de pontos
                    floating_points = np.column_stack([
                        X_front[floating_mask],
                        Y_front[floating_mask],
                        Z_front[floating_mask]
                    ])

                    # Clustering para separar drones
                    try:
                        clustering = DBSCAN(eps=2.0, min_samples=5).fit(floating_points)
                        labels = clustering.labels_

                        # Cada cluster é um possível drone
                        for label in set(labels):
                            if label != -1:  # -1 é ruído
                                cluster_mask = labels == label
                                cluster_x = floating_points[cluster_mask, 0].mean()
                                cluster_y = floating_points[cluster_mask, 1].mean()
                                cluster_z = floating_points[cluster_mask, 2].mean()
                                cluster_dist = np.sqrt(cluster_x**2 + cluster_y**2 + cluster_z**2)

                                detected_clusters.append({
                                    "x": float(cluster_x),
                                    "y": float(cluster_y),
                                    "z": float(cluster_z),
                                    "distance": float(cluster_dist),
                                    "points": int(np.sum(cluster_mask))
                                })
                                frame_detections += 1
                    except:
                        pass

                # Projeta todos os pontos
                u = (fx * Y_front / X_front + cx).astype(int)
                v = (fy * (-Z_front) / X_front + cy).astype(int)

                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                z_valid = Z_front[valid]
                dist_valid = dist_front[valid]

                # Desenha pontos
                for i in range(len(u_valid)):
                    z = z_valid[i]
                    dist = dist_valid[i]

                    if z < -8:  # Chão
                        color = (0, 200, 0)  # Verde
                        size = 1
                    elif -8 <= z < -2:  # Transição
                        color = (0, 150, 150)  # Ciano
                        size = 1
                    elif -2 <= z <= 2 and dist < 40:  # Nível do sensor, próximo
                        color = (255, 0, 255)  # Magenta - possível drone
                        size = 3
                    elif z > 2:  # Acima
                        color = (0, 100, 255)  # Laranja
                        size = 2
                    else:  # Outros
                        intensity = int(100 * (1 - dist/MAX_RANGE))
                        color = (intensity, intensity, intensity)
                        size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

                # Marca clusters detectados
                for cluster in detected_clusters:
                    # Projeta centro do cluster
                    if cluster["x"] > 0:
                        u_cluster = int(fx * cluster["y"] / cluster["x"] + cx)
                        v_cluster = int(fy * (-cluster["z"]) / cluster["x"] + cy)

                        if 0 <= u_cluster < w and 0 <= v_cluster < h:
                            # Desenha caixa ao redor
                            cv2.rectangle(fusion,
                                        (u_cluster-20, v_cluster-20),
                                        (u_cluster+20, v_cluster+20),
                                        (255, 255, 0), 2)
                            cv2.putText(fusion, f"D:{cluster['distance']:.1f}m",
                                      (u_cluster-20, v_cluster-30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Informações
            cv2.putText(fusion, f"{scenario['name']} | Alt:{drone_altitude:.1f}m | Yaw:{yaw}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Drones detectados: {frame_detections}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            if frame_detections > 0:
                cv2.putText(fusion, f"✓ DETECÇÃO OK!",
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                "yaw": yaw,
                "ego_altitude": float(drone_altitude),
                "total_points": len(points_raw),
                "detections": frame_detections,
                "detected_drones": detected_clusters,
                "drone_positions_expected": scenario["drone_configs"]
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1
            total_detections += frame_detections

            if total_frames % 10 == 0:
                print(f"   📊 Progresso: {total_frames}/{max_frames} | Detecções: {total_detections}")

    if total_frames >= max_frames:
        break

# Pousa
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
print("🎉 DATASET COMPLETO!")
print("="*70)
print(f"\n📊 Estatísticas:")
print(f"   • Frames: {total_frames}")
print(f"   • Detecções totais: {total_detections}")
print(f"   • Média: {total_detections/total_frames:.2f} drones/frame" if total_frames > 0 else "")
print(f"\n✅ Com tags Lidar no Unreal, os drones agora são detectados!")
print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")