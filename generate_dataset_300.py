#!/usr/bin/env python3
"""
Gerador de Dataset Grande - 300 imagens com fusão LiDAR-câmera funcionando
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json

print("\n" + "="*70)
print("🚀 GERADOR DE DATASET - 300 IMAGENS COM FUSÃO LIDAR")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n✅ Conectado! Veículos: {vehicles}")

# Prepara todos os drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola todos
print("🛫 Decolando todos os drones...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v)
    except:
        pass
time.sleep(5)

# Cria diretório para o dataset
output_dir = Path("dataset_300_fusion")
output_dir.mkdir(exist_ok=True)

# Subdiretórios organizados
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "lidar_points").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset será salvo em: {output_dir.absolute()}")

# Parâmetros da câmera (constantes)
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360
camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

# Configurações de cenários variados
scenarios = [
    # Formação próxima
    {
        "name": "close_formation",
        "ego_height": -10,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 5, "y": -2, "z_offset": 0},
            {"vehicle": "Drone4", "x": 7, "y": 2, "z_offset": 0},
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z_offset": 2}
        ]
    },
    # Formação média
    {
        "name": "medium_formation",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": -5, "z_offset": -3},
            {"vehicle": "Drone4", "x": 15, "y": 5, "z_offset": 0},
            {"vehicle": "Intruder1", "x": 20, "y": 0, "z_offset": 3}
        ]
    },
    # Formação distante
    {
        "name": "far_formation",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z_offset": -5},
            {"vehicle": "Drone4", "x": 25, "y": 8, "z_offset": 0},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z_offset": 5}
        ]
    },
    # Formação vertical
    {
        "name": "vertical_formation",
        "ego_height": -25,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": 0, "z_offset": -10},
            {"vehicle": "Drone4", "x": 12, "y": -2, "z_offset": 0},
            {"vehicle": "Intruder1", "x": 14, "y": 2, "z_offset": 10}
        ]
    },
    # Formação lateral
    {
        "name": "lateral_formation",
        "ego_height": -18,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -10, "z_offset": 0},
            {"vehicle": "Drone4", "x": 10, "y": 0, "z_offset": -2},
            {"vehicle": "Intruder1", "x": 12, "y": 10, "z_offset": 2}
        ]
    }
]

# Ângulos de yaw para variar perspectiva
yaw_angles = [-30, -20, -10, 0, 10, 20, 30]

# Estatísticas globais
total_frames = 0
total_detections = 0
start_time = time.time()

print(f"\n📊 Configuração do dataset:")
print(f"   • {len(scenarios)} cenários diferentes")
print(f"   • {len(yaw_angles)} ângulos por cenário")
print(f"   • Meta: 300 imagens")

# Gera o dataset
for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")
    print("-"*50)

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()
    print(f"   Ego posicionado a {-ego_height}m de altura")

    # Posiciona outros drones
    for drone_cfg in scenario["drone_configs"]:
        vehicle_name = drone_cfg["vehicle"]
        if vehicle_name in vehicles:
            x = drone_cfg["x"]
            y = drone_cfg["y"]
            z = ego_height + drone_cfg["z_offset"]
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle_name).join()
            print(f"   {vehicle_name}: {x}m frente, {y}m lateral, offset {drone_cfg['z_offset']}m")

    time.sleep(2)

    # Captura para cada ângulo
    for yaw_idx, yaw in enumerate(yaw_angles):
        # Para após 300 imagens
        if total_frames >= 300:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # Captura imagem RGB
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Captura LiDAR
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

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

                # IMPORTANTE: Detecta anomalias ANTES de filtrar
                z_mean = Z.mean()
                z_std = Z.std()
                anomalies_mask = np.abs(Z - z_mean) > 2 * z_std

                # Aplica correção de pitch
                X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
                Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

                # Projeta
                u = (fx * Y / X_rot + cx).astype(int)
                v = (fy * Z_rot / X_rot + cy).astype(int)

                # Filtra válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                x_valid = X[valid]
                anomalies_valid = anomalies_mask[valid]

                # Desenha pontos
                for i in range(len(u_valid)):
                    if anomalies_valid[i]:
                        # Drone detectado!
                        color = (255, 0, 255)  # Magenta
                        size = 4
                        frame_detections += 1
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 8, (255, 255, 0), 2)
                    else:
                        # Chão/normal
                        depth = min(1.0, x_valid[i] / 50)
                        color = (0, int(255*(1-depth)), 0)
                        size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Adiciona informações na imagem
            cv2.putText(fusion, f"{scenario['name']} | Frame {total_frames} | Yaw {yaw}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Detections: {frame_detections}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Salva arquivos
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            # Comparação lado a lado
            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            # Salva pontos LiDAR
            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points)

            # Salva metadados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "scenario_idx": scenario_idx,
                "yaw": yaw,
                "ego_height": ego_height,
                "total_points": len(points),
                "front_points": len(front_points),
                "anomalies": int(anomalies_mask.sum()) if len(front_points) > 0 else 0,
                "detections": frame_detections,
                "timestamp": time.time()
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_detections += frame_detections
            total_frames += 1

            # Status a cada 10 frames
            if total_frames % 10 == 0:
                elapsed = time.time() - start_time
                fps = total_frames / elapsed
                eta = (300 - total_frames) / fps if fps > 0 else 0
                print(f"\n   📊 Progresso: {total_frames}/300 frames")
                print(f"      Tempo: {elapsed:.1f}s | FPS: {fps:.2f} | ETA: {eta:.1f}s")
                print(f"      Detecções totais: {total_detections}")

    if total_frames >= 300:
        break

# Volta Ego ao centro
client.rotateToYawAsync(0, vehicle_name="Ego").join()

# Pousa todos os drones
print("\n🛬 Pousando todos os drones...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

# Estatísticas finais
elapsed_total = time.time() - start_time

print("\n" + "="*70)
print("🎉 DATASET COMPLETO!")
print("="*70)
print(f"\n📊 ESTATÍSTICAS FINAIS:")
print(f"   • Frames gerados: {total_frames}")
print(f"   • Detecções totais: {total_detections}")
print(f"   • Média detecções/frame: {total_detections/total_frames:.2f}")
print(f"   • Tempo total: {elapsed_total:.1f} segundos")
print(f"   • FPS médio: {total_frames/elapsed_total:.2f}")
print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")
print("\n📂 Estrutura:")
print("   • images_rgb/: Imagens originais da câmera")
print("   • images_fusion/: Imagens com fusão LiDAR")
print("   • images_comparison/: Comparações lado a lado")
print("   • lidar_points/: Nuvens de pontos (.npy)")
print("   • metadata/: Metadados de cada frame (.json)")
print("\n✅ Dataset pronto para treinamento de modelos de detecção!")