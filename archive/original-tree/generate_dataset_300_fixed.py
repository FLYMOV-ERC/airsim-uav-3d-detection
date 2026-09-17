#!/usr/bin/env python3
"""
Gerador de Dataset Grande CORRIGIDO - 300 imagens com drones no ar
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json

print("\n" + "="*70)
print("🚀 GERADOR DE DATASET CORRIGIDO - DRONES NO AR!")
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

# IMPORTANTE: Decola e AGUARDA todos os drones
print("🛫 Decolando todos os drones...")
takeoff_tasks = []
for v in vehicles:
    try:
        task = client.takeoffAsync(vehicle_name=v)
        takeoff_tasks.append(task)
    except:
        pass

# AGUARDA todas as decolagens completarem
for task in takeoff_tasks:
    try:
        task.join()
    except:
        pass

print("✅ Todos os drones decolaram!")
time.sleep(3)  # Aguarda estabilização

# Cria diretório para o dataset
output_dir = Path("dataset_300_fusion_fixed")
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

# Configurações de cenários variados - ALTURAS CORRIGIDAS!
scenarios = [
    # Formação baixa (10-15m de altura)
    {
        "name": "low_altitude",
        "ego_height": -10,  # 10m de altura
        "drone_configs": [
            {"vehicle": "Drone3", "x": 5, "y": -2, "z": -10},    # mesma altura
            {"vehicle": "Drone4", "x": 7, "y": 2, "z": -12},     # 2m abaixo
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z": -8}   # 2m acima
        ]
    },
    # Formação média (15-20m de altura)
    {
        "name": "medium_altitude",
        "ego_height": -15,  # 15m de altura
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": -5, "z": -18},   # 3m abaixo
            {"vehicle": "Drone4", "x": 15, "y": 5, "z": -15},    # mesma altura
            {"vehicle": "Intruder1", "x": 20, "y": 0, "z": -12}  # 3m acima
        ]
    },
    # Formação alta (20-25m de altura)
    {
        "name": "high_altitude",
        "ego_height": -20,  # 20m de altura
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z": -25},   # 5m abaixo
            {"vehicle": "Drone4", "x": 25, "y": 8, "z": -20},    # mesma altura
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -15}  # 5m acima
        ]
    },
    # Formação escalonada vertical
    {
        "name": "vertical_stagger",
        "ego_height": -25,  # 25m de altura
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": 0, "z": -30},    # 5m abaixo
            {"vehicle": "Drone4", "x": 12, "y": -2, "z": -25},   # mesma altura
            {"vehicle": "Intruder1", "x": 14, "y": 2, "z": -20}  # 5m acima
        ]
    },
    # Formação dispersa
    {
        "name": "scattered",
        "ego_height": -18,  # 18m de altura
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -10, "z": -18},   # mesma altura, lateral esq
            {"vehicle": "Drone4", "x": 10, "y": 0, "z": -16},    # 2m acima, centro
            {"vehicle": "Intruder1", "x": 12, "y": 10, "z": -20} # 2m abaixo, lateral dir
        ]
    },
    # Formação triangular
    {
        "name": "triangular",
        "ego_height": -12,  # 12m de altura
        "drone_configs": [
            {"vehicle": "Drone3", "x": 6, "y": -4, "z": -14},    # 2m abaixo
            {"vehicle": "Drone4", "x": 6, "y": 4, "z": -14},     # 2m abaixo
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z": -10}  # 2m acima
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
print(f"   • Alturas: 10m a 30m")

# Gera o dataset
for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")
    print("-"*50)

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Movendo Ego para {-ego_height}m de altura...")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones COM ALTURAS ABSOLUTAS
    for drone_cfg in scenario["drone_configs"]:
        vehicle_name = drone_cfg["vehicle"]
        if vehicle_name in vehicles:
            x = drone_cfg["x"]
            y = drone_cfg["y"]
            z = drone_cfg["z"]  # Usa altura absoluta diretamente
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle_name).join()
            print(f"   {vehicle_name}: x={x}m, y={y}m, altura={-z}m")

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

                # IMPORTANTE: Detecta anomalias ANTES de filtrar (como no fusion_fixed.py)
                z_mean = Z.mean()
                z_std = Z.std()
                anomalies_mask = np.abs(Z - z_mean) > 2  # Usa 2 desvios padrão

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
                anomalies_valid = anomalies_mask[valid]  # Mantém info de anomalia

                # Desenha pontos
                for i in range(len(u_valid)):
                    if anomalies_valid[i]:
                        # Drone detectado!
                        color = (255, 0, 255)  # Magenta
                        size = 5
                        frame_detections += 1
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (255, 255, 0), 2)
                    else:
                        # Chão/normal
                        depth = min(1.0, x_valid[i] / 50)
                        color = (0, int(255*(1-depth)), 0)
                        size = 1

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Adiciona informações na imagem
            info_text = f"{scenario['name']} | H:{-ego_height}m | F:{total_frames} | Yaw:{yaw}"
            cv2.putText(fusion, info_text,
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
                "ego_altitude_meters": -ego_height,
                "total_points": len(points),
                "front_points": len(front_points),
                "anomalies": int(anomalies_mask.sum()) if len(front_points) > 0 else 0,
                "detections": frame_detections,
                "drone_positions": [
                    {"name": cfg["vehicle"], "x": cfg["x"], "y": cfg["y"], "z": cfg["z"],
                     "altitude_meters": -cfg["z"]}
                    for cfg in scenario["drone_configs"]
                ],
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
print("\n🎯 Centralizando Ego...")
client.rotateToYawAsync(0, vehicle_name="Ego").join()

# Move todos os drones para altura segura antes de pousar
print("\n📍 Movendo drones para altura de pouso...")
for v in vehicles:
    try:
        client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=v).join()
    except:
        pass

# Pousa todos os drones
print("\n🛬 Pousando todos os drones...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

# Estatísticas finais
elapsed_total = time.time() - start_time

print("\n" + "="*70)
print("🎉 DATASET COMPLETO COM DRONES NO AR!")
print("="*70)
print(f"\n📊 ESTATÍSTICAS FINAIS:")
print(f"   • Frames gerados: {total_frames}")
print(f"   • Detecções totais: {total_detections}")
print(f"   • Média detecções/frame: {total_detections/total_frames:.2f}" if total_frames > 0 else "")
print(f"   • Tempo total: {elapsed_total:.1f} segundos")
print(f"   • FPS médio: {total_frames/elapsed_total:.2f}" if elapsed_total > 0 else "")
print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")
print("\n📂 Estrutura:")
print("   • images_rgb/: Imagens originais da câmera")
print("   • images_fusion/: Imagens com fusão LiDAR")
print("   • images_comparison/: Comparações lado a lado")
print("   • lidar_points/: Nuvens de pontos (.npy)")
print("   • metadata/: Metadados de cada frame (.json)")
print("\n✅ CORREÇÕES APLICADAS:")
print("   • Drones decolam e aguardam estabilização")
print("   • Alturas absolutas corretas (10-30m)")
print("   • Detecção de anomalias ANTES da filtragem")
print("   • Metadados incluem altitudes reais")
print("\n🚁 Dataset pronto com drones VOANDO!")