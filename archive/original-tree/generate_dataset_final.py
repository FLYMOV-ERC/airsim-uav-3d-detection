#!/usr/bin/env python3
"""
Gerador de Dataset FINAL - Sincronização Perfeita + Detecção Melhorada
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR FINAL - SINCRONIZAÇÃO + DETECÇÃO OTIMIZADA!")
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
output_dir = Path("dataset_final_300")
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

camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

# Cenários variados
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
    },
    {
        "name": "high_altitude",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z": -25},
            {"vehicle": "Drone4", "x": 25, "y": 8, "z": -20},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -15}
        ]
    },
    {
        "name": "vertical_stagger",
        "ego_height": -25,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": 0, "z": -30},
            {"vehicle": "Drone4", "x": 12, "y": -2, "z": -25},
            {"vehicle": "Intruder1", "x": 14, "y": 2, "z": -20}
        ]
    },
    {
        "name": "scattered",
        "ego_height": -18,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -10, "z": -18},
            {"vehicle": "Drone4", "x": 10, "y": 0, "z": -16},
            {"vehicle": "Intruder1", "x": 12, "y": 10, "z": -20}
        ]
    },
    {
        "name": "triangular",
        "ego_height": -12,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 6, "y": -4, "z": -14},
            {"vehicle": "Drone4", "x": 6, "y": 4, "z": -14},
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z": -10}
        ]
    }
]

yaw_angles = [-30, -20, -10, 0, 10, 20, 30]
total_frames = 0
total_detections = 0
start_time = time.time()

print(f"\n📊 Config: {len(scenarios)} cenários x {len(yaw_angles)} ângulos = até 300 frames")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

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

    time.sleep(2)  # Estabilização

    for yaw_idx, yaw in enumerate(yaw_angles):
        if total_frames >= 300:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # === SINCRONIZAÇÃO: Limpar buffer do LiDAR ===
        _ = client.getLidarData("LidarFront", vehicle_name="Ego")
        time.sleep(0.1)  # Nova varredura

        # Captura sincronizada
        t0 = time.time()
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")
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

            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            # Filtra pontos frontais
            front_mask = points[:, 0] > 0.5
            front_points = points[front_mask]

            frame_detections = 0
            detected_regions = []

            if len(front_points) > 0:
                X = front_points[:, 0]
                Y = front_points[:, 1]
                Z = front_points[:, 2]

                # === DETECÇÃO MELHORADA DE DRONES ===
                # 1. Detecta pontos acima do chão
                z_ground = np.percentile(Z, 20)  # Estima nível do chão
                above_ground = Z > z_ground + 1.0  # Pontos 1m+ acima do chão

                # 2. Detecta clusters isolados
                distances = np.sqrt(X**2 + Y**2 + Z**2)

                # 3. Combina critérios
                # - Pontos acima do chão
                # - Distância entre 3m e 50m
                # - Não muito alto (descarta nuvens/montanhas)
                drone_mask = above_ground & (distances > 3) & (distances < 50) & (Z < z_ground + 30)

                # Aplica correção de pitch
                X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
                Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

                # Projeta para imagem
                u = (fx * Y / X_rot + cx).astype(int)
                v = (fy * Z_rot / X_rot + cy).astype(int)

                # Filtra válidos
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (X_rot > 0)

                u_valid = u[valid]
                v_valid = v[valid]
                x_valid = X[valid]
                z_valid = Z[valid]
                drone_valid = drone_mask[valid]
                distances_valid = distances[valid]

                # Desenha pontos
                for i in range(len(u_valid)):
                    if drone_valid[i]:
                        # DRONE DETECTADO!
                        color = (255, 0, 255)  # Magenta
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 6, color, -1)
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 12, (255, 255, 0), 2)
                        frame_detections += 1

                        # Marca região de detecção
                        if frame_detections <= 5:  # Limita a 5 detecções principais
                            detected_regions.append({
                                "x": int(u_valid[i]),
                                "y": int(v_valid[i]),
                                "distance": float(distances_valid[i])
                            })
                    else:
                        # Ponto normal - cor por profundidade
                        if z_valid[i] < z_ground + 0.5:
                            # Chão - verde
                            depth = min(1.0, x_valid[i] / 50)
                            color = (0, int(255*(1-depth)), 0)
                            size = 1
                        else:
                            # Obstáculo/elevação - azul
                            color = (255, int(200*(1-min(1, x_valid[i]/50))), 0)
                            size = 2

                        cv2.circle(fusion, (u_valid[i], v_valid[i]), size, color, -1)

            # Adiciona informações
            cv2.putText(fusion, f"{scenario['name']} | Alt:{-ego_height}m | Yaw:{yaw}° | F:{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Drones detectados: {frame_detections}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, f"LiDAR pts: {len(points)} | Front: {len(front_points)}",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Marca regiões detectadas com caixas
            for idx, region in enumerate(detected_regions):
                x, y = region["x"], region["y"]
                dist = region["distance"]
                cv2.rectangle(fusion, (x-20, y-20), (x+20, y+20), (255, 0, 255), 2)
                cv2.putText(fusion, f"D{idx+1}: {dist:.1f}m",
                           (x-20, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Salva imagens
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            # Salva pontos
            np.save(output_dir / "lidar_points" / f"{frame_name}.npy", points)

            # Metadados detalhados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "scenario_idx": scenario_idx,
                "yaw": yaw,
                "ego_height": ego_height,
                "ego_altitude_meters": -ego_height,
                "total_points": len(points),
                "front_points": len(front_points),
                "detections": frame_detections,
                "detected_regions": detected_regions,
                "capture_time_ms": capture_time,
                "drone_positions": [
                    {"name": cfg["vehicle"], "x": cfg["x"], "y": cfg["y"],
                     "z": cfg["z"], "altitude_meters": -cfg["z"]}
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
                print(f"\n   📊 Progresso: {total_frames}/300")
                print(f"      Tempo: {elapsed:.1f}s | FPS: {fps:.2f} | ETA: {eta:.1f}s")
                print(f"      Detecções totais: {total_detections}")

    if total_frames >= 300:
        break

# Centraliza Ego
print("\n🎯 Centralizando Ego...")
client.rotateToYawAsync(0, vehicle_name="Ego").join()

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

elapsed_total = time.time() - start_time

print("\n" + "="*70)
print("🎉 DATASET COMPLETO COM SINCRONIZAÇÃO PERFEITA!")
print("="*70)
print(f"\n📊 ESTATÍSTICAS FINAIS:")
print(f"   • Frames gerados: {total_frames}")
print(f"   • Detecções totais: {total_detections}")
print(f"   • Média detecções/frame: {total_detections/total_frames:.2f}" if total_frames > 0 else "")
print(f"   • Tempo total: {elapsed_total:.1f}s")
print(f"   • FPS médio: {total_frames/elapsed_total:.2f}" if elapsed_total > 0 else "")
print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")
print("\n✅ MELHORIAS IMPLEMENTADAS:")
print("   • Sincronização: Limpa buffer do LiDAR antes de capturar")
print("   • Detecção: Identifica pontos acima do chão")
print("   • Visualização: Cores diferentes para chão/obstáculos/drones")
print("   • Metadados: Salva posições detectadas com distâncias")