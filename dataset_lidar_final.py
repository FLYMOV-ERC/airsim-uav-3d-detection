#!/usr/bin/env python3
"""
Dataset FINAL com LiDAR detectando drones
Posiciona drones em alturas onde o LiDAR consegue detectar
"""

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import math
from tqdm import tqdm

def detect_drones_simple(points, drone_positions, ego_position):
    """Detecção simplificada de drones"""
    detections = []

    for drone_name, pos in drone_positions.items():
        # Posição relativa ao Ego
        rel_pos = np.array([
            pos['x'] - ego_position['x'],
            pos['y'] - ego_position['y'],
            pos['z'] - ego_position['z']
        ])

        # Conta pontos próximos
        if len(points) > 0:
            distances = np.linalg.norm(points - rel_pos, axis=1)
            near_points = np.sum(distances < 3.0)  # Raio de 3m

            detected = near_points >= 5  # Mínimo 5 pontos

            detections.append({
                'name': drone_name,
                'points': int(near_points),
                'detected': bool(detected),  # Converter numpy.bool_ para Python bool
                'distance': float(np.linalg.norm(rel_pos))
            })

    return detections

def main():
    print("\n" + "="*60)
    print("🚀 DATASET FINAL - LIDAR COM DETECÇÃO FUNCIONANDO")
    print("="*60)

    # Configuração
    output_dir = Path("dataset_lidar_final")
    output_dir.mkdir(exist_ok=True)

    dirs = {
        'rgb': output_dir / 'rgb',
        'lidar': output_dir / 'lidar',
        'lidar_filtered': output_dir / 'lidar_filtered',
        'detections': output_dir / 'detections'
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True)

    print(f"\n📁 Dataset em: {output_dir.absolute()}")

    # Conecta
    print("\n🔌 Conectando...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Drones: {vehicles}")

    # Prepara drones
    print("\n🎮 Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
        except:
            pass

    # Decola
    print("\n🛫 Decolando...")
    for vehicle in vehicles:
        try:
            client.takeoffAsync(vehicle_name=vehicle)
        except:
            pass
    time.sleep(5)

    # Configuração
    num_frames = 30
    frames_captured = 0
    detection_stats = {
        'total_frames': 0,
        'frames_with_drones': 0,
        'total_detections': 0
    }

    print(f"\n📸 Capturando {num_frames} frames")
    print("🔑 Estratégia: Drones em alturas baixas (0-2m) onde LiDAR detecta\n")

    with tqdm(total=num_frames, desc="Capturando") as pbar:
        for frame_idx in range(num_frames):

            # POSICIONAMENTO ESTRATÉGICO DOS DRONES
            # Mantemos drones em alturas baixas onde o LiDAR detecta

            # Ego sempre alto para ter boa visão
            client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego")

            # Outros drones: movimento circular baixo
            angle_base = (frame_idx * 10) % 360

            # Drone3: muito baixo, próximo
            angle = math.radians(angle_base)
            x = 10 * math.cos(angle)
            y = 10 * math.sin(angle)
            z = -0.5  # 50cm do chão
            client.moveToPositionAsync(x, y, z, 5, vehicle_name="Drone3")

            # Drone4: baixo, distância média
            angle = math.radians(angle_base + 120)
            x = 15 * math.cos(angle)
            y = 15 * math.sin(angle)
            z = -1.0  # 1m do chão
            client.moveToPositionAsync(x, y, z, 5, vehicle_name="Drone4")

            # Intruder1: baixo, mais longe
            angle = math.radians(angle_base + 240)
            x = 20 * math.cos(angle)
            y = 20 * math.sin(angle)
            z = -1.5  # 1.5m do chão
            client.moveToPositionAsync(x, y, z, 5, vehicle_name="Intruder1")

            # Aguarda movimentos
            time.sleep(0.5)

            # CAPTURA
            try:
                # 1. RGB
                responses = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
                ], vehicle_name="Ego")

                if responses[0].image_data_uint8:
                    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # Adiciona informação
                    cv2.putText(img_bgr, f"Frame {frames_captured}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_file), img_bgr)

                # 2. LiDAR
                lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

                if lidar_data and len(lidar_data.point_cloud) > 3:
                    points_raw = np.array(lidar_data.point_cloud, dtype=np.float32)
                    points_raw = points_raw.reshape(-1, 3)

                    # Salva pontos originais
                    np.save(dirs['lidar'] / f"frame_{frames_captured:06d}.npy", points_raw)

                    # Filtra FOV da câmera
                    x = points_raw[:, 0]
                    y = points_raw[:, 1]

                    # Pontos frontais no FOV
                    in_front = x > 0
                    angles = np.degrees(np.arctan2(y, x))
                    in_fov = (angles >= -45) & (angles <= 45)

                    points_filtered = points_raw[in_front & in_fov]
                    np.save(dirs['lidar_filtered'] / f"frame_{frames_captured:06d}.npy", points_filtered)

                    # Obtém posições
                    ego_state = client.getMultirotorState(vehicle_name="Ego")
                    ego_position = {
                        'x': ego_state.kinematics_estimated.position.x_val,
                        'y': ego_state.kinematics_estimated.position.y_val,
                        'z': ego_state.kinematics_estimated.position.z_val
                    }

                    drone_positions = {}
                    for vehicle in vehicles:
                        if vehicle != "Ego":
                            try:
                                state = client.getMultirotorState(vehicle_name=vehicle)
                                drone_positions[vehicle] = {
                                    'x': state.kinematics_estimated.position.x_val,
                                    'y': state.kinematics_estimated.position.y_val,
                                    'z': state.kinematics_estimated.position.z_val
                                }
                            except:
                                pass

                    # Detecta drones
                    drone_detections = detect_drones_simple(points_filtered, drone_positions, ego_position)

                    # Estatísticas
                    num_detected = sum(1 for d in drone_detections if d['detected'])
                    if num_detected > 0:
                        detection_stats['frames_with_drones'] += 1
                        detection_stats['total_detections'] += num_detected

                    pbar.write(f"   Frame {frames_captured}: {len(points_filtered)} pts no FOV, "
                              f"{num_detected}/{len(drone_detections)} drones detectados")

                    # Salva detecções
                    detections_data = {
                        'frame': frames_captured,
                        'total_points': int(len(points_raw)),
                        'points_in_fov': int(len(points_filtered)),
                        'drone_detections': drone_detections
                    }

                    det_file = dirs['detections'] / f"frame_{frames_captured:06d}.json"
                    with open(det_file, 'w') as f:
                        json.dump(detections_data, f, indent=2)

                frames_captured += 1
                detection_stats['total_frames'] += 1
                pbar.update(1)

            except Exception as e:
                pbar.write(f"⚠️ Erro frame {frame_idx}: {e}")
                continue

    # Pousa
    print("\n🛬 Pousando...")
    for vehicle in vehicles:
        try:
            client.landAsync(vehicle_name=vehicle)
        except:
            pass

    time.sleep(3)

    for vehicle in vehicles:
        try:
            client.armDisarm(False, vehicle)
            client.enableApiControl(False, vehicle)
        except:
            pass

    # ESTATÍSTICAS
    print("\n" + "="*60)
    print("✅ DATASET COMPLETO!")
    print("="*60)

    print(f"\n📊 Estatísticas:")
    print(f"   Frames totais: {frames_captured}")
    print(f"   Frames com drones detectados: {detection_stats['frames_with_drones']}")
    print(f"   Taxa de detecção: {detection_stats['frames_with_drones']/max(1, detection_stats['total_frames'])*100:.1f}%")
    print(f"   Total de detecções: {detection_stats['total_detections']}")

    print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")

    if detection_stats['frames_with_drones'] > 0:
        print("\n🎯 SUCESSO! LiDAR está detectando drones na nuvem de pontos!")
    else:
        print("\n⚠️ Baixa detecção - pode ser necessário ajustar parâmetros")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")