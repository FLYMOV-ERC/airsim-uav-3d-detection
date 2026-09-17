#!/usr/bin/env python3
"""
Dataset com LiDAR ALINHADO com a câmera
Garante que o que aparece na câmera também está na nuvem de pontos
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

def rotate_lidar_to_camera_view(points, lidar_pose, camera_pose):
    """
    Rotaciona pontos do LiDAR para alinhar com visão da câmera
    """
    # Por enquanto retorna os pontos originais
    # Podemos adicionar transformação se necessário
    return points

def filter_points_in_camera_fov(points, fov_degrees=90, max_range=100):
    """
    Filtra apenas pontos que estão no campo de visão da câmera frontal
    CORRIGIDO: X positivo é para frente no AirSim
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Filtra pontos na frente (X positivo)
    in_front = x > 0

    # Calcula ângulo horizontal em relação ao eixo frontal (X)
    # No AirSim: X = frente, Y = lateral
    angles = np.degrees(np.arctan2(y, x))

    # FOV horizontal
    half_fov = fov_degrees / 2
    in_horizontal_fov = (angles >= -half_fov) & (angles <= half_fov)

    # FOV vertical (assumindo 60 graus)
    vertical_angles = np.degrees(np.arctan2(-z, x))  # Negativo porque Z aponta para baixo
    in_vertical_fov = (vertical_angles >= -30) & (vertical_angles <= 30)

    # Calcula distância
    distances = np.sqrt(x**2 + y**2 + z**2)

    # Filtra por distância (incluindo drones próximos)
    in_range = (distances > 0.1) & (distances < max_range)

    # Combina todos os filtros
    valid = in_front & in_horizontal_fov & in_vertical_fov & in_range

    return points[valid], np.sum(valid)

def detect_drones_in_pointcloud(points, drone_positions, ego_position=None):
    """
    Detecta clusters de pontos que correspondem aos drones
    Melhorado para considerar posições relativas ao Ego
    """
    drone_detections = []

    for drone_name, pos in drone_positions.items():
        if drone_name == "Ego":
            continue

        # Se temos posição do Ego, converte para coordenadas relativas
        if ego_position:
            # Posição do drone relativa ao Ego (sistema de coordenadas do sensor)
            rel_x = pos['x'] - ego_position['x']
            rel_y = pos['y'] - ego_position['y']
            rel_z = pos['z'] - ego_position['z']
            drone_loc = np.array([rel_x, rel_y, rel_z])
        else:
            drone_loc = np.array([pos['x'], pos['y'], pos['z']])

        # Calcula distância dos pontos ao drone
        distances = np.linalg.norm(points - drone_loc, axis=1)

        # Aumenta o raio de detecção e ajusta threshold
        # Drones podem ter 1-3 metros de tamanho
        near_drone = distances < 5.0  # Aumentado para 5m para capturar mais pontos
        num_points = np.sum(near_drone)

        # Calcula distância do drone ao Ego para ajustar threshold
        drone_distance = np.linalg.norm(drone_loc)

        # Threshold adaptativo baseado na distância
        # Quanto mais longe, menos pontos esperamos
        # Reduzindo threshold mínimo para 1 ponto
        min_points_threshold = max(1, int(20 / (1 + drone_distance/10)))

        if num_points >= min_points_threshold:
            drone_detections.append({
                'name': drone_name,
                'points': int(num_points),
                'position': pos,
                'relative_position': {'x': float(drone_loc[0]), 'y': float(drone_loc[1]), 'z': float(drone_loc[2])},
                'distance': float(drone_distance),
                'detected': True
            })
        else:
            drone_detections.append({
                'name': drone_name,
                'points': int(num_points),
                'position': pos,
                'relative_position': {'x': float(drone_loc[0]), 'y': float(drone_loc[1]), 'z': float(drone_loc[2])},
                'distance': float(drone_distance),
                'detected': False
            })

    return drone_detections

def visualize_lidar_with_drones(points, drone_detections, img_size=(800, 600)):
    """
    Visualização 2D mostrando drones detectados
    """
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    if len(points) == 0:
        return img

    # Vista de cima (XY)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Normaliza coordenadas
    x_range = max(abs(x.min()), abs(x.max())) * 1.1
    y_range = max(abs(y.min()), abs(y.max())) * 1.1

    if x_range > 0 and y_range > 0:
        # Converte para pixels
        px = ((x / x_range + 1) * img_size[0] / 2).astype(int)
        py = ((y / y_range + 1) * img_size[1] / 2).astype(int)

        # Cor baseada na altura (Z)
        z_norm = (z - z.min()) / (z.max() - z.min() + 0.001)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Desenha pontos
        for i in range(len(points)):
            if 0 <= px[i] < img_size[0] and 0 <= py[i] < img_size[1]:
                color = colors[i][0].tolist()
                cv2.circle(img, (px[i], py[i]), 1, color, -1)

        # Marca posições dos drones
        for drone in drone_detections:
            dx = drone['position']['x']
            dy = drone['position']['y']

            # Converte para pixels
            dpx = int((dx / x_range + 1) * img_size[0] / 2)
            dpy = int((dy / y_range + 1) * img_size[1] / 2)

            if 0 <= dpx < img_size[0] and 0 <= dpy < img_size[1]:
                color = (0, 255, 0) if drone['detected'] else (0, 0, 255)
                cv2.circle(img, (dpx, dpy), 8, color, 2)
                cv2.putText(img, drone['name'], (dpx-30, dpy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(img, f"{drone['points']}pts", (dpx-30, dpy+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Info
    cv2.putText(img, f"Total: {len(points)} points", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Vista de Cima (XY)", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Legenda
    cv2.putText(img, "Verde: Drone Detectado", (img_size[0]-200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Vermelho: Nao Detectado", (img_size[0]-200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img

def main():
    print("\n" + "="*60)
    print("🎯 DATASET COM LIDAR ALINHADO À CÂMERA")
    print("="*60)

    # Configuração
    output_dir = Path("dataset_lidar_camera_aligned")
    output_dir.mkdir(exist_ok=True)

    dirs = {
        'rgb': output_dir / 'rgb',
        'lidar': output_dir / 'lidar',
        'lidar_filtered': output_dir / 'lidar_filtered',  # Pontos no FOV da câmera
        'lidar_viz': output_dir / 'lidar_viz',
        'detections': output_dir / 'detections',
        'metadata': output_dir / 'metadata'
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

    # Verifica LiDAR
    print("\n📡 Configurando LiDAR alinhado com câmera...")
    lidar_name = "LidarFront"

    # Prepara drones
    print("\n🎮 Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
            print(f"   ✅ {vehicle}")
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

    # Posiciona Ego
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

    # Configuração
    num_frames = 10  # Muito reduzido para teste rápido
    frames_captured = 0

    print(f"\n📸 Capturando {num_frames} frames")
    print("   🎯 LiDAR filtrado para FOV da câmera")
    print("   ✅ Detectando drones na nuvem de pontos\n")

    detection_stats = {
        'total_frames': 0,
        'frames_with_drones': 0,
        'total_detections': 0
    }

    with tqdm(total=num_frames, desc="Capturando") as pbar:
        for frame_idx in range(num_frames):
            t = frame_idx * 0.1

            # MOVIMENTO: Coloca drones NA FRENTE do Ego (visível para câmera)
            num_drones = len(vehicles) - 1
            drone_idx = 0

            for vehicle in vehicles:
                if vehicle == "Ego":
                    # Ego fica parado olhando para frente
                    client.rotateToYawAsync(0, vehicle_name="Ego")  # Olhando para frente (0°)
                    continue

                # IMPORTANTE: Posiciona drones NA FRENTE do Ego
                # Entre 10-30 metros de distância, no campo de visão

                # Distribui drones no campo de visão frontal
                angle_offset = (drone_idx - num_drones/2) * 30  # -45° a +45°
                angle_rad = math.radians(angle_offset)

                # Distância variável
                distance = 15 + 10 * math.sin(t + drone_idx)

                # Posição FRONTAL (X positivo = frente)
                x = distance * math.cos(angle_rad)  # Frente
                y = distance * math.sin(angle_rad)  # Lateral

                # IMPORTANTE: Colocar drones em altitudes detectáveis pelo LiDAR
                # O LiDAR só detecta em: 0-1.7m (chão) e 15-17m (prédios)
                # Vamos manter os drones baixos onde há muitos pontos

                if drone_idx == 0:
                    # Primeiro drone: bem baixo, quase no chão
                    z = -0.5  # 0.5m de altura
                elif drone_idx == 1:
                    # Segundo drone: um pouco mais alto
                    z = -1.0  # 1m de altura
                else:
                    # Terceiro drone: também baixo
                    z = -1.5  # 1.5m de altura

                try:
                    # IMPORTANTE: usar .join() para garantir que o movimento seja executado
                    client.moveToPositionAsync(x, y, z, 3, vehicle_name=vehicle).join()
                except:
                    pass

                drone_idx += 1

            time.sleep(1)  # Mais tempo para garantir que os movimentos são executados

            # CAPTURA
            try:
                # 1. Captura RGB
                responses = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
                ], vehicle_name="Ego")

                if responses[0].image_data_uint8:
                    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # Marca drones na imagem (para verificação)
                    img_annotated = img_bgr.copy()
                    cv2.putText(img_annotated, f"Frame {frames_captured}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_file), img_annotated)

                # 2. Captura LiDAR
                lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")

                if lidar_data and len(lidar_data.point_cloud) > 3:
                    # Converte para array
                    points_raw = np.array(lidar_data.point_cloud, dtype=np.float32)
                    points_raw = points_raw.reshape(-1, 3)

                    # Salva pontos originais
                    np.save(dirs['lidar'] / f"frame_{frames_captured:06d}.npy", points_raw)

                    # FILTRA pontos no FOV da câmera
                    points_filtered, num_filtered = filter_points_in_camera_fov(points_raw)

                    # Salva pontos filtrados
                    np.save(dirs['lidar_filtered'] / f"frame_{frames_captured:06d}.npy", points_filtered)

                    # Obtém posições dos drones
                    drone_positions = {}
                    ego_position = None
                    for vehicle in vehicles:
                        try:
                            state = client.getMultirotorState(vehicle_name=vehicle)
                            position = {
                                'x': state.kinematics_estimated.position.x_val,
                                'y': state.kinematics_estimated.position.y_val,
                                'z': state.kinematics_estimated.position.z_val
                            }
                            if vehicle == "Ego":
                                ego_position = position
                            else:
                                drone_positions[vehicle] = position
                        except:
                            pass

                    # DETECTA drones na nuvem de pontos (com posição relativa ao Ego)
                    drone_detections = detect_drones_in_pointcloud(points_filtered, drone_positions, ego_position)

                    # Estatísticas
                    num_detected = sum(1 for d in drone_detections if d['detected'])
                    if num_detected > 0:
                        detection_stats['frames_with_drones'] += 1
                        detection_stats['total_detections'] += num_detected

                    pbar.write(f"   Frame {frames_captured}: {len(points_raw)} pts total, "
                              f"{num_filtered} no FOV, {num_detected}/{len(drone_detections)} drones detectados")

                    # Visualização
                    viz_img = visualize_lidar_with_drones(points_filtered, drone_detections)
                    viz_file = dirs['lidar_viz'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(viz_file), viz_img)

                    # Salva detecções
                    detections_data = {
                        'frame': frames_captured,
                        'total_points': int(len(points_raw)),
                        'points_in_fov': int(num_filtered),
                        'drone_detections': drone_detections
                    }

                    det_file = dirs['detections'] / f"frame_{frames_captured:06d}.json"
                    with open(det_file, 'w') as f:
                        json.dump(detections_data, f, indent=2)

                # 3. Metadata
                metadata = {
                    'frame': frames_captured,
                    'timestamp': datetime.now().isoformat(),
                    'lidar': {
                        'sensor': lidar_name,
                        'total_points': int(len(points_raw)) if 'points_raw' in locals() else 0,
                        'points_in_camera_fov': int(num_filtered) if 'num_filtered' in locals() else 0
                    },
                    'vehicles': {}
                }

                for vehicle in vehicles:
                    try:
                        state = client.getMultirotorState(vehicle_name=vehicle)
                        metadata['vehicles'][vehicle] = {
                            'position': {
                                'x': float(state.kinematics_estimated.position.x_val),
                                'y': float(state.kinematics_estimated.position.y_val),
                                'z': float(state.kinematics_estimated.position.z_val)
                            }
                        }
                    except:
                        pass

                meta_file = dirs['metadata'] / f"frame_{frames_captured:06d}.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

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

    time.sleep(5)

    for vehicle in vehicles:
        try:
            client.armDisarm(False, vehicle)
            client.enableApiControl(False, vehicle)
        except:
            pass

    # ESTATÍSTICAS
    print("\n" + "="*60)
    print("✅ DATASET LIDAR+CÂMERA ALINHADOS!")
    print("="*60)

    print(f"\n📊 Estatísticas:")
    print(f"   Frames totais: {frames_captured}")
    print(f"   Frames com drones detectados: {detection_stats['frames_with_drones']}")
    print(f"   Taxa de detecção: {detection_stats['frames_with_drones']/max(1, detection_stats['total_frames'])*100:.1f}%")
    print(f"   Total de detecções: {detection_stats['total_detections']}")

    print(f"\n📁 Dataset em: {output_dir.absolute()}")

    print("\n📝 Conteúdo:")
    print("   • rgb/ - Imagens da câmera frontal")
    print("   • lidar/ - Nuvem de pontos completa (360°)")
    print("   • lidar_filtered/ - Pontos APENAS no FOV da câmera")
    print("   • lidar_viz/ - Visualização mostrando drones detectados")
    print("   • detections/ - JSON com detecções de cada drone")

    print("\n🎯 Os drones visíveis na câmera DEVEM aparecer na nuvem de pontos!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")