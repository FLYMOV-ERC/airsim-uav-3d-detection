#!/usr/bin/env python3
"""
Gerador de Dataset usando API simGetDetections do AirSim
Esta é a forma CORRETA de obter bounding boxes de drones!
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 DATASET COM API simGetDetections (FORMA CORRETA!)")
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

# Verifica se a API existe
if not hasattr(client, 'simGetDetections'):
    print("⚠️ AVISO: simGetDetections não disponível nesta versão do AirSim")
    print("   Usando método alternativo...")
    USE_DETECTION_API = False
else:
    USE_DETECTION_API = True
    print("✅ API simGetDetections disponível!")

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

# Estrutura de diretórios
output_dir = Path("dataset_detection_api")
output_dir.mkdir(exist_ok=True)

# YOLO Dataset
yolo_dir = output_dir / "yolo"
yolo_dir.mkdir(exist_ok=True)
(yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
(yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
(yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
(yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

# PointNet Dataset
pointnet_dir = output_dir / "pointnet"
pointnet_dir.mkdir(exist_ok=True)
(pointnet_dir / "point_clouds").mkdir(exist_ok=True)
(pointnet_dir / "labels_3d").mkdir(exist_ok=True)

# Visualizações
vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)

# Metadata
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset será salvo em: {output_dir.absolute()}")

# Parâmetros da câmera
image_width = 1280
image_height = 720
FOV_H = 90
FOV_V = 60

fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = image_height / (2 * np.tan(np.radians(FOV_V/2)))
cx = image_width / 2
cy = image_height / 2

# Configurar filtros de detecção (se a API estiver disponível)
if USE_DETECTION_API:
    try:
        # Define raio de detecção (metros)
        client.simSetDetectionFilterRadius("front_center", airsim.ImageType.Scene, 100.0, vehicle_name="Ego")

        # Adiciona filtros para detectar drones
        for v in vehicles:
            if v != "Ego":
                client.simAddDetectionFilterMeshName("front_center", airsim.ImageType.Scene, v, vehicle_name="Ego")

        print("✅ Filtros de detecção configurados!")
    except Exception as e:
        print(f"⚠️ Erro configurando filtros: {e}")
        USE_DETECTION_API = False

def get_drone_positions():
    """
    Obtém posições ground truth dos drones
    """
    positions = {}
    ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
    ego_pos = np.array([ego_pose.position.x_val, ego_pose.position.y_val, ego_pose.position.z_val])

    for v in vehicles:
        if v != "Ego":
            try:
                pose = client.simGetVehiclePose(vehicle_name=v)
                pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
                relative_pos = pos - ego_pos
                positions[v] = {
                    'absolute': pos.tolist(),
                    'relative': relative_pos.tolist()
                }
            except:
                pass

    return positions

def project_3d_to_2d(position_3d, fx, fy, cx, cy):
    """
    Projeta posição 3D para pixels 2D
    """
    x, y, z = position_3d

    if x <= 0:  # Atrás da câmera
        return None

    u = fx * y / x + cx
    v = fy * (-z) / x + cy

    if 0 <= u < image_width and 0 <= v < image_height:
        return [u, v]

    return None

def depth_to_pointcloud(depth_array, fx, fy, cx, cy):
    """
    Converte depth map para point cloud
    """
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    valid = (depth_array > 0.1) & (depth_array < 100)

    z = depth_array[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    return np.stack([x, y, z], axis=-1)

# Cenários
scenarios = [
    {
        "name": "close_range",
        "ego_height": -10,
        "drones": [
            {"vehicle": "Drone3", "x": 5, "y": -2, "z": -10},
            {"vehicle": "Drone4", "x": 7, "y": 3, "z": -9},
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z": -11}
        ]
    },
    {
        "name": "medium_range",
        "ego_height": -15,
        "drones": [
            {"vehicle": "Drone3", "x": 10, "y": -4, "z": -15},
            {"vehicle": "Drone4", "x": 15, "y": 4, "z": -14},
            {"vehicle": "Intruder1", "x": 20, "y": 0, "z": -16}
        ]
    }
]

yaw_angles = [-20, -10, 0, 10, 20]
total_frames = 0
max_frames = 100

print(f"\n📊 Configuração:")
print(f"   • {len(scenarios)} cenários")
print(f"   • {len(yaw_angles)} ângulos por cenário")
print(f"   • Até {max_frames} frames")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona drones
    for drone_cfg in scenario["drones"]:
        if drone_cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(
                drone_cfg["x"], drone_cfg["y"], drone_cfg["z"], 5,
                vehicle_name=drone_cfg["vehicle"]
            ).join()

    time.sleep(2)

    for yaw in yaw_angles:
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:06d}"

        # Captura RGB e Depth
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
        ], vehicle_name="Ego")

        # Processa RGB
        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Processa Depth
        depth_array = None
        if responses[1].image_data_float:
            depth_1d = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_array = depth_1d.reshape(responses[1].height, responses[1].width)

            if depth_array.shape != (image_height, image_width):
                depth_array = cv2.resize(depth_array, (image_width, image_height))

        if img_bgr is not None and depth_array is not None:
            detections = []

            if USE_DETECTION_API:
                # USA A API OFICIAL DE DETECÇÃO
                try:
                    detection_results = client.simGetDetections("front_center", airsim.ImageType.Scene, vehicle_name="Ego")

                    for det in detection_results:
                        # det.box2D tem min e max pontos
                        x_min = int(det.box2D.min.x)
                        y_min = int(det.box2D.min.y)
                        x_max = int(det.box2D.max.x)
                        y_max = int(det.box2D.max.y)

                        # det.box3D tem center e halfExtents
                        center_3d = [det.box3D.center.x, det.box3D.center.y, det.box3D.center.z]
                        size_3d = [det.box3D.halfExtents.x * 2, det.box3D.halfExtents.y * 2, det.box3D.halfExtents.z * 2]

                        detections.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'center_3d': center_3d,
                            'size_3d': size_3d,
                            'name': det.name if hasattr(det, 'name') else 'drone'
                        })

                    print(f"   Frame {frame_name}: {len(detections)} drones detectados via API")

                except Exception as e:
                    print(f"   ⚠️ Erro na API de detecção: {e}")
                    USE_DETECTION_API = False

            if not USE_DETECTION_API:
                # FALLBACK: Usa posições conhecidas e projeta
                drone_positions = get_drone_positions()

                for drone_name, pos_info in drone_positions.items():
                    relative_pos = pos_info['relative']

                    # Projeta centro para 2D
                    center_2d = project_3d_to_2d(relative_pos, fx, fy, cx, cy)

                    if center_2d:
                        # Estima tamanho da bbox baseado na distância
                        distance = np.linalg.norm(relative_pos)
                        # Tamanho aproximado do drone em pixels (inversamente proporcional à distância)
                        bbox_size = max(20, min(200, 500 / distance))

                        x_min = int(center_2d[0] - bbox_size/2)
                        y_min = int(center_2d[1] - bbox_size/2)
                        x_max = int(center_2d[0] + bbox_size/2)
                        y_max = int(center_2d[1] + bbox_size/2)

                        # Clipa para limites da imagem
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(image_width-1, x_max)
                        y_max = min(image_height-1, y_max)

                        detections.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'center_3d': relative_pos,
                            'size_3d': [1.0, 0.5, 1.0],  # Tamanho estimado
                            'name': drone_name
                        })

                print(f"   Frame {frame_name}: {len(detections)} drones via projeção")

            # Cria visualização
            vis = img_bgr.copy()

            # Split treino/validação
            is_train = total_frames % 5 != 0
            split = "train" if is_train else "val"

            # Labels YOLO
            yolo_labels = []

            # Labels 3D
            labels_3d = []

            for det in detections:
                x_min, y_min, x_max, y_max = det['bbox']

                # Para YOLO
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Validação
                if 0 < width < 1 and 0 < height < 1:
                    yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    labels_3d.append({
                        "class": "drone",
                        "center": det['center_3d'],
                        "size": det['size_3d'],
                        "bbox_2d": det['bbox'],
                        "name": det['name']
                    })

                    # Desenha visualização
                    cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(vis, det['name'],
                               (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 1)

            # Salva imagem RGB
            cv2.imwrite(str(yolo_dir / "images" / split / f"{frame_name}.jpg"), img_bgr)

            # Salva label YOLO
            if yolo_labels:
                with open(yolo_dir / "labels" / split / f"{frame_name}.txt", 'w') as f:
                    f.write('\n'.join(yolo_labels))

            # Gera e salva point cloud
            point_cloud = depth_to_pointcloud(depth_array, fx, fy, cx, cy)
            np.save(pointnet_dir / "point_clouds" / f"{frame_name}.npy", point_cloud)

            # Salva labels 3D
            if labels_3d:
                with open(pointnet_dir / "labels_3d" / f"{frame_name}.json", 'w') as f:
                    json.dump(labels_3d, f, indent=2)

            # Salva visualização
            cv2.imwrite(str(vis_dir / f"{frame_name}.jpg"), vis)

            # Metadata
            metadata = {
                "frame_id": int(total_frames),
                "scenario": scenario['name'],
                "yaw": float(yaw),
                "split": split,
                "num_detections": len(detections),
                "detections": detections,
                "api_used": "simGetDetections" if USE_DETECTION_API else "projection"
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

    if total_frames >= max_frames:
        break

# Pousa drones
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

# Cria dataset.yaml para YOLO
dataset_yaml = f"""
path: {yolo_dir.absolute()}
train: images/train
val: images/val

nc: 1
names: ['drone']
"""

with open(yolo_dir / "dataset.yaml", 'w') as f:
    f.write(dataset_yaml)

print("\n" + "="*70)
print("🎉 DATASET COMPLETO!")
print("="*70)
print(f"\n📊 Estatísticas:")
print(f"   • Frames gerados: {total_frames}")
print(f"   • Método usado: {'API simGetDetections' if USE_DETECTION_API else 'Projeção de posições'}")
print(f"\n✅ Este é o método correto para detectar drones no AirSim!")