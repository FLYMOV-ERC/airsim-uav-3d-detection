#!/usr/bin/env python3
"""
Gerador de Dataset para YOLO + PointNet (Frustum-PointNet Pipeline)
Gera:
1. Dataset RGB com bounding boxes 2D (formato YOLO)
2. Dataset PointCloud com bounding boxes 3D (formato KITTI/PointNet)
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR DE DATASET YOLO + POINTNET")
print("Pipeline: RGB → YOLO (2D) → Frustum → PointNet (3D)")
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

# Estrutura de diretórios
output_dir = Path("dataset_frustum_pointnet")
output_dir.mkdir(exist_ok=True)

# Dataset 2D (YOLO)
yolo_dir = output_dir / "yolo_dataset"
yolo_dir.mkdir(exist_ok=True)
(yolo_dir / "images").mkdir(exist_ok=True)
(yolo_dir / "labels").mkdir(exist_ok=True)

# Dataset 3D (PointNet)
pointnet_dir = output_dir / "pointnet_dataset"
pointnet_dir.mkdir(exist_ok=True)
(pointnet_dir / "point_clouds").mkdir(exist_ok=True)
(pointnet_dir / "labels_3d").mkdir(exist_ok=True)
(pointnet_dir / "frustums").mkdir(exist_ok=True)

# Visualizações
vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)
(vis_dir / "bbox_2d").mkdir(exist_ok=True)
(vis_dir / "bbox_3d").mkdir(exist_ok=True)

# Metadados
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset será salvo em: {output_dir.absolute()}")

# Parâmetros da câmera (CRÍTICOS para projeção)
FOV_H = 90  # graus horizontal
FOV_V = 60  # graus vertical (estimado para 16:9)
image_width = 1280
image_height = 720

fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = image_height / (2 * np.tan(np.radians(FOV_V/2)))
cx = image_width / 2
cy = image_height / 2

# Matriz intrínseca da câmera (K)
K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
])

# Cenários variados
scenarios = [
    {
        "name": "close_range",
        "ego_height": -10,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 5, "y": -2, "z": -10, "class": "drone"},
            {"vehicle": "Drone4", "x": 7, "y": 2, "z": -9, "class": "drone"},
            {"vehicle": "Intruder1", "x": 10, "y": 0, "z": -11, "class": "intruder"}
        ]
    },
    {
        "name": "medium_range",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": -4, "z": -15, "class": "drone"},
            {"vehicle": "Drone4", "x": 15, "y": 3, "z": -14, "class": "drone"},
            {"vehicle": "Intruder1", "x": 20, "y": 0, "z": -16, "class": "intruder"}
        ]
    },
    {
        "name": "long_range",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -6, "z": -21, "class": "drone"},
            {"vehicle": "Drone4", "x": 20, "y": 5, "z": -19, "class": "drone"},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -20, "class": "intruder"}
        ]
    },
    {
        "name": "varied_heights",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -10, "class": "drone"},    # 5m acima
            {"vehicle": "Drone4", "x": 12, "y": 4, "z": -15, "class": "drone"},   # mesma altura
            {"vehicle": "Intruder1", "x": 16, "y": 0, "z": -20, "class": "intruder"} # 5m abaixo
        ]
    }
]

# Classes para YOLO (IDs)
class_names = ["drone", "intruder"]
class_ids = {"drone": 0, "intruder": 1}

yaw_angles = [-30, -15, 0, 15, 30]
total_frames = 0
max_frames = 500

# Tamanho aproximado dos drones (metros)
DRONE_SIZE = {"width": 0.8, "height": 0.3, "depth": 0.8}  # Ajuste conforme seu modelo

print(f"\n📊 Configuração:")
print(f"   • {len(scenarios)} cenários x {len(yaw_angles)} ângulos")
print(f"   • Até {max_frames} frames")
print(f"   • Classes: {class_names}")
print(f"   • Formato: YOLO (2D) + KITTI-like (3D)")

def get_drone_actual_positions(client, vehicles):
    """Pega posições reais dos drones via API"""
    positions = {}
    for v in vehicles:
        if v != "Ego":  # Não pega posição do Ego
            try:
                pose = client.simGetVehiclePose(vehicle_name=v)
                positions[v] = {
                    "x": pose.position.x_val,
                    "y": pose.position.y_val,
                    "z": pose.position.z_val,
                    "orientation": {
                        "w": pose.orientation.w_val,
                        "x": pose.orientation.x_val,
                        "y": pose.orientation.y_val,
                        "z": pose.orientation.z_val
                    }
                }
            except:
                pass
    return positions

def project_3d_to_2d(point_3d, K):
    """Projeta ponto 3D para coordenadas 2D da imagem"""
    if point_3d[0] <= 0:  # Ponto atrás da câmera
        return None

    # Projeta
    point_2d_h = K @ point_3d
    point_2d = point_2d_h[:2] / point_2d_h[2]

    return point_2d.astype(int)

def get_bbox_2d_from_3d(corners_3d, K, img_width, img_height):
    """Converte bounding box 3D para 2D"""
    points_2d = []

    for corner in corners_3d:
        p2d = project_3d_to_2d(corner, K)
        if p2d is not None:
            points_2d.append(p2d)

    if len(points_2d) < 4:
        return None

    points_2d = np.array(points_2d)

    # Calcula bounding box 2D
    x_min = max(0, points_2d[:, 0].min())
    x_max = min(img_width-1, points_2d[:, 0].max())
    y_min = max(0, points_2d[:, 1].min())
    y_max = min(img_height-1, points_2d[:, 1].max())

    # Verifica se bbox é válida
    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max

def get_3d_bbox_corners(center, size):
    """Gera os 8 cantos de uma bounding box 3D"""
    w, h, d = size['width'], size['height'], size['depth']
    x, y, z = center

    # 8 cantos da caixa
    corners = np.array([
        [x - w/2, y - h/2, z - d/2],
        [x + w/2, y - h/2, z - d/2],
        [x - w/2, y + h/2, z - d/2],
        [x + w/2, y + h/2, z - d/2],
        [x - w/2, y - h/2, z + d/2],
        [x + w/2, y - h/2, z + d/2],
        [x - w/2, y + h/2, z + d/2],
        [x + w/2, y + h/2, z + d/2]
    ])

    return corners

def save_yolo_label(label_path, bboxes_2d, img_width, img_height):
    """Salva labels no formato YOLO"""
    with open(label_path, 'w') as f:
        for bbox in bboxes_2d:
            class_id = bbox['class_id']
            x_min, y_min, x_max, y_max = bbox['bbox']

            # Converte para formato YOLO (centro normalizado + largura/altura)
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def save_3d_label(label_path, bboxes_3d):
    """Salva labels 3D no formato KITTI-like"""
    with open(label_path, 'w') as f:
        for bbox in bboxes_3d:
            # Formato: class x y z w h d yaw
            f.write(f"{bbox['class']} ")
            f.write(f"{bbox['center'][0]:.3f} {bbox['center'][1]:.3f} {bbox['center'][2]:.3f} ")
            f.write(f"{bbox['size']['width']:.3f} {bbox['size']['height']:.3f} {bbox['size']['depth']:.3f} ")
            f.write(f"{bbox.get('yaw', 0):.3f}\n")

# Loop principal
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
            print(f"   {cfg['vehicle']}: x={cfg['x']}, y={cfg['y']}, z={cfg['z']}")

    time.sleep(2)

    for yaw in yaw_angles:
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:06d}"

        # Pega posição do Ego
        ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
        ego_position = np.array([
            ego_pose.position.x_val,
            ego_pose.position.y_val,
            ego_pose.position.z_val
        ])

        # Captura RGB + Depth
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

            # Redimensiona se necessário
            if depth_array.shape != (image_height, image_width):
                depth_array = cv2.resize(depth_array, (image_width, image_height))

        if img_bgr is not None and depth_array is not None:
            # Pega posições reais dos drones
            drone_positions = get_drone_actual_positions(client, vehicles)

            bboxes_2d = []  # Para YOLO
            bboxes_3d = []  # Para PointNet

            # Cria visualização
            vis_2d = img_bgr.copy()

            # Processa cada drone
            for cfg in scenario["drone_configs"]:
                vehicle_name = cfg["vehicle"]

                if vehicle_name in drone_positions:
                    drone_pos = drone_positions[vehicle_name]

                    # Posição relativa ao Ego
                    relative_pos = np.array([
                        drone_pos["x"] - ego_position[0],
                        drone_pos["y"] - ego_position[1],
                        drone_pos["z"] - ego_position[2]
                    ])

                    # Só processa se drone está na frente
                    if relative_pos[0] > 0:
                        # Gera bounding box 3D
                        corners_3d = get_3d_bbox_corners(relative_pos, DRONE_SIZE)

                        # Projeta para 2D
                        bbox_2d = get_bbox_2d_from_3d(corners_3d, K, image_width, image_height)

                        if bbox_2d is not None:
                            x_min, y_min, x_max, y_max = bbox_2d

                            # Adiciona à lista YOLO
                            bboxes_2d.append({
                                'class_id': int(class_ids[cfg["class"]]),
                                'class_name': cfg["class"],
                                'bbox': [int(x) for x in bbox_2d],  # Converte para int Python
                                'vehicle': vehicle_name
                            })

                            # Adiciona à lista 3D
                            bboxes_3d.append({
                                'class': cfg["class"],
                                'center': relative_pos.tolist(),
                                'size': DRONE_SIZE,
                                'vehicle': vehicle_name
                            })

                            # Desenha na visualização
                            color = (0, 255, 0) if cfg["class"] == "drone" else (0, 0, 255)
                            cv2.rectangle(vis_2d, (x_min, y_min), (x_max, y_max), color, 2)
                            cv2.putText(vis_2d, f"{cfg['class']}",
                                      (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, color, 1)

            # Salva imagem RGB (para YOLO)
            cv2.imwrite(str(yolo_dir / "images" / f"{frame_name}.jpg"), img_bgr)

            # Salva label YOLO
            if len(bboxes_2d) > 0:
                save_yolo_label(
                    yolo_dir / "labels" / f"{frame_name}.txt",
                    bboxes_2d, image_width, image_height
                )

            # Gera e salva point cloud
            # Converte depth para point cloud
            h, w = depth_array.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            valid_mask = (depth_array > 0.1) & (depth_array < 100)

            z = depth_array[valid_mask]
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]

            # Backproject para 3D
            x = (u_valid - cx) * z / fx
            y = (v_valid - cy) * z / fy

            point_cloud = np.stack([x, y, z], axis=-1)

            # Salva point cloud
            np.save(pointnet_dir / "point_clouds" / f"{frame_name}.npy", point_cloud)

            # Salva label 3D
            if len(bboxes_3d) > 0:
                save_3d_label(
                    pointnet_dir / "labels_3d" / f"{frame_name}.txt",
                    bboxes_3d
                )

            # Salva visualização
            cv2.imwrite(str(vis_dir / "bbox_2d" / f"{frame_name}.jpg"), vis_2d)

            # Salva metadados completos
            metadata = {
                "frame_id": int(total_frames),
                "scenario": scenario['name'],
                "yaw": float(yaw),
                "ego_position": ego_position.tolist(),
                "camera_matrix": K.tolist(),
                "image_size": [int(image_width), int(image_height)],
                "bboxes_2d": bboxes_2d,
                "bboxes_3d": bboxes_3d,
                "num_points": int(len(point_cloud))
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

            if total_frames % 10 == 0:
                print(f"   📊 Progresso: {total_frames}/{max_frames} frames")

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

# Cria arquivo de configuração YOLO
yolo_yaml = {
    "path": str(yolo_dir.absolute()),
    "train": "images",
    "val": "images",
    "names": {i: name for i, name in enumerate(class_names)}
}

with open(yolo_dir / "dataset.yaml", 'w') as f:
    import yaml
    yaml.dump(yolo_yaml, f)

print("\n" + "="*70)
print("🎉 DATASETS COMPLETOS!")
print("="*70)
print(f"\n📊 Estatísticas:")
print(f"   • Frames gerados: {total_frames}")
print(f"\n📁 Estrutura gerada:")
print(f"   {output_dir}/")
print(f"   ├── yolo_dataset/        (Para treinar YOLO)")
print(f"   │   ├── images/          (Imagens RGB)")
print(f"   │   ├── labels/          (Bounding boxes 2D)")
print(f"   │   └── dataset.yaml     (Config YOLO)")
print(f"   ├── pointnet_dataset/    (Para treinar PointNet)")
print(f"   │   ├── point_clouds/    (Nuvens de pontos .npy)")
print(f"   │   ├── labels_3d/       (Bounding boxes 3D)")
print(f"   │   └── frustums/        (Para frustum extraction)")
print(f"   └── metadata/            (Sincronização 2D↔3D)")
print(f"\n🔥 Pipeline Frustum-PointNet pronto!")
print(f"   1. Treine YOLO com yolo_dataset/")
print(f"   2. Use detecções 2D para criar frustums")
print(f"   3. Treine PointNet com frustums extraídos")