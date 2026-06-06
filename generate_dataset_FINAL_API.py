#!/usr/bin/env python3
"""
DATASET GENERATOR FINAL - USANDO API simGetDetections FUNCIONANDO!
Gera dataset para YOLO e PointNet com bounding boxes corretas
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 DATASET GENERATOR FINAL - API FUNCIONANDO!")
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
output_dir = Path("dataset_final_api")
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

# CONFIGURAÇÃO DA API DE DETECÇÃO - APENAS DRONES!
camera_name = "front_center"  # Nome da câmera
image_type = airsim.ImageType.Scene
detection_radius_cm = 10000  # 100 metros em centímetros

# Lista de mesh names EXATOS dos drones (descobertos via simListSceneObjects)
drone_mesh_names = ["Drone3", "Drone4", "Intruder1"]

print(f"\n🔧 Configuração de detecção:")
print(f"   • Câmera: {camera_name}")
print(f"   • Detectando APENAS drones: {drone_mesh_names}")
print(f"   • Raio: {detection_radius_cm/100}m")

# Configura detecção para detectar APENAS os drones
client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera_name, image_type, detection_radius_cm, vehicle_name="Ego")

# Adiciona cada drone especificamente
for drone_name in drone_mesh_names:
    if drone_name in vehicles:  # Só adiciona se o drone existe
        client.simAddDetectionFilterMeshName(camera_name, image_type, drone_name, vehicle_name="Ego")
        print(f"   ✅ Adicionado: {drone_name}")

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
    },
    {
        "name": "long_range",
        "ego_height": -20,
        "drones": [
            {"vehicle": "Drone3", "x": 20, "y": -6, "z": -20},
            {"vehicle": "Drone4", "x": 25, "y": 5, "z": -19},
            {"vehicle": "Intruder1", "x": 35, "y": 0, "z": -21}
        ]
    },
    {
        "name": "varied_heights",
        "ego_height": -15,
        "drones": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -10},
            {"vehicle": "Drone4", "x": 12, "y": 4, "z": -15},
            {"vehicle": "Intruder1", "x": 16, "y": 0, "z": -20}
        ]
    },
    {
        "name": "formation",
        "ego_height": -12,
        "drones": [
            {"vehicle": "Drone3", "x": 10, "y": -5, "z": -12},
            {"vehicle": "Drone4", "x": 10, "y": 0, "z": -12},
            {"vehicle": "Intruder1", "x": 10, "y": 5, "z": -12}
        ]
    }
]

yaw_angles = [-30, -15, 0, 15, 30]
total_frames = 0
max_frames = 500

def depth_to_pointcloud(depth_array, fx, fy, cx, cy):
    """Converte depth map para point cloud"""
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    valid = (depth_array > 0.1) & (depth_array < 100)

    z = depth_array[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    return np.stack([x, y, z], axis=-1)

def filter_drone_detections(detections):
    """
    Como agora detectamos APENAS drones, apenas valida as detecções
    """
    filtered = []

    for det in detections:
        if hasattr(det, 'box2D'):
            # Valida se a detecção tem dados válidos
            try:
                if hasattr(det.box2D.min, 'x_val'):
                    width = det.box2D.max.x_val - det.box2D.min.x_val
                    height = det.box2D.max.y_val - det.box2D.min.y_val
                else:
                    width = det.box2D.max.x - det.box2D.min.x
                    height = det.box2D.max.y - det.box2D.min.y

                # Validação básica - bbox tem tamanho razoável
                if width > 5 and height > 5 and width < image_width and height < image_height:
                    filtered.append(det)
                    if hasattr(det, 'name'):
                        print(f"      Detectado: {det.name}")

            except Exception as e:
                print(f"   ⚠️ Erro validando detecção: {e}")

    return filtered

print(f"\n📊 Configuração:")
print(f"   • {len(scenarios)} cenários")
print(f"   • {len(yaw_angles)} ângulos por cenário")
print(f"   • Até {max_frames} frames")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Ego: altura {-ego_height}m")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona drones
    for drone_cfg in scenario["drones"]:
        if drone_cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(
                drone_cfg["x"], drone_cfg["y"], drone_cfg["z"], 5,
                vehicle_name=drone_cfg["vehicle"]
            ).join()
            print(f"   {drone_cfg['vehicle']}: x={drone_cfg['x']}, y={drone_cfg['y']}, z={drone_cfg['z']}")

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
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
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
            # OBTER DETECÇÕES VIA API - JÁ RETORNA APENAS DRONES!
            detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

            # Valida as detecções (não precisa mais filtrar, já são só drones)
            detections = filter_drone_detections(detections) if detections else []

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
                if hasattr(det, 'box2D'):
                    try:
                        # Extrai coordenadas da bbox
                        if hasattr(det.box2D.min, 'x_val'):
                            x_min = int(det.box2D.min.x_val)
                            y_min = int(det.box2D.min.y_val)
                            x_max = int(det.box2D.max.x_val)
                            y_max = int(det.box2D.max.y_val)
                        else:
                            x_min = int(det.box2D.min.x)
                            y_min = int(det.box2D.min.y)
                            x_max = int(det.box2D.max.x)
                            y_max = int(det.box2D.max.y)

                        # Garante que está dentro dos limites
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(image_width-1, x_max)
                        y_max = min(image_height-1, y_max)

                        # Para YOLO (formato normalizado)
                        x_center = (x_min + x_max) / 2 / image_width
                        y_center = (y_min + y_max) / 2 / image_height
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height

                        # Validação final
                        if 0 < width < 1 and 0 < height < 1:
                            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                            # Estima posição 3D usando depth
                            roi_depth = depth_array[y_min:y_max, x_min:x_max]
                            valid_depths = roi_depth[(roi_depth > 0.1) & (roi_depth < 100)]

                            if len(valid_depths) > 0:
                                z_3d = float(np.median(valid_depths))
                                u_center = (x_min + x_max) / 2
                                v_center = (y_min + y_max) / 2
                                x_3d = (u_center - cx) * z_3d / fx
                                y_3d = (v_center - cy) * z_3d / fy

                                # Extrai informações 3D da API se disponível
                                if hasattr(det, 'box3D'):
                                    if hasattr(det.box3D, 'center'):
                                        center_3d = [det.box3D.center.x_val, det.box3D.center.y_val, det.box3D.center.z_val]
                                    else:
                                        center_3d = [float(x_3d), float(y_3d), float(z_3d)]

                                    if hasattr(det.box3D, 'halfExtents'):
                                        size_3d = [det.box3D.halfExtents.x_val*2, det.box3D.halfExtents.y_val*2, det.box3D.halfExtents.z_val*2]
                                    else:
                                        size_3d = [1.0, 0.5, 1.0]  # Tamanho estimado
                                else:
                                    center_3d = [float(x_3d), float(y_3d), float(z_3d)]
                                    size_3d = [1.0, 0.5, 1.0]

                                labels_3d.append({
                                    "class": "drone",
                                    "center": center_3d,
                                    "size": size_3d,
                                    "bbox_2d": [x_min, y_min, x_max, y_max],
                                    "name": det.name if hasattr(det, 'name') else "unknown"
                                })

                                # Desenha visualização
                                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                cv2.putText(vis, f"D:{z_3d:.1f}m",
                                           (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                                           0.5, (0, 255, 0), 1)

                    except Exception as e:
                        print(f"   ⚠️ Erro processando detecção: {e}")

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
                "total_detections": len(detections) if detections else 0,
                "filtered_detections": len(detections),
                "labels_saved": len(yolo_labels),
                "camera_params": {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(cx), "cy": float(cy),
                    "width": int(image_width), "height": int(image_height)
                }
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

            if total_frames % 10 == 0:
                print(f"   📊 Progresso: {total_frames}/{max_frames} frames")
                if len(yolo_labels) > 0:
                    print(f"      Última detecção: {len(yolo_labels)} drones")

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

# Estatísticas finais
num_train = len(list((yolo_dir / "labels" / "train").glob("*.txt")))
num_val = len(list((yolo_dir / "labels" / "val").glob("*.txt")))

print("\n" + "="*70)
print("🎉 DATASET COMPLETO COM API FUNCIONANDO!")
print("="*70)
print(f"\n📊 Estatísticas:")
print(f"   • Frames gerados: {total_frames}")
print(f"   • Frames com labels (treino): {num_train}")
print(f"   • Frames com labels (validação): {num_val}")
print(f"\n📁 Estrutura:")
print(f"   {output_dir}/")
print(f"   ├── yolo/")
print(f"   │   ├── images/train/")
print(f"   │   ├── images/val/")
print(f"   │   ├── labels/train/")
print(f"   │   ├── labels/val/")
print(f"   │   └── dataset.yaml")
print(f"   ├── pointnet/")
print(f"   │   ├── point_clouds/")
print(f"   │   └── labels_3d/")
print(f"   └── visualizations/")
print(f"\n✅ Pronto para treinar YOLO e PointNet!")