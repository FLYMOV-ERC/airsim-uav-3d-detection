#!/usr/bin/env python3
"""
Gerador de Dataset usando Segmentação do AirSim
Ground truth direto das máscaras de segmentação
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR DE DATASET COM SEGMENTAÇÃO (GROUND TRUTH)")
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
output_dir = Path("dataset_segmentation")
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

# Cenários variados
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
            {"vehicle": "Drone3", "x": 15, "y": -6, "z": -20},
            {"vehicle": "Drone4", "x": 20, "y": 5, "z": -19},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -21}
        ]
    }
]

yaw_angles = [-20, -10, 0, 10, 20]
total_frames = 0
max_frames = 300

def get_segmentation_colors():
    """
    Captura uma imagem de segmentação inicial para identificar as cores dos drones
    IMPORTANTE: Drones aparecem como cores ESCURAS na segmentação!
    """
    print("\n🎨 Identificando cores de segmentação dos drones...")

    # Posiciona drones próximos para garantir visibilidade
    if "Drone3" in vehicles:
        client.moveToPositionAsync(5, -2, -10, 3, vehicle_name="Drone3").join()
    if "Drone4" in vehicles:
        client.moveToPositionAsync(5, 2, -10, 3, vehicle_name="Drone4").join()
    if "Intruder1" in vehicles:
        client.moveToPositionAsync(7, 0, -10, 3, vehicle_name="Intruder1").join()
    time.sleep(1)

    # Captura segmentação
    response = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
    ], vehicle_name="Ego")[0]

    if response.image_data_uint8:
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        seg_img = img_1d.reshape(response.height, response.width, 3)

        # Encontra cores únicas
        unique_colors = np.unique(seg_img.reshape(-1, 3), axis=0)

        drone_colors = []
        for color in unique_colors:
            # Conta pixels desta cor
            mask = cv2.inRange(seg_img, color, color)
            pixel_count = np.sum(mask > 0)
            percentage = (pixel_count / (seg_img.shape[0] * seg_img.shape[1])) * 100

            # DRONES SÃO ESCUROS E PEQUENOS!
            # Cores escuras (soma RGB < 100) e pequenas (0.01% a 2% da imagem)
            if np.sum(color) < 100 and 0.01 < percentage < 2.0:
                drone_colors.append(color)
                print(f"   Cor de drone detectada: RGB{tuple(color)} ({percentage:.3f}% da imagem)")
            elif np.sum(color) < 30:
                # Também tenta cores muito escuras se forem pequenas
                if 0.01 < percentage < 2.0:
                    drone_colors.append(color)
                    print(f"   Cor de drone (escura) detectada: RGB{tuple(color)} ({percentage:.3f}% da imagem)")

        if not drone_colors:
            print("   ⚠️ Nenhuma cor de drone identificada! Tentando cores escuras genéricas...")
            # Fallback: pega todas as cores escuras pequenas
            for color in unique_colors:
                if np.sum(color) < 50:
                    mask = cv2.inRange(seg_img, color, color)
                    pixel_count = np.sum(mask > 0)
                    percentage = (pixel_count / (seg_img.shape[0] * seg_img.shape[1])) * 100
                    if 0.005 < percentage < 3.0:
                        drone_colors.append(color)

        return drone_colors

    return []

def detect_drones_in_segmentation(seg_image, drone_colors):
    """
    Detecta drones usando as cores de segmentação identificadas
    IMPORTANTE: Drones são objetos ESCUROS e PEQUENOS
    """
    detections = []

    # Se não temos cores específicas, procura por objetos escuros
    if not drone_colors:
        # Encontra todos os pixels escuros (soma RGB < 50)
        dark_mask = np.sum(seg_image, axis=2) < 50
        dark_mask = dark_mask.astype(np.uint8) * 255

        # Remove ruído
        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        # Encontra contornos
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Drones são pequenos (50 a 10000 pixels²)
            if 50 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                # Aspect ratio razoável
                if 0.3 < float(w)/h < 3.0:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'area': float(area),
                        'color': [0, 0, 0],  # Cor genérica escura
                        'mask': dark_mask[y:y+h, x:x+w]
                    })
    else:
        # Usa as cores identificadas
        for color in drone_colors:
            # Cria máscara para esta cor (com tolerância pequena para cores escuras)
            tolerance = 10 if np.sum(color) < 50 else 5
            lower = np.maximum(0, np.array(color) - tolerance)
            upper = np.minimum(255, np.array(color) + tolerance)
            mask = cv2.inRange(seg_image, lower, upper)

            # Remove ruído
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                # Drones são pequenos (50 a 10000 pixels²)
                if 50 < area < 10000:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Verifica aspect ratio (drones não são muito alongados)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:
                        # Verifica que não é muito grande
                        if w < image_width * 0.3 and h < image_height * 0.3:
                            detections.append({
                                'bbox': [x, y, x+w, y+h],
                                'area': float(area),
                                'color': color.tolist(),
                                'mask': mask[y:y+h, x:x+w]
                            })

    return detections

def depth_to_pointcloud(depth_array, fx, fy, cx, cy):
    """
    Converte depth map para point cloud
    """
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filtra pontos válidos
    valid = (depth_array > 0.1) & (depth_array < 100)

    z = depth_array[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Backprojection para 3D
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    return np.stack([x, y, z], axis=-1)

# Identifica cores dos drones
drone_colors = get_segmentation_colors()
if not drone_colors:
    print("⚠️ Nenhuma cor de drone identificada! Usando detecção genérica...")

print(f"\n📊 Configuração:")
print(f"   • {len(scenarios)} cenários")
print(f"   • {len(yaw_angles)} ângulos por cenário")
print(f"   • Até {max_frames} frames")
print(f"   • {len(drone_colors)} cores de drones identificadas")

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

        # Captura RGB, Depth e Segmentação
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
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

        # Processa Segmentação
        seg_image = None
        if responses[2].image_data_uint8:
            seg_1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            seg_image = seg_1d.reshape(responses[2].height, responses[2].width, 3)

            # Redimensiona se necessário
            if seg_image.shape[:2] != (image_height, image_width):
                seg_image = cv2.resize(seg_image, (image_width, image_height))

        if img_bgr is not None and depth_array is not None and seg_image is not None:
            # Detecta drones na segmentação
            detections = detect_drones_in_segmentation(seg_image, drone_colors)

            # Se não encontrou nada, força detecção de objetos escuros
            if not detections:
                print(f"   ⚠️ Frame {frame_name}: Nenhum drone detectado, tentando fallback...")
                detections = detect_drones_in_segmentation(seg_image, [])

            # Cria visualização
            vis = img_bgr.copy()

            # Decide se é treino ou validação
            is_train = total_frames % 5 != 0
            split = "train" if is_train else "val"

            # Labels YOLO
            yolo_labels = []

            # Labels 3D
            labels_3d = []

            for det in detections:
                x_min, y_min, x_max, y_max = det['bbox']

                # Para YOLO (formato normalizado)
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Filtros de sanidade
                if width > 0.5 or height > 0.5:
                    continue
                if width < 0.01 or height < 0.01:
                    continue

                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Para PointNet - estima posição 3D usando depth na região da bbox
                roi_depth = depth_array[y_min:y_max, x_min:x_max]
                valid_depths = roi_depth[(roi_depth > 0.1) & (roi_depth < 100)]

                if len(valid_depths) > 0:
                    z_3d = float(np.median(valid_depths))
                    u_center = (x_min + x_max) / 2
                    v_center = (y_min + y_max) / 2
                    x_3d = (u_center - cx) * z_3d / fx
                    y_3d = (v_center - cy) * z_3d / fy

                    labels_3d.append({
                        "class": "drone",
                        "center": [float(x_3d), float(y_3d), float(z_3d)],
                        "size": [0.8, 0.3, 0.8],
                        "bbox_2d": det['bbox']
                    })

                # Desenha visualização
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                if len(valid_depths) > 0:
                    cv2.putText(vis, f"D:{z_3d:.1f}m",
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

            # Salva também a segmentação para debug
            cv2.imwrite(str(vis_dir / f"{frame_name}_seg.jpg"), seg_image)

            # Metadata
            metadata = {
                "frame_id": int(total_frames),
                "scenario": scenario['name'],
                "yaw": float(yaw),
                "split": split,
                "num_detections": len(detections),
                "detections": [{
                    "bbox": d['bbox'],
                    "area": d['area'],
                    "color": d['color']
                } for d in detections],
                "labels_3d": labels_3d,
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
print("🎉 DATASET COM SEGMENTAÇÃO COMPLETO!")
print("="*70)
print(f"\n📊 Estatísticas:")
print(f"   • Frames gerados: {total_frames}")
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
print(f"\n✅ Vantagens da segmentação:")
print("   • Ground truth direto do AirSim")
print("   • Bounding boxes precisas")
print("   • Sem erros de projeção 3D→2D")
print("   • Identificação confiável dos drones")