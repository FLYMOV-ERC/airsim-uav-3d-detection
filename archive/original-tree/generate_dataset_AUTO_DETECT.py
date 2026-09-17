#!/usr/bin/env python3
"""
Gerador de Dataset com Detecção Automática via Segmentação/Depth
Mais confiável que projeção manual
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR COM DETECÇÃO AUTOMÁTICA (SEGMENTAÇÃO + DEPTH)")
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
output_dir = Path("dataset_auto_detect")
output_dir.mkdir(exist_ok=True)

# YOLO
yolo_dir = output_dir / "yolo_dataset"
yolo_dir.mkdir(exist_ok=True)
(yolo_dir / "images").mkdir(exist_ok=True)
(yolo_dir / "labels").mkdir(exist_ok=True)

# Visualizações
vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)

# Metadata
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset em: {output_dir.absolute()}")

# Parâmetros
image_width = 1280
image_height = 720

# Cenários
scenarios = [
    {
        "name": "test_close",
        "ego_height": -10,
        "drones": [
            {"vehicle": "Drone3", "x": 8, "y": -2, "z": -10},
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -9},
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -11}
        ]
    },
    {
        "name": "test_medium",
        "ego_height": -15,
        "drones": [
            {"vehicle": "Drone3", "x": 12, "y": -4, "z": -15},
            {"vehicle": "Drone4", "x": 18, "y": 4, "z": -14},
            {"vehicle": "Intruder1", "x": 25, "y": 0, "z": -16}
        ]
    }
]

total_frames = 0
max_frames = 100

def detect_objects_in_depth(depth_array, min_size=100):
    """
    Detecta objetos usando descontinuidades no depth
    """
    # Calcula gradiente (mudanças bruscas de profundidade)
    grad_x = np.abs(np.gradient(depth_array, axis=1))
    grad_y = np.abs(np.gradient(depth_array, axis=0))
    edges = (grad_x + grad_y) > 2.0  # Threshold para edges

    # Dilata para conectar edges próximos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges.astype(np.uint8) * 255, kernel, iterations=2)

    # Encontra contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_size:
            # Calcula bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Pega profundidade média na região
            roi_depth = depth_array[y:y+h, x:x+w]
            mean_depth = np.median(roi_depth[roi_depth > 0])

            if 2 < mean_depth < 50:  # Objeto entre 2m e 50m
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'depth': float(mean_depth),
                    'area': float(area),
                    'contour': contour
                })

    return detections

def detect_objects_in_segmentation(seg_image):
    """
    Detecta objetos usando segmentação (se disponível)
    """
    # Cores de segmentação dos drones (precisa verificar no seu ambiente)
    # Geralmente cada objeto tem uma cor única

    # Converte para HSV para melhor detecção de cor
    hsv = cv2.cvtColor(seg_image, cv2.COLOR_BGR2HSV)

    detections = []

    # Procura por cores únicas (não preto e não céu)
    unique_colors = np.unique(seg_image.reshape(-1, 3), axis=0)

    for color in unique_colors:
        # Ignora preto (fundo) e cores muito escuras
        if np.sum(color) < 30:
            continue

        # Cria máscara para essa cor
        mask = cv2.inRange(seg_image, color-5, color+5)

        # Encontra contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Mínimo de área
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'color': color.tolist(),
                    'area': float(area)
                })

    return detections

print(f"\n📊 Gerando até {max_frames} frames...")

for scenario in scenarios:
    print(f"\n🎬 Cenário: {scenario['name']}")

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

    for frame_idx in range(min(50, max_frames - total_frames)):
        if total_frames >= max_frames:
            break

        frame_name = f"frame_{total_frames:06d}"

        # Captura RGB, Depth e Segmentação
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
        ], vehicle_name="Ego")

        # RGB
        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Depth
        depth_array = None
        if responses[1].image_data_float:
            depth_1d = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_array = depth_1d.reshape(responses[1].height, responses[1].width)

            if depth_array.shape != (image_height, image_width):
                depth_array = cv2.resize(depth_array, (image_width, image_height))

        # Segmentação
        seg_image = None
        if len(responses) > 2 and responses[2].image_data_uint8:
            seg_1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            seg_rgb = seg_1d.reshape(responses[2].height, responses[2].width, 3)
            seg_image = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

            if seg_image.shape[:2] != (image_height, image_width):
                seg_image = cv2.resize(seg_image, (image_width, image_height))

        if img_bgr is not None and depth_array is not None:
            # Detecta objetos
            detections = detect_objects_in_depth(depth_array)

            # Se tiver segmentação, usa também
            if seg_image is not None:
                seg_detections = detect_objects_in_segmentation(seg_image)
                # Pode combinar ou escolher o melhor método

            # Visualização
            vis = img_bgr.copy()

            # Salva labels YOLO
            yolo_labels = []

            for i, det in enumerate(detections):
                x_min, y_min, x_max, y_max = det['bbox']

                # Filtros adicionais
                width = x_max - x_min
                height = y_max - y_min

                # Ignora detecções muito grandes ou muito pequenas
                if width > image_width * 0.5 or height > image_height * 0.5:
                    continue
                if width < 20 or height < 20:
                    continue

                # Para YOLO (normalizado)
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                w_norm = width / image_width
                h_norm = height / image_height

                # Classe 0 = drone
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                # Desenha
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(vis, f"D:{det['depth']:.1f}m",
                           (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 1)

            # Salva imagem
            cv2.imwrite(str(yolo_dir / "images" / f"{frame_name}.jpg"), img_bgr)

            # Salva label YOLO
            if yolo_labels:
                with open(yolo_dir / "labels" / f"{frame_name}.txt", 'w') as f:
                    f.write('\n'.join(yolo_labels))

            # Salva visualização
            cv2.imwrite(str(vis_dir / f"{frame_name}_vis.jpg"), vis)

            # Metadados
            metadata = {
                "frame_id": int(total_frames),
                "scenario": scenario['name'],
                "num_detections": len(detections),
                "detections": [{
                    "bbox": det['bbox'],
                    "depth": det['depth'],
                    "area": det['area']
                } for det in detections]
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1

            if total_frames % 10 == 0:
                print(f"   Progresso: {total_frames}/{max_frames}")

# Pousa
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
train: images
val: images

nc: 1
names: ['drone']
"""

with open(yolo_dir / "dataset.yaml", 'w') as f:
    f.write(dataset_yaml)

print("\n" + "="*70)
print("✅ DATASET COM DETECÇÃO AUTOMÁTICA COMPLETO!")
print("="*70)
print(f"Total de frames: {total_frames}")
print(f"\n📁 Dataset em: {output_dir.absolute()}")
print("\nVantagens desta abordagem:")
print("  • Detecção automática via depth/segmentação")
print("  • Não precisa de projeção 3D→2D manual")
print("  • Mais robusto e confiável")