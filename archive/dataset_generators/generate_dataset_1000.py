#!/usr/bin/env python3
"""ARCHIVED. Dataset generator, 1000+ frames.

Generates a large dataset with many variations for YOLO and PointNet. The
1,000-frame-per-scene precursor of the canonical collector; kept because
docs/guides/urban-dataset.md still refers to it.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys
import random

print("\n" + "="*70)
print("DATASET GENERATOR - 1000+ FRAMES")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"Connected. Vehicles: {vehicles}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\nTaking off all the drones...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Directory layout
output_dir = Path("dataset_final_1000_frames")
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

# Visualisations (only a few, to save space)
vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)

# Metadata
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\nThe dataset will be written to: {output_dir.absolute()}")

# Camera parameters
image_width = 1280
image_height = 720
FOV_H = 90
FOV_V = 60

fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = image_height / (2 * np.tan(np.radians(FOV_V/2)))
cx = image_width / 2
cy = image_height / 2

# DETECTION-API CONFIGURATION
camera_name = "front_center"
image_type = airsim.ImageType.Scene
detection_radius_cm = 15000  # 150 m, to detect farther away
drone_mesh_names = ["Drone3", "Drone4", "Intruder1"]

print(f"\nDetection configuration:")
print(f"   - camera: {camera_name}")
print(f"   • Detectando: {drone_mesh_names}")
print(f"   • Raio: {detection_radius_cm/100}m")

# Configure the detection
client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera_name, image_type, detection_radius_cm, vehicle_name="Ego")

for drone_name in drone_mesh_names:
    if drone_name in vehicles:
        client.simAddDetectionFilterMeshName(camera_name, image_type, drone_name, vehicle_name="Ego")

# MANY MORE SCENARIOS AND VARIATIONS
scenarios = []

# Scenarios at varying ranges
distances = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
heights = [-8, -10, -12, -15, -18, -20, -25, -30]

for dist_idx, base_dist in enumerate(distances):
    for height_idx, ego_height in enumerate(heights):
        scenario_name = f"dist_{base_dist}m_height_{-ego_height}m"

        # Vary the drone positions
        angle_variation = random.uniform(-30, 30)

        scenario = {
            "name": scenario_name,
            "ego_height": ego_height,
            "drones": []
        }

        # Drone3 - vary the position
        x3 = base_dist + random.uniform(-3, 3)
        y3 = random.uniform(-base_dist/2, -2)
        z3 = ego_height + random.uniform(-3, 3)
        scenario["drones"].append({"vehicle": "Drone3", "x": x3, "y": y3, "z": z3})

        # Drone4 - vary the position
        x4 = base_dist + random.uniform(-2, 5)
        y4 = random.uniform(-2, base_dist/2)
        z4 = ego_height + random.uniform(-2, 4)
        scenario["drones"].append({"vehicle": "Drone4", "x": x4, "y": y4, "z": z4})

        # Intruder1 - vary the position
        x1 = base_dist + random.uniform(0, 8)
        y1 = random.uniform(-3, 3)
        z1 = ego_height + random.uniform(-4, 2)
        scenario["drones"].append({"vehicle": "Intruder1", "x": x1, "y": y1, "z": z1})

        scenarios.append(scenario)

# Add formation scenarios
formations = [
    "linha_horizontal", "linha_vertical", "triangulo", "diagonal",
    "v_formation", "random_cluster", "wide_spread", "tight_group"
]

for formation in formations:
    for dist in [10, 15, 20, 25, 30]:
        scenario = {
            "name": f"formation_{formation}_{dist}m",
            "ego_height": random.choice(heights),
            "drones": []
        }

        if formation == "linha_horizontal":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist, "y": -5, "z": scenario["ego_height"]},
                {"vehicle": "Drone4", "x": dist, "y": 0, "z": scenario["ego_height"]},
                {"vehicle": "Intruder1", "x": dist, "y": 5, "z": scenario["ego_height"]}
            ]
        elif formation == "linha_vertical":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist, "y": 0, "z": scenario["ego_height"] + 3},
                {"vehicle": "Drone4", "x": dist, "y": 0, "z": scenario["ego_height"]},
                {"vehicle": "Intruder1", "x": dist, "y": 0, "z": scenario["ego_height"] - 3}
            ]
        elif formation == "triangulo":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist, "y": -3, "z": scenario["ego_height"] - 2},
                {"vehicle": "Drone4", "x": dist, "y": 3, "z": scenario["ego_height"] - 2},
                {"vehicle": "Intruder1", "x": dist + 3, "y": 0, "z": scenario["ego_height"] + 2}
            ]
        elif formation == "diagonal":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist - 3, "y": -3, "z": scenario["ego_height"] + 2},
                {"vehicle": "Drone4", "x": dist, "y": 0, "z": scenario["ego_height"]},
                {"vehicle": "Intruder1", "x": dist + 3, "y": 3, "z": scenario["ego_height"] - 2}
            ]
        elif formation == "v_formation":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist - 2, "y": -4, "z": scenario["ego_height"]},
                {"vehicle": "Drone4", "x": dist + 2, "y": 0, "z": scenario["ego_height"] + 1},
                {"vehicle": "Intruder1", "x": dist - 2, "y": 4, "z": scenario["ego_height"]}
            ]
        elif formation == "random_cluster":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist + random.uniform(-3, 3),
                 "y": random.uniform(-5, 5), "z": scenario["ego_height"] + random.uniform(-3, 3)},
                {"vehicle": "Drone4", "x": dist + random.uniform(-3, 3),
                 "y": random.uniform(-5, 5), "z": scenario["ego_height"] + random.uniform(-3, 3)},
                {"vehicle": "Intruder1", "x": dist + random.uniform(-3, 3),
                 "y": random.uniform(-5, 5), "z": scenario["ego_height"] + random.uniform(-3, 3)}
            ]
        elif formation == "wide_spread":
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist, "y": -10, "z": scenario["ego_height"]},
                {"vehicle": "Drone4", "x": dist + 5, "y": 0, "z": scenario["ego_height"] + 2},
                {"vehicle": "Intruder1", "x": dist - 2, "y": 8, "z": scenario["ego_height"] - 1}
            ]
        else:  # tight_group
            scenario["drones"] = [
                {"vehicle": "Drone3", "x": dist, "y": -1, "z": scenario["ego_height"]},
                {"vehicle": "Drone4", "x": dist + 1, "y": 0, "z": scenario["ego_height"] + 1},
                {"vehicle": "Intruder1", "x": dist, "y": 1, "z": scenario["ego_height"] - 1}
            ]

        scenarios.append(scenario)

# Many yaw angles per scenario
yaw_angles = list(range(-45, 46, 3))  # -45 to 45 deg in 3 deg steps = 31 angles

total_frames = 0
max_frames = 1500  # target 1500 frames, to guarantee 1000+ with labels

def depth_to_pointcloud(depth_array, fx, fy, cx, cy):
    """Convert a depth map into a point cloud."""
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    valid = (depth_array > 0.1) & (depth_array < 100)

    z = depth_array[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    return np.stack([x, y, z], axis=-1)

print(f"\nConfiguration:")
print(f"   - {len(scenarios)} scenarios")
print(f"   - {len(yaw_angles)} angles per scenario")
print(f"   • Potencial de {len(scenarios) * len(yaw_angles)} frames")
print(f"   • Meta: {max_frames} frames")

# Shuffle the scenarios for variety
random.shuffle(scenarios)

# Contadores
frames_with_detections = 0
frames_without_detections = 0

for scenario_idx, scenario in enumerate(scenarios):
    if total_frames >= max_frames:
        break

    if scenario_idx % 10 == 0:  # report progress every ten scenarios
        print(f"\nScenario {scenario_idx+1}/{len(scenarios)}: {scenario['name'][:30]}")
        print(f"   Progresso total: {total_frames}/{max_frames} frames")
        print(f"   with detections: {frames_with_detections}, without: {frames_without_detections}")

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

    time.sleep(1)  # reduced, to speed things up

    # Randomly select a subset of angles for this scenario
    angles_for_scenario = random.sample(yaw_angles, min(5, len(yaw_angles)))

    for yaw in angles_for_scenario:
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.3)

        frame_name = f"frame_{total_frames:06d}"

        # Captura RGB e Depth
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
        ], vehicle_name="Ego")

        # Process the RGB image
        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Process the depth image
        depth_array = None
        if responses[1].image_data_float:
            depth_1d = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_array = depth_1d.reshape(responses[1].height, responses[1].width)

            if depth_array.shape != (image_height, image_width):
                depth_array = cv2.resize(depth_array, (image_width, image_height))

        if img_bgr is not None and depth_array is not None:
            # Get the detections
            detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

            if detections:
                frames_with_detections += 1

                # Train / validation split
                is_train = total_frames % 5 != 0
                split = "train" if is_train else "val"

                # Labels YOLO
                yolo_labels = []

                # 3D labels
                labels_3d = []

                # Build a visualisation for a few frames only
                vis = None
                if total_frames % 50 == 0:  # a visualisation every 50 frames
                    vis = img_bgr.copy()

                for det in detections:
                    if hasattr(det, 'box2D'):
                        try:
                            # Extrai coordenadas
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

                            # Garante limites
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(image_width-1, x_max)
                            y_max = min(image_height-1, y_max)

                            # For YOLO
                            x_center = (x_min + x_max) / 2 / image_width
                            y_center = (y_min + y_max) / 2 / image_height
                            width = (x_max - x_min) / image_width
                            height = (y_max - y_min) / image_height

                            if 0 < width < 1 and 0 < height < 1:
                                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                                # Estimate the 3D position
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
                                        "size": [1.0, 0.5, 1.0],
                                        "bbox_2d": [x_min, y_min, x_max, y_max],
                                        "name": det.name if hasattr(det, 'name') else "unknown"
                                    })

                                    # Draw the visualisation
                                    if vis is not None:
                                        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                        cv2.putText(vis, f"D:{z_3d:.1f}m",
                                                   (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                                                   0.5, (0, 255, 0), 1)

                        except Exception as e:
                            if total_frames % 100 == 0:  # report errors occasionally
                                print(f"   error while processing a detection: {e}")

                # Save only when there are labels
                if yolo_labels:
                    # Save the RGB image
                    cv2.imwrite(str(yolo_dir / "images" / split / f"{frame_name}.jpg"), img_bgr)

                    # Save the YOLO label
                    with open(yolo_dir / "labels" / split / f"{frame_name}.txt", 'w') as f:
                        f.write('\n'.join(yolo_labels))

                    # Generate and save the point cloud
                    point_cloud = depth_to_pointcloud(depth_array, fx, fy, cx, cy)
                    np.save(pointnet_dir / "point_clouds" / f"{frame_name}.npy", point_cloud)

                    # Save the 3D labels
                    if labels_3d:
                        with open(pointnet_dir / "labels_3d" / f"{frame_name}.json", 'w') as f:
                            json.dump(labels_3d, f, indent=2)

                    # Save the visualisation (a few frames only)
                    if vis is not None:
                        cv2.imwrite(str(vis_dir / f"{frame_name}.jpg"), vis)

                    # Metadata (only for some frames, to save space)
                    if total_frames % 10 == 0:
                        metadata = {
                            "frame_id": int(total_frames),
                            "scenario": scenario['name'],
                            "yaw": float(yaw),
                            "split": split,
                            "num_detections": len(detections),
                            "num_labels": len(yolo_labels)
                        }

                        with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                            json.dump(metadata, f, indent=2)
            else:
                frames_without_detections += 1

            total_frames += 1

            if total_frames % 100 == 0:
                print(f"\n    Milestone: {total_frames} frames processados")
                print(f"      with detections: {frames_with_detections}")
                print(f"      success rate: {frames_with_detections/total_frames*100:.1f}%")

# Pousa drones
print("\nLanding...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

# Write dataset.yaml for YOLO
dataset_yaml = f"""
path: {yolo_dir.absolute()}
train: images/train
val: images/val

nc: 1
names: ['drone']
"""

with open(yolo_dir / "dataset.yaml", 'w') as f:
    f.write(dataset_yaml)

# Final statistics
num_train = len(list((yolo_dir / "labels" / "train").glob("*.txt")))
num_val = len(list((yolo_dir / "labels" / "val").glob("*.txt")))

print("\n" + "="*70)
print("DATASET 1000+ COMPLETO!")
print("="*70)
print(f"\nFinal statistics:")
print(f"   • Total de frames processados: {total_frames}")
print(f"   - frames with detections: {frames_with_detections}")
print(f"   - success rate: {frames_with_detections/total_frames*100:.1f}%")
print(f"   - training images: {num_train}")
print(f"   - validation images: {num_val}")
print(f"   - total with labels: {num_train + num_val}")
print(f"\n Estrutura:")
print(f"   {output_dir}/")
print(f"   ├── yolo/")
print(f"   |   +-- images/train/ ({num_train} images)")
print(f"   |   +-- images/val/ ({num_val} images)")
print(f"   │   ├── labels/train/")
print(f"   │   ├── labels/val/")
print(f"   │   └── dataset.yaml")
print(f"   ├── pointnet/")
print(f"   |   +-- point_clouds/ ({num_train + num_val} .npy files)")
print(f"   │   └── labels_3d/")
print(f"   +-- visualizations/ (samples)")
print(f"\nDataset ready for training YOLO and PointNet")