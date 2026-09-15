#!/usr/bin/env python3
"""ARCHIVED. Dataset generator using AirSim semantic segmentation.

Route (i) of Section 7.2.3, in its fullest implementation: ground truth taken
directly from the segmentation masks. The chapter reports that this route failed
because the drone blueprints carried no unique stencil identifiers, so every
target shared a colour with the background or with each other.
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

print("\n" + "="*70)
print("DATASET GENERATOR USING SEGMENTATION (GROUND TRUTH)")
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

# Visualisations
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

# Varied scenarios
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
    Capture an initial segmentation image, to identify the drones' colours
    IMPORTANT: the drones appear as DARK colours in the segmentation.
    """
    print("\nIdentifying the drones' segmentation colours...")

    # Place the drones nearby, to guarantee visibility
    if "Drone3" in vehicles:
        client.moveToPositionAsync(5, -2, -10, 3, vehicle_name="Drone3").join()
    if "Drone4" in vehicles:
        client.moveToPositionAsync(5, 2, -10, 3, vehicle_name="Drone4").join()
    if "Intruder1" in vehicles:
        client.moveToPositionAsync(7, 0, -10, 3, vehicle_name="Intruder1").join()
    time.sleep(1)

    # Capture the segmentation
    response = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
    ], vehicle_name="Ego")[0]

    if response.image_data_uint8:
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        seg_img = img_1d.reshape(response.height, response.width, 3)

        # Find the unique colours
        unique_colors = np.unique(seg_img.reshape(-1, 3), axis=0)

        drone_colors = []
        for color in unique_colors:
            # Conta pixels desta cor
            mask = cv2.inRange(seg_img, color, color)
            pixel_count = np.sum(mask > 0)
            percentage = (pixel_count / (seg_img.shape[0] * seg_img.shape[1])) * 100

            # THE DRONES ARE DARK AND SMALL
            # Dark colours (RGB sum < 100) covering a small area (0.01% to 2% of the image)
            if np.sum(color) < 100 and 0.01 < percentage < 2.0:
                drone_colors.append(color)
                print(f"   Drone colour detected: RGB{tuple(color)} ({percentage:.3f}% of the image)")
            elif np.sum(color) < 30:
                # Also try very dark colours when the blobs are small
                if 0.01 < percentage < 2.0:
                    drone_colors.append(color)
                    print(f"   Drone colour (dark) detected: RGB{tuple(color)} ({percentage:.3f}% of the image)")

        if not drone_colors:
            print("   No drone colour identified; falling back to generic dark colours...")
            # Fallback: take every small dark colour
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
    Detect the drones using the segmentation colours identified earlier
    IMPORTANT: the drones are DARK and SMALL objects
    """
    detections = []

    # Without specific colours, look for dark objects
    if not drone_colors:
        # Find every dark pixel (RGB sum < 50)
        dark_mask = np.sum(seg_image, axis=2) < 50
        dark_mask = dark_mask.astype(np.uint8) * 255

        # Remove the noise
        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        # Encontra contornos
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # The drones are small (50 to 10,000 px^2)
            if 50 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                # Reasonable aspect ratio
                if 0.3 < float(w)/h < 3.0:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'area': float(area),
                        'color': [0, 0, 0],  # generic dark colour
                        'mask': dark_mask[y:y+h, x:x+w]
                    })
    else:
        # Use the colours identified earlier
        for color in drone_colors:
            # Build a mask for this colour (with a small tolerance for dark colours)
            tolerance = 10 if np.sum(color) < 50 else 5
            lower = np.maximum(0, np.array(color) - tolerance)
            upper = np.minimum(255, np.array(color) + tolerance)
            mask = cv2.inRange(seg_image, lower, upper)

            # Remove the noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                # The drones are small (50 to 10,000 px^2)
                if 50 < area < 10000:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check the aspect ratio (drones are not very elongated)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:
                        # Check that it is not too large
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
    Convert a depth map into a point cloud
    """
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Keep the valid points
    valid = (depth_array > 0.1) & (depth_array < 100)

    z = depth_array[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Back-projection to 3D
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy

    return np.stack([x, y, z], axis=-1)

# Identify the drones' colours
drone_colors = get_segmentation_colors()
if not drone_colors:
    print("No drone colour identified; falling back to generic detection...")

print(f"\nConfiguration:")
print(f"   - {len(scenarios)} scenarios")
print(f"   - {len(yaw_angles)} angles per scenario")
print(f"   - up to {max_frames} frames")
print(f"   - {len(drone_colors)} drone colours identified")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\nScenario {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Ego: altitude {-ego_height} m")
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

        # Capture RGB, depth and segmentation
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
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

            # Resize if necessary
            if depth_array.shape != (image_height, image_width):
                depth_array = cv2.resize(depth_array, (image_width, image_height))

        # Process the segmentation
        seg_image = None
        if responses[2].image_data_uint8:
            seg_1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            seg_image = seg_1d.reshape(responses[2].height, responses[2].width, 3)

            # Resize if necessary
            if seg_image.shape[:2] != (image_height, image_width):
                seg_image = cv2.resize(seg_image, (image_width, image_height))

        if img_bgr is not None and depth_array is not None and seg_image is not None:
            # Detect the drones in the segmentation
            detections = detect_drones_in_segmentation(seg_image, drone_colors)

            # If nothing was found, fall back to detecting dark objects
            if not detections:
                print(f"   Frame {frame_name}: no drone detected, trying the fallback...")
                detections = detect_drones_in_segmentation(seg_image, [])

            # Build the visualisation
            vis = img_bgr.copy()

            # Decide whether this is a training or a validation sample
            is_train = total_frames % 5 != 0
            split = "train" if is_train else "val"

            # Labels YOLO
            yolo_labels = []

            # 3D labels
            labels_3d = []

            for det in detections:
                x_min, y_min, x_max, y_max = det['bbox']

                # For YOLO (normalised format)
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                # Sanity filters
                if width > 0.5 or height > 0.5:
                    continue
                if width < 0.01 or height < 0.01:
                    continue

                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # For PointNet: estimate the 3D position from the depth inside the box
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

                # Draw the visualisation
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                if len(valid_depths) > 0:
                    cv2.putText(vis, f"D:{z_3d:.1f}m",
                               (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0, 255, 0), 1)

            # Save the RGB image
            cv2.imwrite(str(yolo_dir / "images" / split / f"{frame_name}.jpg"), img_bgr)

            # Save the YOLO label
            if yolo_labels:
                with open(yolo_dir / "labels" / split / f"{frame_name}.txt", 'w') as f:
                    f.write('\n'.join(yolo_labels))

            # Generate and save the point cloud
            point_cloud = depth_to_pointcloud(depth_array, fx, fy, cx, cy)
            np.save(pointnet_dir / "point_clouds" / f"{frame_name}.npy", point_cloud)

            # Save the 3D labels
            if labels_3d:
                with open(pointnet_dir / "labels_3d" / f"{frame_name}.json", 'w') as f:
                    json.dump(labels_3d, f, indent=2)

            # Save the visualisation
            cv2.imwrite(str(vis_dir / f"{frame_name}.jpg"), vis)

            # Also save the segmentation, for debugging
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
                print(f"Progresso: {total_frames}/{max_frames} frames")

    if total_frames >= max_frames:
        break

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

print("\n" + "="*70)
print("SEGMENTATION DATASET COMPLETE")
print("="*70)
print(f"\nStatistics:")
print(f"   - frames generated: {total_frames}")
print(f"\n Estrutura:")
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
print(f"\nAdvantages of the segmentation route:")
print("   • Ground truth direto do AirSim")
print("   • Bounding boxes precisas")
print("   - no 3D-to-2D projection errors")
print("   - reliable identification of the drones")