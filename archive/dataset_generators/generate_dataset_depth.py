#!/usr/bin/env python3
"""ARCHIVED. Dataset generator using the DEPTH CAMERA.

This is the step at which the depth camera displaced the LiDAR as the primary
source of the working point cloud (Section 7.2.1). A distinct approach, kept as
its representative.
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
print("GENERATOR USING THE DEPTH CAMERA")
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

# Directories
output_dir = Path("dataset_depth")
output_dir.mkdir(exist_ok=True)
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_depth").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "depth_arrays").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\nThe dataset will be written to: {output_dir.absolute()}")

# Camera parameters
FOV_H = 90  # graus
FOV_V = 60  # estimated for 16:9
image_width = 1280
image_height = 720

fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = image_height / (2 * np.tan(np.radians(FOV_V/2)))
cx = image_width / 2
cy = image_height / 2

# Scenarios
scenarios = [
    {
        "name": "low_altitude",
        "ego_height": -10,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -10},
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -8},
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -12}
        ]
    },
    {
        "name": "medium_altitude",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 12, "y": -5, "z": -15},
            {"vehicle": "Drone4", "x": 18, "y": 5, "z": -13},
            {"vehicle": "Intruder1", "x": 25, "y": 0, "z": -17}
        ]
    },
    {
        "name": "high_altitude",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z": -22},
            {"vehicle": "Drone4", "x": 20, "y": 8, "z": -20},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -18}
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
    }
]

yaw_angles = [-30, -20, -10, 0, 10, 20, 30]
total_frames = 0
total_detections = 0
max_frames = 300
start_time = time.time()

print(f"\nConfiguration:")
print(f"   - {len(scenarios)} scenarios x {len(yaw_angles)} angles")
print(f"   - up to {max_frames} frames")
print(f"   - using the DEPTH CAMERA for exact detection")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\nScenario {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Moving the ego to {-ego_height} m altitude...")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        if cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(cfg["x"], cfg["y"], cfg["z"], 5,
                                      vehicle_name=cfg["vehicle"]).join()
            print(f"   {cfg['vehicle']}: x={cfg['x']}, y={cfg['y']}, altitude={-cfg['z']} m")

    time.sleep(2)  # settling time

    for yaw in yaw_angles:
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # Captura RGB e Depth simultaneamente
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
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

            # Clamp the range for plotting
            depth_array = np.clip(depth_array, 0, 100)

        if img_bgr is not None and depth_array is not None:
            h_rgb, w_rgb = img_bgr.shape[:2]
            h_depth, w_depth = depth_array.shape

            # Resize the depth to match the RGB image if necessary
            if h_depth != h_rgb or w_depth != w_rgb:
                depth_array = cv2.resize(depth_array, (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)

            h, w = h_rgb, w_rgb

            # Build the fusion image
            fusion = img_bgr.copy()

            # Convert the depth image into a 3D point cloud
            # Build the pixel-coordinate grid
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Mask of the valid (finite) points
            valid_mask = (depth_array > 0.1) & (depth_array < 100)

            # Compute the 3D coordinates of the valid points
            z = depth_array[valid_mask]
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]

            # Convert pixels into 3D coordinates
            x = (u_valid - cx) * z / fx
            y = (v_valid - cy) * z / fy

            # Subsample the points for plotting (too dense otherwise)
            sample_rate = 50  # show one point in every 50
            sample_indices = np.arange(0, len(z), sample_rate)

            z_sampled = z[sample_indices]
            u_sampled = u_valid[sample_indices]
            v_sampled = v_valid[sample_indices]

            # Detect nearby objects (candidate drones)
            frame_detections = 0
            detected_objects = []

            # Look for depth discontinuities (objects)
            depth_grad = np.gradient(depth_array)[0]
            edges = np.abs(depth_grad) > 2  # abrupt change in depth

            # Encontra contornos de objetos
            edges_uint8 = (edges * 255).astype(np.uint8)
            contours, _ = cv2.findContours(edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyse each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # discard small objects
                    # Compute the centre and the range
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx_obj = int(M["m10"] / M["m00"])
                        cy_obj = int(M["m01"] / M["m00"])

                        # Read the depth at the object's centre
                        if 0 <= cx_obj < w and 0 <= cy_obj < h:
                            obj_depth = depth_array[cy_obj, cx_obj]

                            if 3 < obj_depth < 50:  # object between 3 m and 50 m
                                # Draw the detection
                                cv2.drawContours(fusion, [contour], -1, (255, 255, 0), 2)
                                cv2.circle(fusion, (cx_obj, cy_obj), 5, (255, 0, 255), -1)
                                cv2.putText(fusion, f"{obj_depth:.1f}m",
                                          (cx_obj-20, cy_obj-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                                frame_detections += 1
                                detected_objects.append({
                                    "x": int(cx_obj),
                                    "y": int(cy_obj),
                                    "depth": float(obj_depth),
                                    "area": float(area)
                                })

            # Draw the depth points as a coloured overlay
            for i in range(len(z_sampled)):
                depth_val = z_sampled[i]
                u_pt = u_sampled[i]
                v_pt = v_sampled[i]

                # Colour by depth
                if depth_val < 10:  # near - red
                    color = (0, 0, 255)
                elif depth_val < 30:  # mid range - yellow
                    color = (0, 255, 255)
                elif depth_val < 50:  # Longe - verde
                    color = (0, 255, 0)
                else:  # very far - blue
                    color = (255, 0, 0)

                cv2.circle(fusion, (u_pt, v_pt), 1, color, -1)

            # Build the colourised depth visualisation
            depth_vis = cv2.applyColorMap(
                (255 * depth_array / depth_array.max()).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            # Add the information
            cv2.putText(fusion, f"{scenario['name']} | Alt:{-ego_height}m | Yaw:{yaw}° | F:{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Objetos detectados: {frame_detections}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, "DEPTH CAMERA",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save the images
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_depth" / f"{frame_name}.png"), depth_vis)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            # Save the depth array
            np.save(output_dir / "depth_arrays" / f"{frame_name}.npy", depth_array)

            # Metadata
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "scenario_idx": scenario_idx,
                "yaw": yaw,
                "ego_height": -ego_height,
                "detections": frame_detections,
                "detected_objects": detected_objects,
                "depth_min": float(depth_array.min()),
                "depth_max": float(depth_array.max()),
                "drone_positions": scenario["drone_configs"]
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1
            total_detections += frame_detections

            # Status
            if total_frames % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Progresso: {total_frames}/{max_frames}")
                print(f"      total detections: {total_detections}")

    if total_frames >= max_frames:
        break

# Land all the drones
print("\nLanding all the drones...")
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

print("\n" + "="*70)
print("DEPTH-CAMERA DATASET COMPLETE")
print("="*70)
print(f"\nFinal statistics:")
print(f"   - frames generated: {total_frames}")
print(f"   - total detections: {total_detections}")
print(f"   - mean: {total_detections/total_frames:.2f} objects/frame" if total_frames > 0 else "")
print(f"\n Vantagens da Depth Camera:")
print("   - detects EVERY object")
print("   - needs no configuration in Unreal")
print("   - gives the exact range of every pixel")
print("   - always works")
print(f"\nDataset saved to: {output_dir.absolute()}")