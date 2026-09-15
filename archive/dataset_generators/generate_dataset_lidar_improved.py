#!/usr/bin/env python3
"""ARCHIVED. Dataset generator with an improved LiDAR configuration.

The LiDAR-primary route after its vertical field of view was widened -- the
experiment Section 7.2.1 refers to when it says that "even after widening its
vertical field of view the beams that strike a distant drone are too few".
The last representative of the LiDAR-primary approach.
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

print("\n" + "="*70)
print("DATASET WITH THE IMPROVED LIDAR - WIDENED VERTICAL FOV")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\nConnected. Vehicles: {vehicles}")

# Arm all the drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola e aguarda
print("Taking off all the drones...")
takeoff_tasks = []
for v in vehicles:
    try:
        task = client.takeoffAsync(vehicle_name=v)
        takeoff_tasks.append(task)
    except:
        pass

for task in takeoff_tasks:
    try:
        task.join()
    except:
        pass

print("All airborne")
time.sleep(3)

# Output directory
output_dir = Path("dataset_lidar_improved")
output_dir.mkdir(exist_ok=True)
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\nThe dataset will be written to: {output_dir.absolute()}")

# Camera parameters
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360

# IMPORTANT: several sweeps at different pitch angles
# This simulates a LiDAR with a wider vertical FOV
LIDAR_PITCH_OFFSETS = [-0.4, -0.2, 0, 0.2, 0.4]  # Radianos

def capture_lidar_with_extended_fov(client, vehicle_name="Ego"):
    """
    Capture several LiDAR sweeps at different angles
    and combines them, to simulate a wider vertical FOV
    """
    all_points = []

    # Read the current orientation
    pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
    original_orientation = pose.orientation

    for pitch_offset in LIDAR_PITCH_OFFSETS:
        # Ajusta pitch temporariamente
        client.rotateByYawPitchRollAsync(0, pitch_offset, 0, vehicle_name=vehicle_name).join()
        time.sleep(0.1)  # Aguarda estabilizar

        # Captura LiDAR
        lidar_data = client.getLidarData("LidarFront", vehicle_name=vehicle_name)

        if lidar_data and len(lidar_data.point_cloud) > 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            # Compensate for the pitch rotation of the points
            cos_p = np.cos(-pitch_offset)
            sin_p = np.sin(-pitch_offset)

            # Inverse rotation, to compensate for the pitch
            X_comp = points[:, 0] * cos_p - points[:, 2] * sin_p
            Z_comp = points[:, 0] * sin_p + points[:, 2] * cos_p

            points_compensated = np.column_stack([X_comp, points[:, 1], Z_comp])
            all_points.append(points_compensated)

    # Restore the original orientation
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)

    # Combine all the points
    if all_points:
        combined_points = np.vstack(all_points)
        # Remove duplicatas aproximadas
        # Round to remove points that are too close together
        rounded = np.round(combined_points, decimals=1)
        unique_points = np.unique(rounded, axis=0)
        return unique_points
    else:
        return np.array([])

# Scenarios with drones at different relative altitudes
scenarios = [
    {
        "name": "mixed_heights",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": 0, "z": -8, "desc": "7 m above"},
            {"vehicle": "Drone4", "x": 15, "y": -5, "z": -15, "desc": "same altitude"},
            {"vehicle": "Intruder1", "x": 20, "y": 5, "z": -22, "desc": "7 m below"}
        ]
    },
    {
        "name": "high_targets",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 12, "y": -3, "z": -10, "desc": "10 m above"},
            {"vehicle": "Drone4", "x": 18, "y": 0, "z": -12, "desc": "8 m above"},
            {"vehicle": "Intruder1", "x": 25, "y": 3, "z": -15, "desc": "5 m above"}
        ]
    },
    {
        "name": "vertical_spread",
        "ego_height": -18,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 10, "y": 0, "z": -5, "desc": "13 m above"},
            {"vehicle": "Drone4", "x": 15, "y": -2, "z": -18, "desc": "same altitude"},
            {"vehicle": "Intruder1", "x": 20, "y": 2, "z": -30, "desc": "12 m below"}
        ]
    }
]

# Yaw angles
yaw_angles = [-20, -10, 0, 10, 20]

# Statistics
total_frames = 0
detections_high = 0  # above the horizon
detections_low = 0   # below the horizon

print(f"\nConfiguration:")
print(f"   - {len(scenarios)} scenarios")
print(f"   - {len(yaw_angles)} angles per scenario")
print(f"   - {len(LIDAR_PITCH_OFFSETS)} LiDAR sweeps per frame")
print(f"   • FOV vertical expandido")

start_time = time.time()

# Generate the dataset
for scenario_idx, scenario in enumerate(scenarios):
    print(f"\nScenario {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()
    print(f"   Ego: {-ego_height} m altitude")

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        if cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(
                cfg["x"], cfg["y"], cfg["z"], 5,
                vehicle_name=cfg["vehicle"]
            ).join()
            print(f"   {cfg['vehicle']}: {cfg['desc']}")

    time.sleep(2)

    # Capture at each angle
    for yaw in yaw_angles:
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # Capture the RGB image
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # LIDAR CAPTURE WITH THE WIDENED FOV
        print(f"Capturing the widened LiDAR sweep (frame {total_frames})...")
        points = capture_lidar_with_extended_fov(client, "Ego")

        if len(points) > 3 and img_bgr is not None:
            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            # Keep the forward points
            front_mask = points[:, 0] > 0.5
            front_points = points[front_mask]

            frame_det_high = 0
            frame_det_low = 0

            if len(front_points) > 0:
                X = front_points[:, 0]
                Y = front_points[:, 1]
                Z = front_points[:, 2]

                # Vertical-range statistics
                z_min, z_max = Z.min(), Z.max()
                z_range = z_max - z_min

                # Detecta anomalias
                z_mean = Z.mean()
                z_std = Z.std()
                anomalies_mask = np.abs(Z - z_mean) > 2

                # Base pitch correction
                camera_pitch = -15
                pitch_rad = np.radians(camera_pitch)

                # Apply the rotation
                X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
                Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

                # Projeta
                u = (fx * Y / X_rot + cx).astype(int)
                v = (fy * Z_rot / X_rot + cy).astype(int)

                # Keep the valid entries
                valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

                u_valid = u[valid]
                v_valid = v[valid]
                x_valid = X[valid]
                anomalies_valid = anomalies_mask[valid]

                # Draw the points
                horizon_line = h // 2

                for i in range(len(u_valid)):
                    if anomalies_valid[i]:
                        # Drone detectado!
                        if v_valid[i] < horizon_line:
                            # above the horizon
                            color = (255, 0, 255)  # Magenta
                            frame_det_high += 1
                            cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (255, 100, 255), 2)
                        else:
                            # below the horizon
                            color = (0, 255, 255)  # Amarelo
                            frame_det_low += 1
                            cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, (100, 255, 255), 2)

                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 5, color, -1)
                    else:
                        # Ordinary points
                        depth = min(1.0, x_valid[i] / 50)
                        color = (0, int(255*(1-depth)), 0)
                        cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, color, -1)

                # Reference horizon line
                cv2.line(fusion, (0, horizon_line), (w, horizon_line), (0, 255, 255), 1)

                # Information drawn on the frame
                cv2.putText(fusion, f"{scenario['name']} | Yaw: {yaw}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(fusion, f"High: {frame_det_high} | Low: {frame_det_low}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(fusion, f"Z range: {z_range:.1f}m", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(fusion, "Extended FOV", (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Save the files
                cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
                cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

                # Comparison
                comparison = np.hstack([img_bgr, fusion])
                cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

                # Metadata
                metadata = {
                    "frame_id": total_frames,
                    "scenario": scenario['name'],
                    "yaw": yaw,
                    "ego_height": -ego_height,
                    "total_points": len(points),
                    "front_points": len(front_points),
                    "z_range": float(z_range),
                    "z_min": float(z_min),
                    "z_max": float(z_max),
                    "detections_high": frame_det_high,
                    "detections_low": frame_det_low,
                    "lidar_scans": len(LIDAR_PITCH_OFFSETS),
                    "extended_fov": True
                }

                with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)

                detections_high += frame_det_high
                detections_low += frame_det_low
                total_frames += 1

                print(f"      Frame {total_frames}: High={frame_det_high}, Low={frame_det_low}, Z_range={z_range:.1f}m")

# Land everything
print("\nLanding...")
for v in vehicles:
    try:
        client.moveToPositionAsync(0, 0, -3, 3, vehicle_name=v).join()
    except:
        pass

for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

elapsed = time.time() - start_time

print("\n" + "="*70)
print("IMPROVED-LIDAR DATASET COMPLETE")
print("="*70)
print(f"\nSTATISTICS:")
print(f"   - frames generated: {total_frames}")
print(f"   - detections above the horizon: {detections_high}")
print(f"   - detections below the horizon: {detections_low}")
print(f"   - high-region detection rate: {(detections_high/(detections_high+detections_low)*100):.1f}%" if (detections_high+detections_low) > 0 else "")
print(f"   - total time: {elapsed:.1f} s")
print(f"\n MELHORIAS APLICADAS:")
print(f"   - vertical FOV widened with {len(LIDAR_PITCH_OFFSETS)} sweeps")
print(f"   - captures drones above and below the horizon")
print(f"   - automatic pitch compensation")
print(f"\nDataset saved to: {output_dir.absolute()}")