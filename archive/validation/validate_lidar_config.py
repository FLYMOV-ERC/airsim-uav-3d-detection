#!/usr/bin/env python3
"""ARCHIVED. Validate the widened LiDAR field-of-view configuration.

Evidence for the auxiliary-channel setup the dissertation specifies (Section 7.2.5).
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("LIDAR VALIDATION WITH THE NEW SETTINGS.JSON")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\nVehicles connected: {vehicles}")

# Arm and take everything off
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

print("\nTaking off all the drones...")
takeoff_tasks = []
for v in vehicles:
    task = client.takeoffAsync(vehicle_name=v)
    takeoff_tasks.append(task)

for task in takeoff_tasks:
    task.join()

print("All airborne. Waiting for them to settle...")
time.sleep(3)

# Place the drones at different altitudes, to test the FOV
print("\nPlacing the drones at strategic altitudes:")
print("   Goal: test detection across the full height of the image")

# Ego at mid altitude
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()
print("   Ego: 15 m altitude (reference)")

# Place the drones at different vertical positions
test_positions = [
    ("Drone3", 12, 0, -5, "HIGH: 10 m ABOVE the ego"),
    ("Drone4", 15, -3, -15, "MID: same altitude"),
    ("Intruder1", 18, 3, -25, "LOW: 10 m BELOW")
]

for name, x, y, z, desc in test_positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"   {name}: {desc} (x={x}m, z={z})")

time.sleep(2)

# Output directory
output_dir = Path("validate_lidar")
output_dir.mkdir(exist_ok=True)

# Camera parameters
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360
camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

print("\nCapturing data with the new LiDAR FOV...")

# Capture the image
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")

img_bgr = None
if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Captura LiDAR
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    print(f"\nLIDAR STATISTICS:")
    print(f"   Total points captured: {len(points)}")

    h, w = img_bgr.shape[:2]
    fusion = img_bgr.copy()

    # Keep the forward points
    front_mask = points[:, 0] > 0.5
    front_points = points[front_mask]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        # Vertical range analysis
        z_min = Z.min()
        z_max = Z.max()
        z_range = z_max - z_min
        z_mean = Z.mean()
        z_std = Z.std()

        print(f"   forward points: {len(front_points)}")
        print(f"   minimum Z: {z_min:.2f} m")
        print(f"   maximum Z: {z_max:.2f} m")
        print(f"   total vertical range: {z_range:.2f} m")
        print(f"   mean Z: {z_mean:.2f} m +/- {z_std:.2f}")

        # Detecta anomalias (drones)
        anomalies_mask = np.abs(Z - z_mean) > 2 * z_std

        print(f"\nDETECTIONS:")
        print(f"   anomalies detected (candidate drones): {anomalies_mask.sum()}")

        # Apply the pitch correction and project
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        # Keep the points that are valid in the image
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u_valid = u[valid]
        v_valid = v[valid]
        x_valid = X[valid]
        z_valid = Z[valid]
        anomalies_valid = anomalies_mask[valid]

        # Analysis of the vertical distribution in the image
        if len(v_valid) > 0:
            v_min = v_valid.min()
            v_max = v_valid.max()
            v_coverage = (v_max - v_min) / h * 100

            print(f"\nPROJECTION IN THE IMAGE:")
            print(f"   minimum pixel Y: {v_min} (top=0)")
            print(f"   maximum pixel Y: {v_max} (bottom={h})")
            print(f"   Vertical coverage: {v_coverage:.1f}% of the image")

        # Count the detections per region
        horizon_line = h // 2
        upper_third = h // 3
        lower_third = 2 * h // 3

        detections_top = 0     # upper third
        detections_mid = 0     # middle third
        detections_bot = 0     # lower third

        # Draw the points and count the detections
        for i in range(len(u_valid)):
            if anomalies_valid[i]:
                # Drone detectado!
                if v_valid[i] < upper_third:
                    # Upper third
                    color = (255, 0, 255)  # Magenta
                    detections_top += 1
                    label = "TOP"
                elif v_valid[i] < lower_third:
                    # Middle third
                    color = (255, 255, 0)  # Ciano
                    detections_mid += 1
                    label = "MID"
                else:
                    # Lower third
                    color = (0, 255, 255)  # Amarelo
                    detections_bot += 1
                    label = "BOT"

                # Draw a large circle to highlight it
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 12, color, 2)
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 6, color, -1)
                cv2.putText(fusion, label, (u_valid[i]+15, v_valid[i]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Ordinary points (ground / environment)
                depth = min(1.0, x_valid[i] / 50)
                color = (0, int(255*(1-depth)), 0)
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, color, -1)

        print(f"\nDETECTIONS BY IMAGE REGION:")
        print(f"   UPPER third: {detections_top} drones")
        print(f"   MIDDLE third: {detections_mid} drones")
        print(f"   LOWER third: {detections_bot} drones")
        print(f"   TOTAL: {detections_top + detections_mid + detections_bot} drones")

        # Draw the reference lines
        cv2.line(fusion, (0, upper_third), (w, upper_third), (100, 100, 100), 1)
        cv2.line(fusion, (0, horizon_line), (w, horizon_line), (255, 255, 255), 2)
        cv2.line(fusion, (0, lower_third), (w, lower_third), (100, 100, 100), 1)

        # Draw the information onto the image
        cv2.putText(fusion, "WIDENED FOV VALIDATION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        info_text = f"Detectados: TOP:{detections_top} MID:{detections_mid} BOT:{detections_bot}"
        cv2.putText(fusion, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(fusion, f"Z range: {z_range:.1f}m | Coverage: {v_coverage:.0f}%", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Region labels
        cv2.putText(fusion, "TOP", (w-50, upper_third-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(fusion, "MID", (w-50, horizon_line-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(fusion, "BOT", (w-50, lower_third-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Save the images
        cv2.imwrite(str(output_dir / "fusion_validated.png"), fusion)
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)

        # Side-by-side comparison
        comparison = np.hstack([img_bgr, fusion])
        cv2.imwrite(str(output_dir / "comparison.png"), comparison)

        print(f"\nImages saved to: {output_dir.absolute()}")

        # Success analysis
        print("\n" + "="*50)
        if detections_top > 0:
            print("SUCCESS: the LiDAR is detecting drones at the TOP of the image")
            print("   The widened FOV (70 deg upward) is working")
        else:
            print("No detection in the upper region - check the configuration")

        if v_coverage > 70:
            print("Excellent vertical coverage (>70% of the image)")
        elif v_coverage > 50:
            print("Good vertical coverage (>50% of the image)")
        else:
            print("Limited vertical coverage (<50% of the image)")

# Land all the drones
print("\nLanding all the drones...")
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

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
print("\n RESUMO:")
print("   If drones were detected at the TOP of the image, the widened FOV works")
print("   Coverage above 70% means excellent vertical reach")
print("   Check the images in: validate_lidar/")
print("="*70)