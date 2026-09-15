#!/usr/bin/env python3
"""ARCHIVED. Study of the LiDAR vertical field of view, for capturing higher-flying drones.

The substantive FOV study behind the decision to keep the LiDAR only as an
auxiliary channel (Section 7.2.5).
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
print("TESTE FOV VERTICAL DO LIDAR")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\nVehicles: {vehicles}")

# Arm everything
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Take everything off
print("\nTaking off...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

print("\nPlacing the drones at different altitudes...")

# Ego no meio
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

# Drones at different relative altitudes, to test detection
positions = [
    ("Drone3", 10, 0, -20, "5 m below"),     # below the ego
    ("Drone4", 15, 0, -15, "same altitude"),  # same altitude
    ("Intruder1", 20, 0, -10, "5 m above")    # above the ego
]

for name, x, y, z, desc in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"   {name}: {x} m ahead, {desc} (z={z})")

time.sleep(2)

output_dir = Path("test_lidar_fov")
output_dir.mkdir(exist_ok=True)

# Camera parameters
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360

# Try different pitch configurations
pitch_angles = [-30, -20, -15, -10, 0]

for pitch_test in pitch_angles:
    print(f"\nTrying camera pitch: {pitch_test} deg")

    pitch_rad = np.radians(pitch_test)

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

        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Keep the forward points
        front_mask = points[:, 0] > 0.5
        front_points = points[front_mask]

        if len(front_points) > 0:
            X = front_points[:, 0]
            Y = front_points[:, 1]
            Z = front_points[:, 2]

            # Vertical distribution analysis
            z_min = Z.min()
            z_max = Z.max()
            z_range = z_max - z_min

            print(f"LiDAR vertical range:")
            print(f"      Z min: {z_min:.2f} m | Z max: {z_max:.2f} m | range: {z_range:.2f} m")

            # Detect the anomalies BEFORE projecting
            z_mean = Z.mean()
            z_std = Z.std()
            anomalies_mask = np.abs(Z - z_mean) > 2

            # Apply the pitch correction
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
            z_valid = Z[valid]
            anomalies_valid = anomalies_mask[valid]

            # Projection statistics
            if len(v_valid) > 0:
                v_min = v_valid.min()
                v_max = v_valid.max()
                print(f"   Projection in the image:")
                print(f"      pixel Y: {v_min} to {v_max} (image height: {h})")
                print(f"      coverage: {((v_max-v_min)/h)*100:.1f}% of the image height")

            detections_low = 0    # lower part of the image (y > h/2)
            detections_high = 0   # upper part of the image (y < h/2)

            # Draw the points
            for i in range(len(u_valid)):
                if anomalies_valid[i]:
                    # Drone detectado
                    if v_valid[i] < h/2:
                        # Parte superior
                        color = (255, 0, 255)  # Magenta
                        detections_high += 1
                        label = "HIGH"
                    else:
                        # Parte inferior
                        color = (255, 255, 0)  # Ciano
                        detections_low += 1
                        label = "LOW"

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), 8, color, 2)
                    cv2.putText(fusion, label, (u_valid[i]+10, v_valid[i]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Normal
                    depth = min(1.0, x_valid[i] / 50)
                    color = (0, int(255*(1-depth)), 0)
                    cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, color, -1)

            print(f"   Detections:")
            print(f"      Parte superior: {detections_high}")
            print(f"      Parte inferior: {detections_low}")
            print(f"      Total: {detections_high + detections_low}")

            # Horizon line
            cv2.line(fusion, (0, h//2), (w, h//2), (0, 255, 255), 1)
            cv2.putText(fusion, "HORIZON", (10, h//2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Info
            cv2.putText(fusion, f"Pitch: {pitch_test} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(fusion, f"High: {detections_high} | Low: {detections_low}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, f"Z range: {z_min:.1f} to {z_max:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Save
            cv2.imwrite(str(output_dir / f"fusion_pitch_{pitch_test}.png"), fusion)
            cv2.imwrite(str(output_dir / f"original_pitch_{pitch_test}.png"), img_bgr)

            # Comparison
            comp = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / f"comparison_pitch_{pitch_test}.png"), comp)

print("\n" + "="*50)
print("TESTING ADJUSTMENT OF THE LIDAR VERTICAL ANGLE...")
print("="*50)

# Now adjust the LiDAR sweep angle
print("\nAdjusting the LiDAR parameters for a wider vertical FOV...")

# Rotate the ego to look further up
client.rotateByYawPitchRollAsync(0, -0.3, 0, vehicle_name="Ego").join()  # negative pitch = look up
time.sleep(1)

print("Capturing with the adjusted LiDAR...")

# Capture with the adjustment applied
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")

img_bgr = None
if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    h, w = img_bgr.shape[:2]
    fusion = img_bgr.copy()

    # Use an adjusted pitch, to compensate for the drone's tilt
    adjusted_pitch = -15 - 17  # original pitch + compensation
    pitch_rad = np.radians(adjusted_pitch)

    front_points = points[points[:, 0] > 0.5]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        print(f"\nWith the adjusted LiDAR:")
        print(f"   Z range: {Z.min():.2f} a {Z.max():.2f}")

        # Detecta anomalias
        z_mean = Z.mean()
        anomalies_mask = np.abs(Z - z_mean) > 2

        # Project with the adjusted pitch
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u_valid = u[valid]
        v_valid = v[valid]
        anomalies_valid = anomalies_mask[valid]

        high_detections = 0
        low_detections = 0

        for i in range(len(u_valid)):
            if anomalies_valid[i]:
                if v_valid[i] < h/2:
                    color = (255, 0, 255)
                    high_detections += 1
                else:
                    color = (255, 255, 0)
                    low_detections += 1
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, color, 2)
            else:
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, (0, 200, 0), -1)

        cv2.line(fusion, (0, h//2), (w, h//2), (0, 255, 255), 1)
        cv2.putText(fusion, "LIDAR ADJUSTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(fusion, f"High: {high_detections} | Low: {low_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imwrite(str(output_dir / "fusion_adjusted.png"), fusion)
        cv2.imwrite(str(output_dir / "comparison_adjusted.png"), np.hstack([img_bgr, fusion]))

        print(f"   detections, upper region: {high_detections}")
        print(f"   detections, lower region: {low_detections}")

# Restore the normal orientation
client.rotateToYawAsync(0, vehicle_name="Ego").join()

# Pousa
print("\nLanding...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("TESTE COMPLETO!")
print("="*70)
print(f"\nImages saved to: {output_dir.absolute()}")
print("\nANALYSIS:")
print("   - compare the images at different pitch values")
print("   - check how many drones are detected above and below the horizon")
print("   - the 'fusion_adjusted' image shows the result with the optimised LiDAR")
print("="*70)