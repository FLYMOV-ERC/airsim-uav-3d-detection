#!/usr/bin/env python3
"""
Gerador Rápido - 300 frames
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("🚀 GERANDO 300 FRAMES COM FUSÃO LIDAR")

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
vehicles = client.listVehicles()

# Prepara e decola
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        client.takeoffAsync(vehicle_name=v)
    except:
        pass
time.sleep(5)

# Diretórios
out = Path("dataset_300")
out.mkdir(exist_ok=True)
(out/"rgb").mkdir(exist_ok=True)
(out/"fusion").mkdir(exist_ok=True)

# Params
fx, fy = 914, 914
cx, cy = 640, 360
pitch = np.radians(-15)

# Posições variadas para dataset
heights = [-10, -15, -20, -25, -30]
yaws = [-30, -20, -10, 0, 10, 20, 30]

frame_id = 0
target = 300

print(f"Meta: {target} frames\n")

# Loop principal
for h_idx, height in enumerate(heights):
    if frame_id >= target:
        break

    # Move Ego
    client.moveToPositionAsync(0, 0, height, 5, "Ego").join()

    # Move outros drones
    for i, v in enumerate(vehicles):
        if v != "Ego":
            x = 10 + i*5
            y = (i-1)*5
            z = height + (i-1)*3
            try:
                client.moveToPositionAsync(x, y, z, 5, v).join()
            except:
                pass

    time.sleep(2)

    for yaw in yaws:
        if frame_id >= target:
            break

        # Rotaciona
        client.rotateToYawAsync(yaw, "Ego").join()
        time.sleep(0.2)

        # RGB
        resp = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], "Ego")

        img = None
        if resp[0].image_data_uint8:
            arr = np.frombuffer(resp[0].image_data_uint8, np.uint8)
            img = cv2.cvtColor(arr.reshape(720, 1280, 3), cv2.COLOR_RGB2BGR)

        # LiDAR
        lidar = client.getLidarData("LidarFront", "Ego")

        if lidar and img is not None:
            pts = np.array(lidar.point_cloud, np.float32).reshape(-1, 3)

            # Fusão
            fus = img.copy()
            front = pts[pts[:,0] > 0.5]

            if len(front) > 0:
                X, Y, Z = front[:,0], front[:,1], front[:,2]

                # Anomalias
                z_m = Z.mean()
                anom = np.abs(Z - z_m) > 2*Z.std()

                # Projeta
                Xr = X*np.cos(pitch) - Z*np.sin(pitch)
                Zr = X*np.sin(pitch) + Z*np.cos(pitch)

                u = (fx*Y/Xr + cx).astype(int)
                v = (fy*Zr/Xr + cy).astype(int)

                ok = (u>=0) & (u<1280) & (v>=0) & (v<720)

                # Desenha
                det = 0
                for i in np.where(ok)[0]:
                    if anom[i]:
                        cv2.circle(fus, (u[i], v[i]), 4, (255,0,255), -1)
                        det += 1
                    else:
                        cv2.circle(fus, (u[i], v[i]), 1, (0,200,0), -1)

                cv2.putText(fus, f"F{frame_id} D{det}", (10,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # Salva
            cv2.imwrite(str(out/"rgb"/f"{frame_id:04d}.png"), img)
            cv2.imwrite(str(out/"fusion"/f"{frame_id:04d}.png"), fus)

            frame_id += 1
            if frame_id % 10 == 0:
                print(f"✅ {frame_id}/{target}")

# Pousa
for v in vehicles:
    try:
        client.landAsync(v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print(f"\n🎉 COMPLETO! {frame_id} frames em {out.absolute()}")