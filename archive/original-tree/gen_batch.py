#!/usr/bin/env python3
"""
Gera dataset em pequenos lotes
"""
import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import sys

# Argumento: número do lote
batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
frames_per_batch = 20
start_frame = batch_num * frames_per_batch

print(f"🔄 Lote {batch_num}: frames {start_frame} a {start_frame+frames_per_batch-1}")

# Diretórios
out = Path("dataset_300")
out.mkdir(exist_ok=True)
(out/"rgb").mkdir(exist_ok=True)
(out/"fusion").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
vehicles = client.listVehicles()

# Prepara apenas Ego (mais rápido)
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")

# Prepara outros se primeiro lote
if batch_num == 0:
    for v in vehicles:
        if v != "Ego":
            try:
                client.enableApiControl(True, v)
                client.armDisarm(True, v)
                client.takeoffAsync(v)
            except:
                pass
    client.takeoffAsync("Ego").join()
else:
    # Assume que já estão no ar
    pass

# Params
fx, fy = 914, 914
cx, cy = 640, 360
pitch = np.radians(-15)

# Gera frames do lote
for i in range(frames_per_batch):
    frame_id = start_frame + i

    # Varia posições
    height = -10 - (frame_id % 5) * 5  # -10, -15, -20, -25, -30
    yaw = (frame_id * 10) % 61 - 30  # -30 a 30

    client.moveToPositionAsync(0, 0, height, 5, "Ego").join()
    client.rotateToYawAsync(yaw, "Ego").join()

    # Move outros drones se existem
    for idx, v in enumerate(vehicles):
        if v != "Ego":
            x = 10 + (frame_id % 3) * 5
            y = (idx - 1) * 5
            z = height + (idx - 1) * 3
            try:
                client.moveToPositionAsync(x, y, z, 5, v)
            except:
                pass

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
            for j in np.where(ok)[0]:
                if anom[j]:
                    cv2.circle(fus, (u[j], v[j]), 4, (255,0,255), -1)
                    det += 1
                else:
                    cv2.circle(fus, (u[j], v[j]), 1, (0,200,0), -1)

            cv2.putText(fus, f"F{frame_id} D{det}", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Salva
        cv2.imwrite(str(out/"rgb"/f"{frame_id:04d}.png"), img)
        cv2.imwrite(str(out/"fusion"/f"{frame_id:04d}.png"), fus)

    if (i+1) % 5 == 0:
        print(f"   {i+1}/{frames_per_batch} frames")

print(f"✅ Lote {batch_num} completo!")

# Pousa apenas no último lote
if batch_num == 14:  # 15 lotes * 20 = 300
    for v in vehicles:
        try:
            client.landAsync(v)
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass
    print("🛬 Todos pousados!")