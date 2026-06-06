#!/usr/bin/env python3
"""
Completa dataset até 300 frames
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("🚀 COMPLETANDO DATASET ATÉ 300 FRAMES")

# Check existing
out = Path("dataset_300")
existing = len(list((out/"rgb").glob("*.png")))
print(f"Frames existentes: {existing}")

if existing >= 300:
    print("✅ Dataset já completo!")
    exit()

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
vehicles = client.listVehicles()

# Prepara
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        if existing == 0:  # Só decola se começando do zero
            client.takeoffAsync(vehicle_name=v)
    except:
        pass

if existing == 0:
    time.sleep(5)

# Params
fx, fy = 914, 914
cx, cy = 640, 360
pitch = np.radians(-15)

frame_id = existing
target = 300
frames_per_config = 10

print(f"Gerando frames {frame_id} a {target-1}\n")

while frame_id < target:
    # Posição aleatória para variedade
    height = np.random.randint(-30, -10)
    client.moveToPositionAsync(0, 0, height, 5, "Ego").join()

    # Posiciona outros drones aleatoriamente
    for i, v in enumerate(vehicles):
        if v != "Ego":
            x = np.random.randint(5, 25)
            y = np.random.randint(-10, 10)
            z = height + np.random.randint(-5, 5)
            try:
                client.moveToPositionAsync(x, y, z, 5, v).join()
            except:
                pass

    time.sleep(2)

    # Captura múltiplos frames nesta configuração
    for _ in range(min(frames_per_config, target - frame_id)):
        # Ângulo aleatório
        yaw = np.random.randint(-30, 31)
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

                # Anomalias ANTES de filtrar
                z_m = Z.mean()
                z_s = Z.std()
                anom = np.abs(Z - z_m) > 2*z_s

                # Projeta com correção pitch
                Xr = X*np.cos(pitch) - Z*np.sin(pitch)
                Zr = X*np.sin(pitch) + Z*np.cos(pitch)

                u = (fx*Y/Xr + cx).astype(int)
                v = (fy*Zr/Xr + cy).astype(int)

                ok = (u>=0) & (u<1280) & (v>=0) & (v<720)

                # Mantém info de anomalia após filtrar
                u_ok = u[ok]
                v_ok = v[ok]
                anom_ok = anom[ok]
                x_ok = X[ok]

                # Desenha
                det = 0
                for i in range(len(u_ok)):
                    if anom_ok[i]:
                        cv2.circle(fus, (u_ok[i], v_ok[i]), 4, (255,0,255), -1)
                        cv2.circle(fus, (u_ok[i], v_ok[i]), 7, (255,255,0), 2)
                        det += 1
                    else:
                        d = min(1.0, x_ok[i]/50)
                        cv2.circle(fus, (u_ok[i], v_ok[i]), 1, (0,int(255*(1-d)),0), -1)

                cv2.putText(fus, f"Frame {frame_id} | Det: {det}", (10,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # Salva
            cv2.imwrite(str(out/"rgb"/f"{frame_id:04d}.png"), img)
            cv2.imwrite(str(out/"fusion"/f"{frame_id:04d}.png"), fus)

            frame_id += 1
            if frame_id % 10 == 0:
                print(f"✅ {frame_id}/{target}")

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

total = len(list((out/"rgb").glob("*.png")))
print(f"\n🎉 DATASET COMPLETO! {total} frames em {out.absolute()}")
print(f"   • RGB: {out}/rgb/")
print(f"   • Fusão: {out}/fusion/")