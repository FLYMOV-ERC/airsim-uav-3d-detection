# collect_dataset_simple.py
# Versão simplificada sem threads - controle sequencial para evitar problemas de event loop

import os, json, time, math, argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# -------- AirSim (Cosys) import (fallback) --------
try:
    import airsim
except Exception:
    import cosysairsim as airsim

# ====================== Utils ======================

def fx_fy_from_fov(width, height, fov_deg):
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return f, f

def pose_to_dict(pose):
    return {
        "position": {
            "x": pose.position.x_val,
            "y": pose.position.y_val,
            "z": pose.position.z_val,
        },
        "orientation_xyzw": [
            pose.orientation.x_val,
            pose.orientation.y_val,
            pose.orientation.z_val,
            pose.orientation.w_val,
        ],
    }

def lidar_to_xyz(lidar):
    if not getattr(lidar, "point_cloud", None):
        return np.zeros((0, 3), np.float32)
    return np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)

def save_image(path, img_bytes, width, height):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    if arr.size == width * height * 3:
        img = arr.reshape(height, width, 3)
    elif arr.size == width * height * 4:
        img = arr.reshape(height, width, 4)[:, :, :3]
    else:
        img = arr.reshape(height, width, -1)[:, :, :3]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def yaw_from_quat(q):
    # q: airsim.Quaternionr (w,x,y,z)
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# ===================== Main ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="172.19.80.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--vehicle", default="Ego")
    ap.add_argument("--intruder", default="Intruder1")
    ap.add_argument("--lidar", default="LidarFront")
    ap.add_argument("--cams", nargs="+", default=["front_center"])
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--log_dist_every", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ["images", "seg", "lidar", "meta"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
        if sub in ["images", "seg"]:
            for cam in args.cams:
                (out / sub / cam).mkdir(parents=True, exist_ok=True)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    print("✅ Conectado")

    # Habilita e arma ambos os drones
    for v in [args.vehicle, args.intruder]:
        print(f"Habilitando {v}...")
        client.enableApiControl(True, v)
        client.armDisarm(True, v)

    # Decola
    print("Decolando drones...")
    client.takeoffAsync(vehicle_name=args.vehicle).join()
    client.takeoffAsync(vehicle_name=args.intruder).join()

    # Posicionamento inicial
    initial_altitude = -10.0
    print(f"Movendo Ego para (0, 0, {initial_altitude})")
    client.moveToPositionAsync(0, 0, initial_altitude, 5, vehicle_name=args.vehicle).join()

    # Posiciona Intruder atrás
    print(f"Movendo Intruder para posição inicial")
    client.moveToPositionAsync(-20, 0, initial_altitude, 5, vehicle_name=args.intruder).join()

    time.sleep(2)
    print("Iniciando coleta...")

    # Calibração
    calib = {"vehicle": args.vehicle, "cams": {}, "image_size": [args.width, args.height]}
    for cam in args.cams:
        info = client.simGetCameraInfo(cam, vehicle_name=args.vehicle)
        fx, fy = fx_fy_from_fov(args.width, args.height, info.fov)
        cx, cy = args.width / 2.0, args.height / 2.0
        calib["cams"][cam] = {
            "name": cam,
            "fov_deg": float(info.fov),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "pose_vehicle": pose_to_dict(info.pose),
        }

    with open(out / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    def make_requests():
        reqs = []
        for cam in args.cams:
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False))
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Segmentation, False, False))
        return reqs

    # Variáveis de estado do movimento
    motion_time = 0.0
    ego_speed = 6.0
    intruder_speed = 8.0
    target_altitude = -10.0

    # LOOP PRINCIPAL - SEM THREADS
    for i in tqdm(range(1, args.frames + 1), desc="Capturando"):

        motion_time += 0.05  # incremento de tempo
        t = motion_time * 2.0  # frequência do padrão sinusoidal

        # === MOVIMENTO DO EGO ===
        ego_st = client.getMultirotorState(vehicle_name=args.vehicle)
        ego_yaw = yaw_from_quat(ego_st.kinematics_estimated.orientation)
        ego_z = ego_st.kinematics_estimated.position.z_val

        # Ego sempre avança na direção do yaw
        ego_vx = ego_speed * math.cos(ego_yaw)
        ego_vy = ego_speed * math.sin(ego_yaw)
        ego_vz = (target_altitude - ego_z) * 0.2
        ego_vz = max(-1.0, min(1.0, ego_vz))

        # Comando de movimento do Ego (não bloqueia)
        client.moveByVelocityAsync(
            ego_vx, ego_vy, ego_vz,
            duration=0.5,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
            vehicle_name=args.vehicle
        )

        # === MOVIMENTO DO INTRUDER ===
        ego_p = client.simGetVehiclePose(vehicle_name=args.vehicle).position
        intr_p = client.simGetVehiclePose(vehicle_name=args.intruder).position

        ex, ey = ego_p.x_val, ego_p.y_val
        ix, iy, iz = intr_p.x_val, intr_p.y_val, intr_p.z_val

        # Distância ao Ego
        dx = ix - ex
        dy = iy - ey
        dist = math.sqrt(dx*dx + dy*dy)

        # Movimento sinusoidal ao redor do Ego
        front_distance = 20.0
        lateral_amplitude = 12.0

        # Posição alvo com padrão sinusoidal
        lateral_offset = lateral_amplitude * math.sin(t)
        target_x = ex + front_distance * math.cos(ego_yaw) + lateral_offset * math.cos(ego_yaw + math.pi/2)
        target_y = ey + front_distance * math.sin(ego_yaw) + lateral_offset * math.sin(ego_yaw + math.pi/2)
        target_z = target_altitude + 2.0 * math.sin(t * 0.7)

        # Movimento direto para a posição alvo
        dx_target = target_x - ix
        dy_target = target_y - iy
        dz_target = target_z - iz
        dist_target = math.sqrt(dx_target*dx_target + dy_target*dy_target)

        if dist_target > 0.1:
            intr_vx = (dx_target / dist_target) * intruder_speed + ego_vx
            intr_vy = (dy_target / dist_target) * intruder_speed + ego_vy
        else:
            # Movimento tangencial
            intr_vx = -lateral_amplitude * math.cos(t) * 2.0 + ego_vx
            intr_vy = lateral_amplitude * math.sin(t) * 2.0 + ego_vy

        intr_vz = dz_target * 0.5
        intr_vz = max(-2.0, min(2.0, intr_vz))

        # Segurança - afasta se muito perto
        if dist < 12.0:
            if dist > 0.1:
                repel_x = (ix - ex) / dist * 10.0
                repel_y = (iy - ey) / dist * 10.0
                intr_vx = repel_x + ego_vx
                intr_vy = repel_y + ego_vy
                print(f"[SEGURANÇA] Muito perto! Dist={dist:.1f}m")

        # Comando de movimento do Intruder (não bloqueia)
        client.moveByVelocityAsync(
            intr_vx, intr_vy, intr_vz,
            duration=0.5,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
            vehicle_name=args.intruder
        )

        # === CAPTURA DE DADOS ===
        responses = client.simGetImages(make_requests(), vehicle_name=args.vehicle)
        it = iter(responses)
        for cam in args.cams:
            scene = next(it)
            seg = next(it)
            img_path = out / "images" / cam / f"{i:06d}.png"
            seg_path = out / "seg" / cam / f"{i:06d}.png"
            save_image(img_path, scene.image_data_uint8, scene.width, scene.height)
            save_image(seg_path, seg.image_data_uint8, seg.width, seg.height)

        # LiDAR
        try:
            lidar = client.getLidarData(lidar_name=args.lidar, vehicle_name=args.vehicle)
            pts = lidar_to_xyz(lidar)
        except:
            pts = np.zeros((0, 3), np.float32)
        np.save(out / "lidar" / f"{i:06d}.npy", pts)

        # Metadados
        meta = {
            "frame": i,
            "timestamp_wall": time.time(),
            "image_size": {"w": args.width, "h": args.height},
            "cams": args.cams,
            "lidar_points": int(pts.shape[0]),
            "vehicles": {}
        }

        v_pose = client.simGetVehiclePose(vehicle_name=args.vehicle)
        meta["vehicles"][args.vehicle] = {"pose": pose_to_dict(v_pose)}

        i_pose = client.simGetVehiclePose(vehicle_name=args.intruder)
        meta["vehicles"][args.intruder] = {"pose": pose_to_dict(i_pose)}

        with open(out / "meta" / f"{i:06d}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Log de distância
        if args.log_dist_every and (i % args.log_dist_every == 0):
            ep = meta["vehicles"][args.vehicle]["pose"]["position"]
            ip = meta["vehicles"][args.intruder]["pose"]["position"]
            dx, dy = ep["x"] - ip["x"], ep["y"] - ip["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            print(f"[{i:06d}] dist(Ego,Intr) = {dist:6.2f}m | Intruder: ({ip['x']:.1f}, {ip['y']:.1f})")

        # Pequeno delay para sincronização
        time.sleep(0.01)

    print(f"✅ Fim — dataset salvo em: {out.resolve()}")

if __name__ == "__main__":
    main()