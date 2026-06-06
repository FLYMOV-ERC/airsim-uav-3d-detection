# collect_dataset.py
# Coleta RGB + Segmentation + LiDAR, com Ego e Intruder1 em movimento seguro (sem colisão).
# Uso (tempo real):
#   python collect_dataset.py --realtime --frames 1200 --cams front_center back_center bottom_center
# Uso (modo step determinístico):
#   python collect_dataset.py --frames 1200 --dt 0.05 --keep_moving --cams front_center back_center bottom_center

import os, json, time, math, argparse, threading
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

# =================== Movimento =====================

RUN_MOTION = True  # flag de execução das threads

def ego_forward_loop(client, ego_name, speed=6.0, target_z=-10.0):
    """Mantém o Ego sempre avançando na direção do yaw atual com altitude constante."""
    while RUN_MOTION:
        try:
            st = client.getMultirotorState(vehicle_name=ego_name)
            yaw = yaw_from_quat(st.kinematics_estimated.orientation)
            vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)

            # Controle suave de altitude
            current_z = st.kinematics_estimated.position.z_val
            vz = (target_z - current_z) * 0.2
            vz = max(-1.0, min(1.0, vz))

            # Move com duration longo para evitar hover mode
            client.moveByVelocityAsync(
                vx, vy, vz, duration=1.0,  # Duration longo
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=ego_name
            )
        except Exception as e:
            pass
        time.sleep(0.2)  # Sleep menor para comandos mais frequentes

def intruder_front_safe_loop(client, ego_name, intr_name,
                             period=0.1, ahead=25.0, lateral=0.0, z_offset=0.0,
                             speed=6.0, min_sep=15.0):
    """
    Movimento dinâmico e contínuo do Intruder.
    Mantém movimento relativo ao Ego com padrão sinusoidal/figura-8.
    """
    motion_time = 0  # Contador para padrão de movimento
    frame_count = 0
    target_altitude = -10.0  # Altitude alvo constante

    # REMOVIDO enableApiControl e armDisarm daqui - agora feito antes da thread

    while RUN_MOTION:
        try:
            frame_count += 1
            motion_time += period

            # Posições atuais
            ego_p = client.simGetVehiclePose(vehicle_name=ego_name).position
            intr_p = client.simGetVehiclePose(vehicle_name=intr_name).position

            ego_st = client.getMultirotorState(vehicle_name=ego_name)

            ex, ey, ez = ego_p.x_val, ego_p.y_val, ego_p.z_val
            ix, iy, iz = intr_p.x_val, intr_p.y_val, intr_p.z_val

            # Velocidade e direção do Ego
            ego_vx = ego_st.kinematics_estimated.linear_velocity.x_val
            ego_vy = ego_st.kinematics_estimated.linear_velocity.y_val
            ego_yaw = yaw_from_quat(ego_st.kinematics_estimated.orientation)

            # Distância ao Ego
            dx = ix - ex
            dy = iy - ey
            dist_horizontal = math.sqrt(dx*dx + dy*dy)

            # MOVIMENTO DINÂMICO RELATIVO AO EGO
            # Padrão sinusoidal/figura-8 ao redor do Ego
            t = motion_time * 2.0  # Velocidade angular aumentada (era 0.5)

            # Parâmetros do padrão de movimento
            front_distance = 20.0  # Distância frontal desejada
            lateral_amplitude = 15.0  # Amplitude lateral do movimento sinusoidal
            vertical_amplitude = 3.0  # Pequena variação de altura

            # Cria padrão figura-8 ou sinusoidal relativo à direção do Ego
            # Offset frontal na direção do movimento do Ego
            forward_offset_x = front_distance * math.cos(ego_yaw)
            forward_offset_y = front_distance * math.sin(ego_yaw)

            # Offset lateral perpendicular à direção do Ego
            lateral_offset = lateral_amplitude * math.sin(t)
            lateral_offset_x = lateral_offset * math.cos(ego_yaw + math.pi/2)
            lateral_offset_y = lateral_offset * math.sin(ego_yaw + math.pi/2)

            # Variação vertical suave
            vertical_offset = vertical_amplitude * math.sin(t * 0.7)

            # Posição alvo relativa ao Ego
            target_x = ex + forward_offset_x + lateral_offset_x
            target_y = ey + forward_offset_y + lateral_offset_y
            target_z = target_altitude + vertical_offset

            # Vetor para o alvo
            dx_to_target = target_x - ix
            dy_to_target = target_y - iy
            dz_to_target = target_z - iz

            dist_to_target = math.sqrt(dx_to_target*dx_to_target + dy_to_target*dy_to_target)

            # VELOCIDADE BASE EM DIREÇÃO AO ALVO
            if dist_to_target > 0.1:
                vx = (dx_to_target / dist_to_target) * speed
                vy = (dy_to_target / dist_to_target) * speed
            else:
                # Movimento tangencial quando próximo ao alvo
                vx = -lateral_amplitude * math.cos(t) * 2.0 * math.cos(ego_yaw + math.pi/2)
                vy = -lateral_amplitude * math.cos(t) * 2.0 * math.sin(ego_yaw + math.pi/2)

            vz = dz_to_target * 0.5  # Controle suave de altitude

            # COMPENSAÇÃO TOTAL da velocidade do Ego (aumentado de 0.3 para 1.0)
            vx += ego_vx * 1.0
            vy += ego_vy * 1.0

            # Segurança: se muito perto do Ego, afasta
            if dist_horizontal < min_sep:
                # Vetor de repulsão do Ego
                if dist_horizontal > 0.1:
                    repel_x = (ix - ex) / dist_horizontal * 10.0  # Aumentado de 8.0
                    repel_y = (iy - ey) / dist_horizontal * 10.0
                    vx = repel_x + ego_vx
                    vy = repel_y + ego_vy
                    vz = 1.0  # Sobe mais rápido
                    print(f"[SEGURANÇA] Muito perto! Dist: {dist_horizontal:.1f}m")

            # Limita velocidades verticais
            vz = max(-2.0, min(2.0, vz))  # Limites aumentados

            # GARANTE MOVIMENTO CONTÍNUO - velocidade mínima sempre
            min_horizontal_speed = 4.0  # Aumentado de 3.0
            current_speed = math.sqrt(vx*vx + vy*vy)

            # SEMPRE adiciona componente de movimento, mesmo se já tem velocidade
            if current_speed < min_horizontal_speed * 1.5:  # Margem maior
                # Adiciona movimento lateral sinusoidal contínuo
                boost_factor = max(1.0, (min_horizontal_speed - current_speed) / min_horizontal_speed)
                vx += math.cos(t * 3.0) * boost_factor * 3.0
                vy += math.sin(t * 3.0) * boost_factor * 3.0

            # Debug simples a cada 30 frames
            if frame_count % 30 == 0:
                print(f"[Intruder-{frame_count:04d}] Pos: ({ix:.1f},{iy:.1f},{iz:.1f}) | Vel CMD: ({vx:.1f},{vy:.1f},{vz:.1f}) | Dist: {dist_horizontal:.1f}m")

            # CRÍTICO: Usar duration MENOR que period para permitir atualizações rápidas
            # duration deve ser ligeiramente maior que period para evitar gaps
            duration_cmd = period * 1.5  # era 1.0 segundos fixo!

            # Movimento por velocidade com duration ajustado
            client.moveByVelocityAsync(
                vx, vy, vz,
                duration=duration_cmd,  # Agora ~0.075s ao invés de 1.0s
                drivetrain=airsim.DrivetrainType.ForwardOnly,  # Mudado de MaxDegreeOfFreedom
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=intr_name
            )

        except Exception as e:
            if frame_count % 20 == 0:  # Debug mais frequente de erros
                print(f"[ERRO] Intruder frame {frame_count}: {e}")

        time.sleep(period)  # Sleep curto para comandos frequentes

# ===================== Main ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="172.19.80.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--vehicle", default="Ego")
    ap.add_argument("--intruder", default="Intruder1")
    ap.add_argument("--lidar", default="LidarFront")
    ap.add_argument("--cams", nargs="+", default=["front_center", "back_center", "bottom_center"])
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--realtime", action="store_true", help="captura em tempo real (sem simPause)")
    ap.add_argument("--dt", type=float, default=0.05, help="passo de tempo no modo step")
    ap.add_argument("--keep_moving", action="store_true", help="no modo step, avança a simulação continuamente")
    ap.add_argument("--log_dist_every", type=int, default=30, help="printa a distância Ego↔Intruder a cada N frames")
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
    try:
        print("Vehicles disponíveis:", client.listVehicles())
    except Exception:
        pass

    # habilita/arma/decola
    for v in [args.vehicle, args.intruder]:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception as e:
            print(f"Erro ao habilitar {v}: {e}")

    print("Decolando drones...")
    client.takeoffAsync(vehicle_name=args.vehicle).join()
    try:
        client.takeoffAsync(vehicle_name=args.intruder).join()
    except Exception as e:
        print(f"⚠️ Intruder não decolou: {e}")

    # posicionamento inicial
    initial_altitude = -10.0
    print(f"Movendo Ego para (0, 0, {initial_altitude})")
    client.moveToPositionAsync(0, 0, initial_altitude, 5, vehicle_name=args.vehicle).join()

    # NÃO MOVER O INTRUDER - ele já está em x=-20 pelo settings.json
    # Intruder em x=-20 (ATRÁS do Ego) com Yaw=180 (olhando para o Ego)
    # Vamos apenas ajustar a altura dele
    try:
        print(f"Ajustando altura do Intruder para {initial_altitude}")
        intr_pose = client.simGetVehiclePose(vehicle_name=args.intruder)
        # Mantém x e y, apenas ajusta z
        client.moveToPositionAsync(
            intr_pose.position.x_val,
            intr_pose.position.y_val,
            initial_altitude,
            5,
            vehicle_name=args.intruder
        ).join()
    except Exception as e:
        print(f"Erro ao ajustar Intruder: {e}")

    # Espera estabilizar
    time.sleep(2)
    print("Iniciando movimento dos drones...")
    print(f"NOTA: Intruder está em x=-20 (atrás) e deve se manter distante do Ego")

    # inicia threads de movimento com parâmetros ajustados
    ego_thr = threading.Thread(
        target=ego_forward_loop,
        args=(client, args.vehicle, 6.0, -10.0),  # speed=6.0, target_z=-10.0
        daemon=True
    )
    ego_thr.start()

    intr_thr = threading.Thread(
        target=intruder_front_safe_loop,
        # period=0.05 (mais rápido), ahead=25.0, lateral=0.0, z_offset=0.0, speed=8.0 (mais rápido), min_sep=12.0
        args=(client, args.vehicle, args.intruder, 0.05, 25.0, 0.0, 0.0, 8.0, 12.0),
        daemon=True
    )
    intr_thr.start()

    # ------- calibração -------
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
            "proj_note": "fx,fy derivados de width/height e fov",
        }
    try:
        lidar0 = client.getLidarData(lidar_name=args.lidar, vehicle_name=args.vehicle)
        calib["lidar"] = {"name": args.lidar, "pose_vehicle": pose_to_dict(lidar0.pose) if hasattr(lidar0, "pose") else None}
    except Exception:
        calib["lidar"] = {"name": args.lidar, "pose_vehicle": None}

    with open(out / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    # ------- modo de tempo -------
    if not args.realtime:
        client.simPause(True)

    def make_requests():
        reqs = []
        for cam in args.cams:
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False))
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Segmentation, False, False))
        return reqs

    last_w, last_h = args.width, args.height

    for i in tqdm(range(1, args.frames + 1), desc="Capturando"):
        # step mode
        if not args.realtime:
            if args.keep_moving:
                client.simContinueForTime(args.dt)
            else:
                client.simPause(False)
                client.simContinueForTime(args.dt)
                client.simPause(True)

        # imagens
        responses = client.simGetImages(make_requests(), vehicle_name=args.vehicle)
        it = iter(responses)
        for cam in args.cams:
            scene = next(it)
            seg = next(it)
            img_path = out / "images" / cam / f"{i:06d}.png"
            seg_path = out / "seg" / cam / f"{i:06d}.png"
            save_image(img_path, scene.image_data_uint8, scene.width, scene.height)
            save_image(seg_path, seg.image_data_uint8, seg.width, seg.height)
            last_w, last_h = int(scene.width), int(scene.height)

        # LiDAR
        try:
            lidar = client.getLidarData(lidar_name=args.lidar, vehicle_name=args.vehicle)
            pts = lidar_to_xyz(lidar)
        except Exception:
            pts = np.zeros((0, 3), np.float32)
        np.save(out / "lidar" / f"{i:06d}.npy", pts)

        # metadados + logs de distância
        meta = {
            "frame": i,
            "timestamp_wall": time.time(),
            "image_size": {"w": last_w, "h": last_h},
            "cams": args.cams,
            "lidar_points": int(pts.shape[0]),
            "vehicles": {}
        }
        try:
            v_pose = client.simGetVehiclePose(vehicle_name=args.vehicle)
            v_state = client.getMultirotorState(vehicle_name=args.vehicle)
            meta["vehicles"][args.vehicle] = {
                "pose": pose_to_dict(v_pose),
                "timestamp_sim": getattr(v_state, "timestamp", None),
            }
        except Exception:
            pass

        try:
            i_pose = client.simGetVehiclePose(vehicle_name=args.intruder)
            i_state = client.getMultirotorState(vehicle_name=args.intruder)
            meta["vehicles"][args.intruder] = {
                "pose": pose_to_dict(i_pose),
                "timestamp_sim": getattr(i_state, "timestamp", None),
            }
        except Exception:
            pass

        with open(out / "meta" / f"{i:06d}.json", "w") as f:
            json.dump(meta, f, indent=2)

        if args.log_dist_every and (i % args.log_dist_every == 0):
            try:
                ep = meta["vehicles"][args.vehicle]["pose"]["position"]
                ip = meta["vehicles"][args.intruder]["pose"]["position"]
                dx, dy, dz = ep["x"] - ip["x"], ep["y"] - ip["y"], ep["z"] - ip["z"]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                print(f"[{i:06d}] dist(Ego,Intr) = {dist:6.2f} m")
            except Exception:
                pass

    # ------- encerrar movimento -------
    global RUN_MOTION
    RUN_MOTION = False
    time.sleep(0.2)  # dá tempo das threads saírem

    if not args.realtime:
        client.simPause(False)

    print(f"✅ Fim — dataset salvo em: {out.resolve()}")

if __name__ == "__main__":
    main()
