"""ARCHIVED. The documented Phase-1 English collector.

Uses simPause plus the Segmentation and LiDAR channels. It has NO VisQuad visual
mesh, so it did not produce the dataset reported in Chapter 7, but it is the
cleanest pre-VisQuad ancestor of the canonical collector. Documented in
docs/environments.md.
"""

# collect_dataset.py
# Collect RGB + segmentation + LiDAR, with the ego and Intruder1 moving safely (no collision).
# Usage (live mode):
#   python collect_dataset.py --realtime --frames 1200 --cams front_center back_center bottom_center
# Usage (deterministic step mode):
#   python collect_dataset.py --frames 1200 --dt 0.05 --keep_moving --cams front_center back_center bottom_center

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

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

RUN_MOTION = True  # run flag for the motion threads

def ego_forward_loop(client, ego_name, speed=6.0, target_z=-10.0):
    """Keep the ego advancing along its current yaw at constant altitude."""
    while RUN_MOTION:
        try:
            st = client.getMultirotorState(vehicle_name=ego_name)
            yaw = yaw_from_quat(st.kinematics_estimated.orientation)
            vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)

            # Controle suave de altitude
            current_z = st.kinematics_estimated.position.z_val
            vz = (target_z - current_z) * 0.2
            vz = max(-1.0, min(1.0, vz))

            # Move with a long duration, to avoid dropping into hover mode
            client.moveByVelocityAsync(
                vx, vy, vz, duration=1.0,  # Duration longo
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=ego_name
            )
        except Exception as e:
            pass
        time.sleep(0.2)  # shorter sleep, for more frequent commands

def intruder_front_safe_loop(client, ego_name, intr_name,
                             period=0.1, ahead=25.0, lateral=0.0, z_offset=0.0,
                             speed=6.0, min_sep=15.0):
    """
    Continuous, dynamic motion of the intruder.
    Keeps the motion relative to the ego, in a sinusoidal / figure-of-eight pattern.
    """
    motion_time = 0  # counter driving the motion pattern
    frame_count = 0
    target_altitude = -10.0  # Altitude alvo constante

    # REMOVED enableApiControl and armDisarm from here -- now done before the thread starts

    while RUN_MOTION:
        try:
            frame_count += 1
            motion_time += period

            # Current positions
            ego_p = client.simGetVehiclePose(vehicle_name=ego_name).position
            intr_p = client.simGetVehiclePose(vehicle_name=intr_name).position

            ego_st = client.getMultirotorState(vehicle_name=ego_name)

            ex, ey, ez = ego_p.x_val, ego_p.y_val, ego_p.z_val
            ix, iy, iz = intr_p.x_val, intr_p.y_val, intr_p.z_val

            # Ego speed and heading
            ego_vx = ego_st.kinematics_estimated.linear_velocity.x_val
            ego_vy = ego_st.kinematics_estimated.linear_velocity.y_val
            ego_yaw = yaw_from_quat(ego_st.kinematics_estimated.orientation)

            # Range to the ego
            dx = ix - ex
            dy = iy - ey
            dist_horizontal = math.sqrt(dx*dx + dy*dy)

            # DYNAMIC MOTION RELATIVE TO THE EGO
            # Sinusoidal / figure-of-eight pattern around the ego
            t = motion_time * 2.0  # angular rate raised (it was 0.5)

            # Motion-pattern parameters
            front_distance = 20.0  # desired forward range
            lateral_amplitude = 15.0  # Amplitude lateral do movimento sinusoidal
            vertical_amplitude = 3.0  # small altitude variation

            # Build a figure-of-eight or sinusoidal pattern relative to the ego heading
            # Forward offset along the ego's direction of travel
            forward_offset_x = front_distance * math.cos(ego_yaw)
            forward_offset_y = front_distance * math.sin(ego_yaw)

            # Lateral offset perpendicular to the ego's heading
            lateral_offset = lateral_amplitude * math.sin(t)
            lateral_offset_x = lateral_offset * math.cos(ego_yaw + math.pi/2)
            lateral_offset_y = lateral_offset * math.sin(ego_yaw + math.pi/2)

            # Gentle vertical variation
            vertical_offset = vertical_amplitude * math.sin(t * 0.7)

            # Target position relative to the ego
            target_x = ex + forward_offset_x + lateral_offset_x
            target_y = ey + forward_offset_y + lateral_offset_y
            target_z = target_altitude + vertical_offset

            # Vector towards the target
            dx_to_target = target_x - ix
            dy_to_target = target_y - iy
            dz_to_target = target_z - iz

            dist_to_target = math.sqrt(dx_to_target*dx_to_target + dy_to_target*dy_to_target)

            # BASE VELOCITY TOWARDS THE TARGET
            if dist_to_target > 0.1:
                vx = (dx_to_target / dist_to_target) * speed
                vy = (dy_to_target / dist_to_target) * speed
            else:
                # Tangential motion when close to the target
                vx = -lateral_amplitude * math.cos(t) * 2.0 * math.cos(ego_yaw + math.pi/2)
                vy = -lateral_amplitude * math.cos(t) * 2.0 * math.sin(ego_yaw + math.pi/2)

            vz = dz_to_target * 0.5  # Controle suave de altitude

            # FULL compensation of the ego velocity (raised from 0.3 to 1.0)
            vx += ego_vx * 1.0
            vy += ego_vy * 1.0

            # Safety: if too close to the ego, move away
            if dist_horizontal < min_sep:
                # Repulsion vector away from the ego
                if dist_horizontal > 0.1:
                    repel_x = (ix - ex) / dist_horizontal * 10.0  # Aumentado de 8.0
                    repel_y = (iy - ey) / dist_horizontal * 10.0
                    vx = repel_x + ego_vx
                    vy = repel_y + ego_vy
                    vz = 1.0  # climb faster
                    print(f"[SAFETY] too close: {dist_horizontal:.1f} m")

            # Limita velocidades verticais
            vz = max(-2.0, min(2.0, vz))  # Limites aumentados

            # GUARANTEE CONTINUOUS MOTION - always keep a minimum speed
            min_horizontal_speed = 4.0  # Aumentado de 3.0
            current_speed = math.sqrt(vx*vx + vy*vy)

            # ALWAYS add a motion component, even when there is already some velocity
            if current_speed < min_horizontal_speed * 1.5:  # Margem maior
                # Add a continuous sinusoidal lateral motion
                boost_factor = max(1.0, (min_horizontal_speed - current_speed) / min_horizontal_speed)
                vx += math.cos(t * 3.0) * boost_factor * 3.0
                vy += math.sin(t * 3.0) * boost_factor * 3.0

            # Simple debug output every 30 frames
            if frame_count % 30 == 0:
                print(f"[Intruder-{frame_count:04d}] Pos: ({ix:.1f},{iy:.1f},{iz:.1f}) | Vel CMD: ({vx:.1f},{vy:.1f},{vz:.1f}) | Dist: {dist_horizontal:.1f}m")

            # CRITICAL: use a duration SHORTER than the period, to allow fast updates
            # the duration must be slightly longer than the period, to avoid gaps
            duration_cmd = period * 1.5  # it used to be a fixed 1.0 s

            # Velocity command with a tuned duration
            client.moveByVelocityAsync(
                vx, vy, vz,
                duration=duration_cmd,  # now ~0.075 s instead of 1.0 s
                drivetrain=airsim.DrivetrainType.ForwardOnly,  # Mudado de MaxDegreeOfFreedom
                yaw_mode=airsim.YawMode(False, 0),
                vehicle_name=intr_name
            )

        except Exception as e:
            if frame_count % 20 == 0:  # report errors more frequently
                print(f"[ERROR] intruder, frame {frame_count}: {e}")

        time.sleep(period)  # short sleep, for frequent commands

# ===================== Main ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=airsim_host())
    ap.add_argument("--port", type=int, default=airsim_port())
    ap.add_argument("--vehicle", default="Ego")
    ap.add_argument("--intruder", default="Intruder1")
    ap.add_argument("--lidar", default="LidarFront")
    ap.add_argument("--cams", nargs="+", default=["front_center", "back_center", "bottom_center"])
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--realtime", action="store_true", help="capture live, without simPause")
    ap.add_argument("--dt", type=float, default=0.05, help="time step in step mode")
    ap.add_argument("--keep_moving", action="store_true", help="in step mode, keep advancing the simulation")
    ap.add_argument("--log_dist_every", type=int, default=30, help="print the ego-intruder range every N frames")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ["images", "seg", "lidar", "meta"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
        if sub in ["images", "seg"]:
            for cam in args.cams:
                (out / sub / cam).mkdir(parents=True, exist_ok=True)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    print("Connected")
    try:
        print("Vehicles available:", client.listVehicles())
    except Exception:
        pass

    # habilita/arma/decola
    for v in [args.vehicle, args.intruder]:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception as e:
            print(f"Error enabling {v}: {e}")

    print("Taking off the drones...")
    client.takeoffAsync(vehicle_name=args.vehicle).join()
    try:
        client.takeoffAsync(vehicle_name=args.intruder).join()
    except Exception as e:
        print(f"The intruder did not take off: {e}")

    # posicionamento inicial
    initial_altitude = -10.0
    print(f"Moving the ego to (0, 0, {initial_altitude})")
    client.moveToPositionAsync(0, 0, initial_altitude, 5, vehicle_name=args.vehicle).join()

    # DO NOT MOVE THE INTRUDER - settings.json already puts it at x=-20
    # Intruder at x=-20 (BEHIND the ego) with yaw=180 (facing the ego)
    # Only its altitude is adjusted
    try:
        print(f"Setting the intruder altitude to {initial_altitude}")
        intr_pose = client.simGetVehiclePose(vehicle_name=args.intruder)
        # Keep x and y, adjust only z
        client.moveToPositionAsync(
            intr_pose.position.x_val,
            intr_pose.position.y_val,
            initial_altitude,
            5,
            vehicle_name=args.intruder
        ).join()
    except Exception as e:
        print(f"Error adjusting the intruder: {e}")

    # Espera estabilizar
    time.sleep(2)
    print("Starting the drone motion...")
    print(f"NOTE: the intruder is at x=-20 (behind) and should stay away from the ego")

    # start the motion threads with the tuned parameters
    ego_thr = threading.Thread(
        target=ego_forward_loop,
        args=(client, args.vehicle, 6.0, -10.0),  # speed=6.0, target_z=-10.0
        daemon=True
    )
    ego_thr.start()

    intr_thr = threading.Thread(
        target=intruder_front_safe_loop,
        # period=0.05 (faster), ahead=25.0, lateral=0.0, z_offset=0.0, speed=8.0 (faster), min_sep=12.0
        args=(client, args.vehicle, args.intruder, 0.05, 25.0, 0.0, 0.0, 8.0, 12.0),
        daemon=True
    )
    intr_thr.start()

    # ------- calibration -------
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

    # ------- time mode -------
    if not args.realtime:
        client.simPause(True)

    def make_requests():
        reqs = []
        for cam in args.cams:
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False))
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Segmentation, False, False))
        return reqs

    last_w, last_h = args.width, args.height

    for i in tqdm(range(1, args.frames + 1), desc="Capturing"):
        # step mode
        if not args.realtime:
            if args.keep_moving:
                client.simContinueForTime(args.dt)
            else:
                client.simPause(False)
                client.simContinueForTime(args.dt)
                client.simPause(True)

        # images
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

        # metadata + range logs
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
    time.sleep(0.2)  # give the threads time to exit

    if not args.realtime:
        client.simPause(False)

    print(f"Done -- dataset written to: {out.resolve()}")

if __name__ == "__main__":
    main()
