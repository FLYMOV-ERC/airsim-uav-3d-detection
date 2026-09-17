#!/usr/bin/env python3
"""FASE 1 — Grava uma sessão crua do AirSim (sem inferência, captura rápida).

Salva por frame: RGB (jpg) + depth (npy float16) + x_plat (pose global do ego) +
timestamp + GT global dos drones. Depois `process_offline.py` aplica YOLO+frustum+EKF.

Ego é SEGURO no ar e drones voam por teleport (FlightThread).
"""
import sys, time, json, argparse, math
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
import cosysairsim as airsim
from cosysairsim.types import ImageResponse, Pose
from drone_visual import DroneVisualizer, VEHICLE_ORIGINS
from live_test_flying import FlightThread, _yaw_quat
from inference_pipeline import airsim_pose_to_x_plat

IMAGE_W, IMAGE_H = 1280, 720
import threading


def obj_pose(c, name):
    return Pose.from_msgpack(c.client.call('simGetObjectPose', name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="172.19.80.1"); ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", default="session_rec")
    ap.add_argument("--regime", default="mix", choices=["near","mid","far","mix"])
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--hour", type=int, default=12)
    ap.add_argument("--ndrones", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--static_ego", action="store_true", help="ego parado (default: ego em movimento)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True); (out / "depth").mkdir(exist_ok=True)

    c = airsim.MultirotorClient(ip=args.ip, port=args.port); c.confirmConnection()
    vehicles = c.listVehicles(); drones = [v for v in vehicles if v != "Ego"][:args.ndrones]
    print(f"[Rec] drones={drones} regime={args.regime} weather={args.weather} hour={args.hour}")

    # Weather + hora (reusa do gerador)
    try:
        from generate_dataset_urban import apply_weather, apply_time_of_day
        apply_weather(c, args.weather); apply_time_of_day(c, args.hour)
    except Exception as e:
        print(f"weather skip: {e}")

    EGO_Z = -20.0
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, EGO_Z), _yaw_quat(0)), True, "Ego")
    time.sleep(0.5)
    ego_g = np.array([0.0, 0.0, EGO_Z + VEHICLE_ORIGINS['Ego'][2]])

    # Posições ALEATÓRIAS dentro do regime de distância (azimute ±32° p/ ficar no FOV)
    DIST = {"near": (18, 32), "mid": (32, 50), "far": (50, 78), "mix": (18, 78)}[args.regime]
    targets_global = {}
    for d in drones:
        dist = rng.uniform(*DIST)
        az = math.radians(rng.uniform(-32, 32))
        gx = ego_g[0] + dist*math.cos(az)
        gy = ego_g[1] + dist*math.sin(az)
        gz = ego_g[2] + dist*math.tan(math.radians(15)) + rng.uniform(-3, 3)  # ~centralizado p/ câmera -15°
        targets_global[d] = (gx, gy, gz)
    centers = {}
    for d in drones:
        gx, gy, gz = targets_global.get(d, (ego_g[0]+30, ego_g[1], ego_g[2]))
        ox, oy, oz = VEHICLE_ORIGINS.get(d, (0,0,0))
        centers[d] = (gx-ox, gy-oy, gz-oz)
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*centers[d]), _yaw_quat(0)), True, d)
    time.sleep(0.5)

    viz = DroneVisualizer(c, drones, scale=4.0, sync_hz=30); viz.attach(); viz.start_sync(); time.sleep(1)
    stop = threading.Event()
    fly = FlightThread(args.ip, args.port, drones, centers, EGO_Z, stop, radius=4.0, w=0.3,
                       ego_move=(not args.static_ego))
    fly.start()
    print(f"[Rec] gravando {args.duration}s (captura crua, sem inferência)...")

    cap = airsim.MultirotorClient(ip=args.ip, port=args.port); cap.confirmConnection()
    meta = []
    t0 = time.time(); i = 0
    try:
        while time.time() - t0 < args.duration:
            ts = time.time()
            raw = cap.client.call('simGetImages', [
                airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest('front_center', airsim.ImageType.DepthPlanar, True, False),
            ], 'Ego', False)
            if not raw or len(raw) < 2: continue
            sc = ImageResponse.from_msgpack(raw[0])
            rgb = np.frombuffer(sc.image_data_uint8, np.uint8).reshape(IMAGE_H, IMAGE_W, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            dp = ImageResponse.from_msgpack(raw[1])
            depth = np.array(dp.image_data_float, np.float32).reshape(dp.height, dp.width)
            if depth.shape != (IMAGE_H, IMAGE_W): depth = cv2.resize(depth, (IMAGE_W, IMAGE_H))
            # pose global do ego → x_plat (ego parado: velocidade 0)
            ego = obj_pose(cap, 'Ego')
            x_plat = airsim_pose_to_x_plat(ego, (0.0, 0.0, 0.0))
            # GT global dos drones (p/ validação offline)
            gt = {}
            for d in drones:
                try:
                    p = obj_pose(cap, 'VisQuad_'+d).position
                    gt[d] = [float(p.x_val), float(p.y_val), float(p.z_val)]
                except Exception: pass

            cv2.imwrite(str(out/"rgb"/f"{i:05d}.jpg"), bgr)
            np.save(out/"depth"/f"{i:05d}.npy", depth.astype(np.float16))
            meta.append({"frame": i, "ts": ts, "x_plat": x_plat.tolist(), "gt_global": gt})
            i += 1
            if i % 10 == 0:
                print(f"  rec {i} frames ({i/(ts-t0+1e-3):.1f} fps)")
    finally:
        stop.set(); fly.join(timeout=3); viz.stop_sync(); viz.detach()

    fps = i / max(time.time() - t0, 1e-3)
    json.dump({"fps": fps, "n": i, "frames": meta}, open(out/"meta.json", "w"))
    print(f"[Rec] {i} frames @ {fps:.1f} fps salvos em {out}/  (use process_offline.py)")


if __name__ == "__main__":
    main()
