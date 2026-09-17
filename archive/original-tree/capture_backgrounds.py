#!/usr/bin/env python3
"""Captura screenshots SEM drones do ambiente atual do AirSim.

Move drones-alvo pra longe (fora da FOV), captura RGB + depth do Ego em várias
poses (lat/lng/alt), yaws, weathers e horas do dia.

Output: backgrounds/<env>/{rgb,depth,meta}/bg_NNNNN.{jpg,npy,json}

Uso:
    python3 capture_backgrounds.py --env neighborhood --count 200 --output backgrounds
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import cosysairsim as airsim
from cosysairsim.types import ImageResponse
import numpy as np
import cv2


IMAGE_W, IMAGE_H = 1280, 720
CAMERA = "front_center"


# Profile de cada env (reusa do generate_dataset_urban.py)
ENV_PROFILES = {
    "neighborhood": {
        "ego_zs": [-15, -25, -35, -45, -55],
        "x_range": (-50, 50),
        "y_range": (-50, 50),
    },
    "city": {
        "ego_zs": [-70, -85, -100, -120],
        "x_range": (-80, 80),
        "y_range": (-80, 80),
    },
    "coastline": {
        "ego_zs": [-25, -40, -55, -70],
        "x_range": (-100, 100),
        "y_range": (-80, 80),
    },
    "blocks": {
        "ego_zs": [-10, -25, -40],
        "x_range": (-50, 50),
        "y_range": (-50, 50),
    },
}


WEATHER_TYPES = [
    None,                          # clear (default)
    airsim.WeatherParameter.Rain if hasattr(airsim, "WeatherParameter") else None,
    airsim.WeatherParameter.Fog if hasattr(airsim, "WeatherParameter") else None,
    airsim.WeatherParameter.Snow if hasattr(airsim, "WeatherParameter") else None,
]
WEATHER_TYPES = [w for w in WEATHER_TYPES if w is not None]


def apply_weather(client, weather_type, intensity):
    try:
        client.simEnableWeather(True)
        for w in [
            airsim.WeatherParameter.Rain,
            airsim.WeatherParameter.Fog,
            airsim.WeatherParameter.Snow,
            airsim.WeatherParameter.Roadwetness,
        ]:
            client.simSetWeatherParameter(w, 0.0)
        if weather_type is not None:
            client.simSetWeatherParameter(weather_type, intensity)
    except Exception as e:
        print(f"  weather erro: {e}")


def apply_time_of_day(client, hour):
    try:
        client.simSetTimeOfDay(True, f"2024-06-15 {hour:02d}:00:00",
                                is_start_datetime_dst=False,
                                celestial_clock_speed=1.0,
                                update_interval_secs=60,
                                move_sun=True)
    except Exception:
        pass


def yaw_quat(yaw_rad):
    half = yaw_rad * 0.5
    return airsim.Quaternionr(0.0, 0.0, math.sin(half), math.cos(half))


def teleport(client, vehicle_name, x, y, z, yaw_rad=0.0):
    pose = airsim.Pose(airsim.Vector3r(x, y, z), yaw_quat(yaw_rad))
    client.simSetVehiclePose(pose, True, vehicle_name)


def legacy_get_images(client, requests, veh="Ego", external=False):
    raw = client.client.call('simGetImages', requests, veh, external)
    return [ImageResponse.from_msgpack(r) for r in raw]


def capture(client):
    """Retorna (rgb_bgr, depth_array) ou (None, None) em caso de falha."""
    resps = legacy_get_images(client, [
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, True, False),
    ], "Ego")
    if (not resps[0].image_data_uint8) or (not resps[1].image_data_float):
        return None, None
    rgb = np.frombuffer(resps[0].image_data_uint8, dtype=np.uint8).reshape(
        resps[0].height, resps[0].width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = np.array(resps[1].image_data_float, dtype=np.float32).reshape(
        resps[1].height, resps[1].width)
    if depth.shape != (IMAGE_H, IMAGE_W):
        depth = cv2.resize(depth, (IMAGE_W, IMAGE_H))
    return bgr, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENV_PROFILES.keys()))
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--output", default="backgrounds")
    ap.add_argument("--ip", default="172.19.80.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    profile = ENV_PROFILES[args.env]
    out_dir = Path(args.output) / args.env
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    print(f"Connecting AirSim {args.ip}:{args.port}...")
    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"Vehicles: {vehicles}")

    # API + arm
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception:
            pass

    # MOVE drones-alvo pra MUITO MUITO longe — DEEP UNDERGROUND pra gravidade não trazer de volta
    print("Movendo drones pra (-5000, -5000, +2000) — deep underground...")
    for v in vehicles:
        if v == "Ego":
            continue
        try:
            teleport(client, v, -5000.0, -5000.0, 2000.0, 0.0)
        except Exception as e:
            print(f"  {v}: {e}")

    time.sleep(1)

    print(f"\nCapturando {args.count} backgrounds em {args.env}...")
    n_saved = 0
    n_tried = 0
    while n_saved < args.count and n_tried < args.count * 3:
        n_tried += 1
        # Random pose ego
        x = random.uniform(*profile["x_range"])
        y = random.uniform(*profile["y_range"])
        z = random.choice(profile["ego_zs"])
        yaw = random.uniform(-math.pi, math.pi)

        # Re-teleport drones pra underground (gravidade pode trazer de volta)
        if n_saved % 10 == 0:
            for v in vehicles:
                if v == "Ego": continue
                try: teleport(client, v, -5000.0, -5000.0, 2000.0, 0.0)
                except: pass

        # Random weather (a cada 20 capturas)
        if n_saved % 20 == 0:
            if WEATHER_TYPES and random.random() < 0.4:
                wt = random.choice(WEATHER_TYPES)
                intensity = random.uniform(0.2, 0.7)
                apply_weather(client, wt, intensity)
            else:
                apply_weather(client, None, 0.0)
            hour = random.choice([7, 10, 13, 16, 18])
            apply_time_of_day(client, hour)
            time.sleep(0.3)

        # Teleport ego
        try:
            teleport(client, "Ego", x, y, z, yaw)
            time.sleep(0.25)
        except Exception as e:
            print(f"  teleport err: {e}")
            continue

        # Capture (pausa pra evitar race)
        try:
            client.simPause(True)
            try:
                bgr, depth = capture(client)
            finally:
                client.simPause(False)
        except Exception as e:
            print(f"  capture err: {e}")
            continue

        if bgr is None:
            continue

        # Verifica colisão (se ego colidiu, possívelmente está dentro de prédio)
        # Skip silenciosamente
        try:
            col = client.simGetCollisionInfo("Ego")
            if col.has_collided:
                continue
        except Exception:
            pass

        bg_id = f"bg_{n_saved:05d}"
        cv2.imwrite(str(out_dir / "rgb" / f"{bg_id}.jpg"), bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        np.save(str(out_dir / "depth" / f"{bg_id}.npy"), depth)
        meta = {
            "env": args.env,
            "ego_pose": [float(x), float(y), float(z), float(yaw)],
            "frame_size": [IMAGE_W, IMAGE_H],
            "fov_deg": 90,
        }
        with open(out_dir / "meta" / f"{bg_id}.json", "w") as f:
            json.dump(meta, f)

        n_saved += 1
        if n_saved % 25 == 0:
            print(f"  [{n_saved}/{args.count}]  pose=({x:+.1f},{y:+.1f},{z:.0f}) yaw={math.degrees(yaw):.0f}°  tries={n_tried}")

    print(f"\nDone! {n_saved} backgrounds saved to {out_dir.absolute()}")


if __name__ == "__main__":
    main()
