#!/usr/bin/env python3
"""Capture drone sprites (RGBA PNGs) of the VisQuad mesh over poses and ranges.

Strategy: raise the ego high, spawn the target in front of it against clear sky,
capture the RGB scene and segment the drone from the sky by a brightness cut,

Output: sprites/drone_NNN.png (RGBA) + sprites/meta.json (orientation, scale, range)
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import argparse
import json
import math
import time
from pathlib import Path

import cosysairsim as airsim
from cosysairsim.types import ImageResponse
import numpy as np
import cv2


IMAGE_W, IMAGE_H = 1280, 720
CAMERA = "front_center"


def legacy_get_images(client, requests, veh="Ego", external=False):
    raw = client.client.call('simGetImages', requests, veh, external)
    return [ImageResponse.from_msgpack(r) for r in raw]


def yaw_quat(yaw_rad):
    half = yaw_rad * 0.5
    return airsim.Quaternionr(0.0, 0.0, math.sin(half), math.cos(half))


def euler_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return airsim.Quaternionr(x, y, z, w)


def capture_rgb(client):
    resps = legacy_get_images(client, [
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
    ], "Ego")
    if not resps[0].image_data_uint8:
        return None
    rgb = np.frombuffer(resps[0].image_data_uint8, dtype=np.uint8).reshape(
        resps[0].height, resps[0].width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def capture_rgb_and_seg(client):
    resps = legacy_get_images(client, [
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
    ], "Ego")
    if not resps[0].image_data_uint8 or not resps[1].image_data_uint8:
        return None, None
    rgb = np.frombuffer(resps[0].image_data_uint8, dtype=np.uint8).reshape(
        resps[0].height, resps[0].width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    seg = np.frombuffer(resps[1].image_data_uint8, dtype=np.uint8).reshape(
        resps[1].height, resps[1].width, 3)
    return bgr, seg


def extract_drone_sprite_threshold(scene_with_drone, brightness_threshold=180,
                                     expected_center=None, search_radius=400):
    """The drone is DARK (the VisQuad mesh is near-black) and the sky is BRIGHT.
    The mask is every pixel whose brightness falls below the threshold.
    """
    gray = cv2.cvtColor(scene_with_drone, cv2.COLOR_BGR2GRAY)
    mask = (gray < brightness_threshold).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None, None
    candidates = []
    for k in range(1, n_labels):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        cx, cy = centroids[k]
        if expected_center is not None:
            d = ((cx - expected_center[0])**2 + (cy - expected_center[1])**2)**0.5
            if d > search_radius:
                continue
        candidates.append((k, area))
    if not candidates:
        return None, None
    best_k, _ = max(candidates, key=lambda t: t[1])
    final_mask = (labels == best_k).astype(np.uint8) * 255
    ys, xs = np.where(final_mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
    sprite_rgb = scene_with_drone[y1:y2, x1:x2]
    sprite_alpha = final_mask[y1:y2, x1:x2]
    return np.dstack([sprite_rgb, sprite_alpha]), (x1, y1, x2, y2)


def extract_drone_sprite_seg(scene_with_drone, seg_with_drone, seg_empty,
                              expected_center=None, search_radius=400):
    """Extract a sprite by segmentation diff: the colour that appeared (the spawned object)."""
    # Unique colours in each image
    seg_e_flat = seg_empty.reshape(-1, 3)
    seg_w_flat = seg_with_drone.reshape(-1, 3)
    empty_colors = set(map(tuple, seg_e_flat))
    # Pixels carrying a NEW colour (absent from the empty capture)
    h, w = seg_with_drone.shape[:2]
    new_color_mask = np.zeros((h, w), dtype=np.uint8)
    # Vectorised: for each unique colour in seg_with_drone, test whether it is in empty
    unique_w_colors, inverse = np.unique(seg_w_flat, axis=0, return_inverse=True)
    is_new = np.array([tuple(c) not in empty_colors for c in unique_w_colors])
    mask_flat = is_new[inverse]
    new_color_mask = (mask_flat.reshape(h, w) * 255).astype(np.uint8)
    # Limpa noise
    kernel = np.ones((3,3), np.uint8)
    new_color_mask = cv2.morphologyEx(new_color_mask, cv2.MORPH_OPEN, kernel)
    # Connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_color_mask, connectivity=8)
    if n_labels <= 1:
        return None, None
    candidates = []
    for k in range(1, n_labels):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        cx, cy = centroids[k]
        if expected_center is not None:
            d = ((cx - expected_center[0])**2 + (cy - expected_center[1])**2)**0.5
            if d > search_radius:
                continue
        candidates.append((k, area))
    if not candidates:
        return None, None
    best_k, _ = max(candidates, key=lambda t: t[1])
    final_mask = (labels == best_k).astype(np.uint8) * 255
    ys, xs = np.where(final_mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
    sprite_rgb = scene_with_drone[y1:y2, x1:x2]
    sprite_alpha = final_mask[y1:y2, x1:x2]
    return np.dstack([sprite_rgb, sprite_alpha]), (x1, y1, x2, y2)


def extract_drone_sprite_diff(scene_with_drone, scene_empty, threshold=25,
                                expected_center=None, search_radius=300):
    """Extract an RGBA sprite by differencing: take the LARGEST connected component near the expected centre.

    expected_center: (cx, cy) where the drone is expected. Components far from it are rejected.
    """
    diff = np.abs(scene_with_drone.astype(np.int16) -
                  scene_empty.astype(np.int16)).sum(axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Connected components -- take the LARGEST one near the expected centre
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None, None
    # Drop the background (label 0) and the small components
    candidates = []
    for k in range(1, n_labels):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        cx, cy = centroids[k]
        if expected_center is not None:
            dx = cx - expected_center[0]; dy = cy - expected_center[1]
            if (dx*dx + dy*dy) ** 0.5 > search_radius:
                continue
        candidates.append((k, area))
    if not candidates:
        return None, None
    # Take the largest
    best_k, _ = max(candidates, key=lambda t: t[1])
    final_mask = (labels == best_k).astype(np.uint8) * 255
    ys, xs = np.where(final_mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
    sprite_rgb = scene_with_drone[y1:y2, x1:x2]
    sprite_alpha = final_mask[y1:y2, x1:x2]
    sprite_rgba = np.dstack([sprite_rgb, sprite_alpha])  # BGRA
    return sprite_rgba, (x1, y1, x2, y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=airsim_host())
    ap.add_argument("--port", type=int, default=airsim_port())
    ap.add_argument("--output", default="sprites")
    ap.add_argument("--ego_alt", type=float, default=-2000.0,
                    help="ego altitude (Z in NED). Very high, so the background is SKY only.")
    ap.add_argument("--threshold", type=int, default=25,
                    help="difference threshold for detecting the drone (0-255). Higher = less noise.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(exist_ok=True)

    print("Connecting AirSim...")
    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()

    # API + arm Ego
    for v in client.listVehicles():
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception:
            pass

    # Move ALL target drones far away (off the map)
    for v in client.listVehicles():
        if v == "Ego":
            continue
        try:
            pose = airsim.Pose(airsim.Vector3r(-5000, -5000, -1000), yaw_quat(0))
            client.simSetVehiclePose(pose, True, v)
        except Exception:
            pass

    # Raise the ego (sky as background)
    print(f"Moving the ego to a high altitude ({args.ego_alt})...")
    pose_ego = airsim.Pose(airsim.Vector3r(0, 0, args.ego_alt), yaw_quat(0))
    client.simSetVehiclePose(pose_ego, True, "Ego")
    time.sleep(1.5)

    # Capture the clean background (reference), including segmentation
    bg_empty, seg_empty = capture_rgb_and_seg(client)
    if bg_empty is None:
        print("Failed to capture a clean background")
        return
    cv2.imwrite(str(out_dir / "_bg_empty.jpg"), bg_empty)
    cv2.imwrite(str(out_dir / "_seg_empty.jpg"), seg_empty)
    print(f"Clean background saved: {out_dir / '_bg_empty.jpg'}")
    print(f"Clean segmentation saved: {out_dir / '_seg_empty.jpg'}")

    # Pose and scale configuration
    # yaws (horizontal rotation): the drone seen from N angles
    yaws_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    pitches_deg = [-15, 0, 15]   # forward/backward tilt
    rolls_deg = [-10, 0, 10]     # tilt lateral
    distances = [10, 15, 25, 40, 60]  # metros forward

    # Cap the total -- sample the combinations
    combos = []
    for d in distances:
        for yaw in yaws_deg:
            for pitch in pitches_deg:
                combos.append((d, yaw, pitch, 0))   # roll=0 mostly
    # Add some with roll
    import random
    random.seed(42)
    for d in distances[1:3]:
        for yaw in [0, 90, 180, 270]:
            for roll in [-10, 10]:
                combos.append((d, yaw, 0, roll))
    print(f"Total combos: {len(combos)}")

    # Expected centre in the image: the drone spawns straight ahead, so v ~ CY
    # but the camera is pitched -15 deg, so a drone at the ego's altitude appears
    # at v ~ centre + pitch offset. A search radius of IMAGE_CENTER +/- 350 px is permissive enough.
    expected_center = (IMAGE_W / 2, IMAGE_H / 2)

    meta = []
    n_saved = 0
    n_skipped = 0
    for i, (dist, yaw_d, pitch_d, roll_d) in enumerate(combos):
    # RESET the ego pose every iteration -- it falls under gravity between captures
        client.simSetVehiclePose(pose_ego, True, "Ego")
        time.sleep(0.05)
    # Spawn the 4x Quadrotor1 in front of the ego, with a UNIQUE name per iteration
        spawn_pos = airsim.Vector3r(0 + dist, 0, args.ego_alt)
        ori = euler_quat(math.radians(roll_d), math.radians(pitch_d), math.radians(yaw_d))
        name = f"SPRITE_Q_{i:04d}"
        try:
            actual_name = client.simSpawnObject(name, "Quadrotor1",
                                    airsim.Pose(spawn_pos, ori),
                                    airsim.Vector3r(4, 4, 4),
                                    physics_enabled=False, is_blueprint=False)
        except Exception as e:
            print(f"  spawn err: {e}")
            n_skipped += 1
            continue
        time.sleep(0.4)

        bgr_with = capture_rgb(client)
        try:
            client.simDestroyObject(actual_name)
        except Exception:
            try: client.simDestroyObject(name)
            except: pass
        if bgr_with is None:
            n_skipped += 1
            continue

        sprite_rgba, bbox = extract_drone_sprite_threshold(bgr_with,
                                                            brightness_threshold=170,
                                                            expected_center=expected_center,
                                                            search_radius=400)
        if sprite_rgba is None:
            n_skipped += 1
            continue
        h, w = sprite_rgba.shape[:2]
        if w < 8 or h < 4:
            n_skipped += 1
            continue
    # Save the RGBA sprite
        sprite_id = f"drone_{n_saved:03d}"
        cv2.imwrite(str(out_dir / f"{sprite_id}.png"), sprite_rgba)
    # Preview with the box drawn
        preview = bgr_with.copy()
        cv2.rectangle(preview, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 1)
        cv2.imwrite(str(out_dir / "preview" / f"{sprite_id}_prev.jpg"), preview)
        meta.append({
            "id": sprite_id,
            "distance_m": dist,
            "yaw_deg": yaw_d,
            "pitch_deg": pitch_d,
            "roll_deg": roll_d,
            "bbox_in_full_image": list(bbox),
            "sprite_size": [w, h],
        })
        n_saved += 1
        if n_saved % 10 == 0:
            print(f"  [{n_saved}/{len(combos)}]  dist={dist} yaw={yaw_d} pitch={pitch_d} sprite={w}x{h}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone! {n_saved} sprites saved ({n_skipped} skipped) → {out_dir.absolute()}")


if __name__ == "__main__":
    main()
