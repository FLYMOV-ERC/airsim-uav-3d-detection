#!/usr/bin/env python3
"""Urban dataset generator -- the collector that produced the Chapter 7 dataset.

This is THE generator behind the multimodal collected set of Section 7.2.4
(three environments, 1,000 frames each: RGB + planar depth + LiDAR + segmentation
+ ego pose + geometric 3D boxes).  It is the only variant in the project history
that carries all of: the VisQuad 4x visual mesh, all four sensor channels,
``simPause`` freezing of the scene during acquisition, and the three environments
the chapter uses.

It runs once per environment -- the operator starts the corresponding Unreal
binary on the simulator host and re-runs with a different ``--env``:

    # 1. start CityEnviron, then:
    python dataset_generation/collect_airsim.py --env city \
        --frames 1000 --output dataset_urban_city

    # 2. start AirSimNH, then:
    python dataset_generation/collect_airsim.py --env neighborhood \
        --frames 1000 --output dataset_urban_nh

    # 3. start CityPark, then:
    python dataset_generation/collect_airsim.py --env citypark \
        --frames 1000 --output dataset_urban_park

The three outputs are merged afterwards by concatenating the YOLO label
directories and the point-cloud/annotation directories.

Output layout matches the earlier collectors: YOLO labels (one class, "drone")
plus point clouds with 3D boxes for the PointNet stage.

Techniques used, each validated in an earlier script of the project:

* ``simGetDetections`` + ``simAddDetectionFilterMeshName`` for the mesh-name
  ground truth (see Section 7.2.3 for why this is NOT used for the 2D labels);
* ``simSetVehiclePose`` for fast teleporting -- much faster than
  ``moveToPositionAsync``, and the simulator's hovering controller does not hold
  position to the tolerance the annotation needs;
* ``simGetCollisionInfo`` with a timestamp difference, to detect a NEW collision
  after a teleport rather than a stale one;
* ``simSetWeatherParameter`` + ``simSetTimeOfDay`` for domain randomisation;
* sampling biased toward the ego's horizontal field of view, so the targets
  actually appear in the image.

Host, port and vehicle origins come from configs/default.yaml; the sensor suite
itself is configs/settings.json, which must be installed on the simulator host.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import CONFIG, airsim_host, airsim_port

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cosysairsim as airsim
from cosysairsim.types import DetectionInfo, ImageResponse
import cv2
import numpy as np

from common.drone_visual import DroneVisualizer


# =============================================================================
# COMPAT SHIM for Microsoft AirSim 1.8.x binaries (AirSimNH/CityEnviron).
# Several endpoints gained an `external: bool` argument that cosysairsim 3.3.0
# does not send (or sends as the string `annotation_name`, causing a "bad cast").
# Each method is probed once; what we found:
#   - simGetImages              : needs external=False in slot 3
#   - simClearDetectionMeshNames: external=False no slot 4
#   - simSetDetectionFilterRadius: external=False no slot 5
#   - simAddDetectionFilterMeshName: external=False no slot 5
#   - simGetDetections          : external=False no slot 4
# The remaining methods (weather, pose, collision) work directly through the cosys wrapper.
# =============================================================================
def legacy_clear_detection(client, cam, img_type, veh="", external=False):
    client.client.call('simClearDetectionMeshNames', cam, img_type, veh, external)


def legacy_set_radius(client, cam, img_type, radius_cm, veh="", external=False):
    client.client.call('simSetDetectionFilterRadius', cam, img_type, radius_cm, veh, external)


def legacy_add_mesh(client, cam, img_type, mesh, veh="", external=False):
    client.client.call('simAddDetectionFilterMeshName', cam, img_type, mesh, veh, external)


def legacy_get_detections(client, cam, img_type, veh="", external=False):
    raw = client.client.call('simGetDetections', cam, img_type, veh, external)
    return [DetectionInfo.from_msgpack(r) for r in raw]


def legacy_get_images(client, requests, veh="", external=False):
    raw = client.client.call('simGetImages', requests, veh, external)
    return [ImageResponse.from_msgpack(r) for r in raw]


# =============================================================================
# PER-ENVIRONMENT PROFILES (Urban Air Mobility)
# =============================================================================
# Bounds in AirSim NED coordinates (X forward, Y right, Z DOWN -> -Z is UP).
# Values based on the typical layouts of the three official environments; refine if needed.

ENV_PROFILES = {
    # NOTE: in NED, Z is DOWN (-Z is UP). z_max (the least negative value) sets the
    # drone's MINIMUM ALTITUDE. z_max is kept well above the environment skyline
    # to guarantee sky as background (no occlusion by buildings, houses or trees).
    "city": {
        "name": "CityEnviron",
        "description": "UAM city -- ego 70-100 m AGL (above the low buildings), drones 50-110 m",
        "ego_vantage_points": [
            (0.0,    0.0,   -80.0),
            (-40.0,  30.0,  -90.0),
            (50.0,  -30.0,  -75.0),
            (-30.0, -60.0, -100.0),
            (80.0,   40.0,  -85.0),
            (-70.0,  80.0,  -70.0),
        ],
        "drone_z_bounds": (-110.0, -50.0),      # 50..110m AGL — voo UAM em cidade
        "search_radius": (10.0, 65.0),
        "typical_building_height": 60.0,
        "safe_init_altitude": -120.0,
    },
    "neighborhood": {
        "name": "AirSimNH",
        "description": "Suburb delivery — voo realista 25-70m AGL (last-mile)",
        "ego_vantage_points": [
            (0.0,    0.0,   -35.0),
            (-30.0,  20.0,  -40.0),
            (40.0,  -30.0,  -45.0),
            (0.0,    60.0,  -30.0),
            (-50.0, -50.0,  -50.0),
        ],
        "drone_z_bounds": (-70.0, -25.0),       # 25..70m AGL — last-mile UAM
        "search_radius": (8.0, 60.0),
        "typical_building_height": 18.0,
        "safe_init_altitude": -80.0,
    },
    "citypark": {
        "name": "CityPark / LandscapeMountains",
        "description": "Edge-of-city -- flight above trees and hills",
        "ego_vantage_points": [
            (0.0,    0.0,   -55.0),
            (-80.0,  60.0,  -60.0),
            (50.0,  -50.0,  -50.0),
            (100.0,  0.0,   -70.0),
        ],
        "drone_z_bounds": (-100.0, -45.0),      # 45..100m AGL
        "search_radius": (10.0, 80.0),
        "typical_building_height": 35.0,
        "safe_init_altitude": -120.0,
    },
    "coastline": {
        "name": "Coastline",
        "description": "Coastal scenario -- recreational/inspection flight, 20-80 m AGL",
        "ego_vantage_points": [
            (0.0,    0.0,   -30.0),
            (60.0,   40.0,  -45.0),
            (-50.0,  80.0,  -25.0),
            (120.0, -30.0,  -40.0),
            (-100.0, -60.0, -55.0),
        ],
        "drone_z_bounds": (-80.0, -20.0),       # 20..80m AGL
        "search_radius": (8.0, 65.0),
        "typical_building_height": 10.0,
        "safe_init_altitude": -90.0,
    },
}

# =============================================================================
# CAMERA CONSTANTS -- must agree with the front_center block of settings.json
# =============================================================================
IMAGE_W = int(CONFIG.get("camera.width", 1280))
IMAGE_H = int(CONFIG.get("camera.height", 720))
FOV_H_DEG = float(CONFIG.get("camera.fov_degrees", 90))
FX = IMAGE_W / (2 * np.tan(np.radians(FOV_H_DEG / 2)))
# Square pixels (AirSim convention): FY = FX, the vertical FOV is not used.
# Calibrated against the API relative_pose vs. bbox: FY=640 matches exactly
# (an earlier FY=623 derived from the vertical FOV was off).
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2

CAMERA = "front_center"
IMG_TYPE = airsim.ImageType.Scene
DETECTION_RADIUS_CM = 15000  # 150m

# front_center camera in settings.json:
#   lever arm in the ego body frame: (X=0.35, Y=0, Z=-0.5) m
#   pitch=-15 deg (looking 15 deg below the horizon), roll=0, yaw=0
CAM_OFFSET_BODY = np.asarray(CONFIG.get("camera.offset_body", [0.35, 0.0, -0.5]), dtype=float)
CAM_PITCH_RAD = math.radians(float(CONFIG.get("camera.pitch_degrees", -15.0)))
# R(body->cam) = R_y(-cam_pitch); for cam_pitch = -15 deg this is R_y(+15 deg).
_a = -CAM_PITCH_RAD
_c, _s = math.cos(_a), math.sin(_a)
R_BODY_TO_CAM = np.array([
    [_c, 0.0, _s],
    [0.0, 1.0, 0.0],
    [-_s, 0.0, _c],
])

# Approximate extents of the NATIVE drone mesh (m): width, depth, height.
# Superseded for annotation by DRONE_BBOX_HALF below, which uses the VisQuad mesh.
DRONE_EXTENT = np.array([1.0, 1.0, 0.3])

# Vehicle spawn origins (X, Y, Z in NED metres) from settings.json.
# simSetVehiclePose writes a pose RELATIVE to the origin, so the origin has to be
# added back to obtain a global pose.  Configured in configs/default.yaml.
VEHICLE_ORIGINS = {
    name: tuple(float(v) for v in origin)
    for name, origin in CONFIG.get("airsim.vehicle_origins", {}).items()
}

# Measured extents of the VisQuad (Quadrotor1 at scale 4x), metres:
#   X = forward = 3.01, Y = right = 3.93, Z = down = 2.79
# obtained with dataset_generation/measure_quadrotor.py.  The 1.38 margin covers
# the spinning rotors that AirSim omits from its axis-aligned bounding box.
# Halving gives the annotated half-extents e = [2.08, 2.71, 1.93] m of Section 7.2.2.
DRONE_BBOX_HALF = (np.asarray(CONFIG.get("target.extents", [3.01, 3.93, 2.79]), dtype=float)
                   * float(CONFIG.get("target.rotor_margin", 1.38)) / 2)

# Segmentation: assign a unique stencil ID to the "drone" class; AirSim maps
# the ID to a colour.  See Section 7.2.3(i) for why this route was abandoned.
SEG_ID_DRONE = 200
# RGB colour discovered during calibration (set at runtime).
DRONE_SEG_COLOR = None

# =============================================================================
# DOMAIN RANDOMIZATION
# =============================================================================
# Only LIGHT levels are kept: fog_heavy (0.7) and rain_heavy washed everything out,
# leaving the drones invisible while their labels were still saved -- garbage for training.
WEATHER_PRESETS = [
    "clear", "clear", "clear", "clear",
    "fog_light",
    "rain_light",
    "dust_light",
]
# Avoid completely dark hours (the drone becomes a black pixel against a black sky).
TIME_OF_DAY_HOURS = [7, 9, 11, 13, 15, 17, 18]

# Image-quality threshold: reject the frame if the contrast is too low (excessive fog).
MIN_IMAGE_STD = 12.0


def apply_weather(client, weather):
    """Clear the weather and apply a preset. Returns the string that was applied."""
    for p in (airsim.WeatherParameter.Fog,
              airsim.WeatherParameter.Rain,
              airsim.WeatherParameter.Snow,
              airsim.WeatherParameter.Dust,
              airsim.WeatherParameter.Roadwetness,
              airsim.WeatherParameter.MapleLeaf,
              airsim.WeatherParameter.RoadLeaf):
        try:
            client.simSetWeatherParameter(p, 0.0)
        except Exception:
            pass

    presets = {
        "fog_light":  {airsim.WeatherParameter.Fog: 0.15},   # light; the drone is still visible
        "rain_light": {airsim.WeatherParameter.Rain: 0.25,
                       airsim.WeatherParameter.Roadwetness: 0.4},
        "dust_light": {airsim.WeatherParameter.Dust: 0.2},   # leve poeira
    }
    if weather in presets:
        try:
            client.simEnableWeather(True)
        except Exception:
            pass
        for p, v in presets[weather].items():
            try:
                client.simSetWeatherParameter(p, v)
            except Exception:
                pass
    return weather


def apply_time_of_day(client, hour):
    """Set a fixed time of day (sun position). celestial_clock_speed=0 -> the sun stands still."""
    try:
        client.simSetTimeOfDay(True, f"2024-06-15 {hour:02d}:00:00",
                               celestial_clock_speed=0)
    except Exception as e:
        print(f"  warning: simSetTimeOfDay failed ({e}) -- continuing without it")
    return hour


# =============================================================================
# COLLISION-AWARE POSITIONING
# =============================================================================

def yaw_quat(yaw_rad):
    """Quaternion for a yaw-only rotation (about the Z axis).
    cosysairsim takes the order (x_val, y_val, z_val, w_val) in its constructor."""
    h = float(yaw_rad) * 0.5
    return airsim.Quaternionr(0.0, 0.0, math.sin(h), math.cos(h))


def teleport(client, vehicle_name, x, y, z, yaw_rad=0.0):
    """Teleport a drone instantaneously. ignore_collision=True lets it pass
    through buildings during the search; the real collision check is done
    afterwards by comparing simGetCollisionInfo timestamps."""
    pose = airsim.Pose(
        airsim.Vector3r(float(x), float(y), float(z)),
        yaw_quat(yaw_rad),
    )
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)


def collided_since(client, vehicle_name, before_ts):
    """True if a new collision event occurred after `before_ts`."""
    info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    return info.has_collided and info.time_stamp > before_ts


def find_safe_pose(client, vehicle_name, ego_pos, ego_yaw_rad,
                   search_radius, z_bounds, fov_h_deg=FOV_H_DEG,
                   max_attempts=18, settle_s=0.18):
    """
    Find a collision-free (x, y, z) for the drone, inside the ego's field of view.

    Strategy:
      1) Sample (r, bearing) in polar coordinates relative to the ego.
      2) Limit the bearing to ~85% of the horizontal FOV, so the drone lands in the image.
      3) Sample z around the ego's z, clipped to the environment's z_bounds.
      4) Teleport with ignore_collision=True and wait for a physics tick.
      5) Compare the collision timestamp; on a new collision, climb 8 m and retry.
    """
    r_min, r_max = search_radius
    z_min, z_max = z_bounds
    half_fov = math.radians(fov_h_deg / 2) * 0.85

    # Snapshot the timestamp BEFORE any teleport -- this is what we compare against.
    before_info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    before_ts = before_info.time_stamp

    for _ in range(max_attempts):
        r = random.uniform(r_min, r_max)
        bearing = ego_yaw_rad + random.uniform(-half_fov, half_fov)
        x = ego_pos[0] + r * math.cos(bearing)
        y = ego_pos[1] + r * math.sin(bearing)
    # Relative altitude: drones can appear almost anywhere vertically in the image.
        z = ego_pos[2] + random.uniform(-12.0, 12.0)
        z = max(z_min, min(z_max, z))

    # Turn the drone's nose in a plausible direction (so the camera sees it side-on).
        drone_yaw = random.uniform(-math.pi, math.pi)
        teleport(client, vehicle_name, x, y, z, drone_yaw)
        time.sleep(settle_s)

        if not collided_since(client, vehicle_name, before_ts):
            return (x, y, z, drone_yaw)

    # Collided -- try HIGHER (above the approximate skyline) on the next iteration.
        before_ts = client.simGetCollisionInfo(vehicle_name=vehicle_name).time_stamp

    # Last attempt: force a position above the environment's typical skyline.
    z_safe = max(z_min, ego_pos[2] - 40.0)  # 40 m above the ego
    bearing = ego_yaw_rad + random.uniform(-half_fov, half_fov)
    r = random.uniform(r_min, (r_min + r_max) / 2)
    x = ego_pos[0] + r * math.cos(bearing)
    y = ego_pos[1] + r * math.sin(bearing)
    teleport(client, vehicle_name, x, y, z_safe, random.uniform(-math.pi, math.pi))
    time.sleep(settle_s)
    return (x, y, z_safe, 0.0)  # best effort


def ego_safe(client):
    info = client.simGetCollisionInfo(vehicle_name="Ego")
    # No robust before_ts available here -- fall back to has_collided alone.
    return not info.has_collided


# =============================================================================
# CAPTURA & PROCESSAMENTO
# =============================================================================

def depth_to_pointcloud(depth_array, max_depth=250.0):
    """Convert a planar depth image to a camera-frame point cloud.
    max_depth: filter removing the (saturated) sky while keeping terrain and objects."""
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth_array > 0.1) & (depth_array < max_depth)
    z = depth_array[valid]
    u_v = u[valid]
    v_v = v[valid]
    x = (u_v - CX) * z / FX
    y = (v_v - CY) * z / FY
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def capture_frame(client):
    """Grab RGB + DepthPlanar + Segmentation from the Ego in a single call."""
    responses = legacy_get_images(client, [
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
    ], "Ego")
    if (not responses[0].image_data_uint8) or (not responses[1].image_data_float):
        return None, None, None
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
        responses[0].height, responses[0].width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(
        responses[1].height, responses[1].width)
    if depth.shape != (IMAGE_H, IMAGE_W):
        depth = cv2.resize(depth, (IMAGE_W, IMAGE_H))
    seg = None
    if responses[2].image_data_uint8:
        # Keep the segmentation in the same RGB order the calibration saw (do NOT convert to BGR).
        seg = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8).reshape(
            responses[2].height, responses[2].width, 3).copy()
    return bgr, depth, seg


def calibrate_drone_seg_color(client, drone_targets, debug_dir=None):
    """
    Discover the drones' segmentation colour by DIFFERENCING (the method that worked in v6):
    1) Move the ego to (0,0,-50) and ALL drones to (500,500,-50), i.e. far away.
    2) Capture seg_empty (no drones in the scene).
    3) Move the first drone to (12, 0, -50).
    4) Captura seg_with
    5) changed pixels = drone; the most frequent colour is the drone's colour
    """
    if not drone_targets:
        return None
    from collections import Counter

    orig_poses = {"Ego": client.simGetVehiclePose("Ego")}
    for d in drone_targets:
        orig_poses[d] = client.simGetVehiclePose(d)

    cal_z = -50.0
    far = airsim.Pose(airsim.Vector3r(500, 500, cal_z), yaw_quat(0))
    client.simSetVehiclePose(airsim.Pose(
        airsim.Vector3r(0, 0, cal_z), yaw_quat(0)), True, "Ego")
    for d in drone_targets:
        client.simSetVehiclePose(far, True, d)
    # Hover so the drone neither falls nor rotates under gravity.
    try:
        client.hoverAsync(vehicle_name="Ego")
        for d in drone_targets:
            client.hoverAsync(vehicle_name=d)
    except Exception:
        pass
    time.sleep(1.2)

    def grab_seg():
        resp = legacy_get_images(client, [
            airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
        ], "Ego")
        return np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(
            resp[0].height, resp[0].width, 3).copy()

    try:
        seg_empty = grab_seg()
        cal = drone_targets[0]
        client.simSetVehiclePose(airsim.Pose(
            airsim.Vector3r(12, 0, cal_z), yaw_quat(0)), True, cal)
        try: client.hoverAsync(vehicle_name=cal)
        except Exception: pass
        time.sleep(1.0)
        seg_with = grab_seg()
    except Exception as e:
        print(f"  calibration failed: {e}")
        for name, pose in orig_poses.items():
            client.simSetVehiclePose(pose, True, name)
        return None

    # After hovering the ego should be stable. Use the CURRENT pose for an accurate projection.
    ep2 = client.simGetVehiclePose("Ego")
    dp2 = client.simGetVehiclePose(cal)
    ego_pos = np.array([ep2.position.x_val, ep2.position.y_val, ep2.position.z_val])
    ego_R = quat_to_rot(ep2.orientation)
    drone_world = np.array([dp2.position.x_val, dp2.position.y_val, dp2.position.z_val])
    proj = world_to_image(drone_world, ego_pos, ego_R)
    color = None
    if proj is not None:
        u, v = int(proj[0]), int(proj[1])
        # Take a 7x7 window centred on the projection. The drone is there (12 m away,
        # ~50 px wide), so at least the central pixel carries the drone's colour.
        # Compare each pixel with seg_empty at the SAME location: a changed colour = drone.
        half = 3
        y1 = max(0, v - half); y2 = min(seg_with.shape[0], v + half + 1)
        x1 = max(0, u - half); x2 = min(seg_with.shape[1], u + half + 1)
        with_roi = seg_with[y1:y2, x1:x2]
        empty_roi = seg_empty[y1:y2, x1:x2]
        # Pixels that changed = drone (at the centre this should be nearly all of them).
        changed_mask = np.any(with_roi != empty_roi, axis=2)
        if changed_mask.sum() > 0:
            drone_pixels = with_roi[changed_mask]
            cnt = Counter([tuple(int(x) for x in p) for p in drone_pixels])
            for col, n in cnt.most_common():
                if col != (0, 0, 0):
                    color = np.array(col, dtype=np.uint8)
                    break

    if debug_dir is not None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(debug_dir) / "calib_seg_empty.png"),
                    cv2.cvtColor(seg_empty, cv2.COLOR_RGB2BGR))
        seg_with_marked = cv2.cvtColor(seg_with, cv2.COLOR_RGB2BGR).copy()
        if proj is not None:
            cv2.circle(seg_with_marked, (int(proj[0]), int(proj[1])), 12, (0, 255, 255), 2)
        cv2.imwrite(str(Path(debug_dir) / "calib_seg_with_drone.png"), seg_with_marked)
        with open(Path(debug_dir) / "calib_color.txt", "w") as f:
            f.write(f"Drone seg color (RGB): "
                    f"{tuple(color.tolist()) if color is not None else None}\n")
            if proj is not None:
                f.write(f"Projected pixel: ({int(proj[0])}, {int(proj[1])})\n")

    for name, pose in orig_poses.items():
        client.simSetVehiclePose(pose, True, name)
    time.sleep(0.3)
    return color


def visibility_via_seg(seg_img, bbox, drone_color,
                       tol=15, min_pixels=4, frac_threshold=0.05):
    """
    True if at least min_pixels OR at least frac_threshold% of the box carries the drone colour.
    """
    if seg_img is None or drone_color is None:
        return None  # fall through to the depth heuristic
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(IMAGE_W - 1, x2); y2 = min(IMAGE_H - 1, y2)
    area = (x2 - x1) * (y2 - y1)
    if area < 6:
        return False
    roi = seg_img[y1:y2, x1:x2]
    diff = np.abs(roi.astype(int) - drone_color.astype(int)).sum(axis=2)
    drone_pixels = int((diff <= tol).sum())
    return drone_pixels >= min_pixels or drone_pixels >= frac_threshold * area


def extract_bbox(det):
    """Return (xmin, ymin, xmax, ymax) or None -- handles both .x_val and .x on box2D."""
    if not hasattr(det, "box2D") or not hasattr(det.box2D, "min"):
        return None
    m, M = det.box2D.min, det.box2D.max
    try:
        if hasattr(m, "x_val"):
            return (int(m.x_val), int(m.y_val), int(M.x_val), int(M.y_val))
        return (int(m.x), int(m.y), int(M.x), int(M.y))
    except Exception:
        return None


# =============================================================================
# 3D -> 2D PROJECTION (for a synthetic box when simGetDetections misses a target)
# =============================================================================
def quat_to_rot(q):
    """Quaternionr -> 3x3 rotation matrix (body to world)."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def world_to_image(point_world, ego_pos, ego_R):
    """Project an NED point to a pixel (u, v) plus depth. None if behind the camera."""
    rel_body = ego_R.T @ (point_world - ego_pos)
    rel_cam = R_BODY_TO_CAM @ (rel_body - CAM_OFFSET_BODY)
    depth_proj = rel_cam[0]
    if depth_proj <= 0.5:
        return None
    u = rel_cam[1] / depth_proj * FX + CX
    v = rel_cam[2] / depth_proj * FY + CY
    return u, v, depth_proj


class SyntheticDet:
    """Imita o suficiente de cosysairsim DetectionInfo: .name + .box2D.min/max."""
    class _Point:
        __slots__ = ('x_val', 'y_val')
        def __init__(self, x, y):
            self.x_val = x; self.y_val = y
    class _Box:
        __slots__ = ('min', 'max')

    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
        self.box2D = SyntheticDet._Box()
        self.box2D.min = SyntheticDet._Point(x1, y1)
        self.box2D.max = SyntheticDet._Point(x2, y2)


def find_drone_clusters_in_seg(seg_img, drone_colors, exclude_bboxes=None,
                                min_area=10, max_area=4000):
    """
    Find clusters of pixels carrying EXACTLY the drone_colors (discovered from
    the API box centres). Returns the boxes not covered by exclude_bboxes.
    """
    if not drone_colors:
        return []
    flat_keys = (seg_img[:, :, 0].astype(np.int32) << 16) | \
                (seg_img[:, :, 1].astype(np.int32) << 8) | \
                 seg_img[:, :, 2].astype(np.int32)
    mask = np.isin(flat_keys, list(drone_colors)).astype(np.uint8)
    if mask.sum() == 0:
        return []
    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    bboxes = []
    for lbl_id in range(1, n_labels):
        ys, xs = np.where(labels == lbl_id)
        if len(ys) < min_area or len(ys) > max_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bboxes.append((x1, y1, x2, y2))
    if exclude_bboxes:
        kept = []
        for bb in bboxes:
            bcx = (bb[0] + bb[2]) / 2; bcy = (bb[1] + bb[3]) / 2
            covered = False
            for (ax1, ay1, ax2, ay2) in exclude_bboxes:
                if ax1 - 5 <= bcx <= ax2 + 5 and ay1 - 5 <= bcy <= ay2 + 5:
                    covered = True
                    break
            if not covered:
                kept.append(bb)
        bboxes = kept
    return bboxes


def discover_drone_colors_via_depth(seg_img, depth_img, api_detections,
                                     drone_world_poses, ego_pos):
    """
    Discover the drones' segmentation colour DETERMINISTICALLY:
    for each API detection we know drone_world_pose, hence the expected distance
    from the ego. Inside the API box, pixels at depth ~ expected_dist ARE the drone.
    Sample the segmentation colour of those pixels to get the exact colour.
    Returns a set of colour keys (packed RGB integers).
    """
    from collections import Counter
    color_counts = Counter()
    for det in api_detections:
        if not hasattr(det, 'name') or det.name not in drone_world_poses:
            continue
        bb = extract_bbox(det)
        if bb is None: continue
        x1, y1, x2, y2 = bb
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(seg_img.shape[1]-1, x2); y2 = min(seg_img.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1: continue

        drone_world = drone_world_poses[det.name]
        expected_dist = float(np.linalg.norm(drone_world - ego_pos))
        tol = max(3.0, 0.20 * expected_dist)

        depth_roi = depth_img[y1:y2, x1:x2]
        seg_roi = seg_img[y1:y2, x1:x2]
        # Mask: physical pixels at the drone's distance
        drone_mask = (depth_roi > expected_dist - tol) & (depth_roi < expected_dist + tol)
        if drone_mask.sum() < 3:
            continue  # box with no physical pixels on the drone (drone against sky = saturated depth)
        drone_pixels = seg_roi[drone_mask]
        for px in drone_pixels:
            key = (int(px[0]) << 16) | (int(px[1]) << 8) | int(px[2])
            if key != 0:
                color_counts[key] += 1
    # Accept only colours seen at least three times (filters edge noise).
    return set(k for k, n in color_counts.items() if n >= 3)


def synthesize_detection(client, drone_name, ego_pos, ego_R, depth_img,
                         seg_img=None, margin_px=10, max_dist=90.0):
    """
    If simGetDetections missed this drone, try to build a box by projection.
    Physically confirm the drone is visible using segmentation when available
    (drone colour at the projected pixel), otherwise fall back to depth.
    """
    try:
        dp = client.simGetVehiclePose(vehicle_name=drone_name).position
    except Exception:
        return None
    drone_world = np.array([dp.x_val, dp.y_val, dp.z_val])
    proj = world_to_image(drone_world, ego_pos, ego_R)
    if proj is None:
        return None
    u_c, v_c, dist = proj
    if dist > max_dist:
        return None
    if not (margin_px <= u_c < IMAGE_W - margin_px and
            margin_px <= v_c < IMAGE_H - margin_px):
        return None
    u_i, v_i = int(u_c), int(v_c)
    half = 5

    # Check via segmentation (preferred) or depth (fallback)
    if seg_img is not None and DRONE_SEG_COLOR is not None:
        win = seg_img[max(0, v_i-half):min(IMAGE_H, v_i+half+1),
                      max(0, u_i-half):min(IMAGE_W, u_i+half+1)]
        diff = np.abs(win.astype(int) - DRONE_SEG_COLOR.astype(int)).sum(axis=2)
        if (diff <= 15).sum() == 0:
            return None  # the drone is not visible at the projected pixel
    else:
        win = depth_img[max(0, v_i-half):min(IMAGE_H, v_i+half+1),
                        max(0, u_i-half):min(IMAGE_W, u_i+half+1)]
        tol = max(2.5, 0.20 * dist)
        if ((win > dist - tol) & (win < dist + tol)).sum() == 0:
            return None

    # Tighter box (no overestimation -- the height multiplier used to be 1.5)
    w_px = (DRONE_EXTENT[1] / dist) * FX * 0.5
    h_px = (DRONE_EXTENT[2] / dist) * FY * 1.0
    x1 = max(0, int(u_c - w_px))
    y1 = max(0, int(v_c - h_px))
    x2 = min(IMAGE_W - 1, int(u_c + w_px))
    y2 = min(IMAGE_H - 1, int(v_c + h_px))
    if (x2 - x1) * (y2 - y1) < 9:
        return None
    return SyntheticDet(drone_name, x1, y1, x2, y2)


def visibility_status(client, det, depth, ego_pos,
                      tol_frac=0.20, tol_min=2.5,
                      occluder_threshold=0.35, min_area=9):
    """
    Returns 'ok' | 'occluded' | 'too_small' | 'no_pose' | 'no_bbox'.
    Stricter than the previous filter:

      - Count the box pixels showing something CLOSER than (drone_dist - tol).
        Those pixels are "occluders" (a building or tree in front).
      - If occluder_fraction > occluder_threshold (35% by default), the target is occluded.
      - Sky pixels (depth >= 100 m) do NOT count as occluders (drone against sky = fine).
      - Pixels at the drone's own depth do NOT count either.
    """
    bbox = extract_bbox(det)
    if bbox is None:
        return ('no_bbox', None)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(IMAGE_W - 1, x2); y2 = min(IMAGE_H - 1, y2)
    area = (x2 - x1) * (y2 - y1)
    if area < min_area:
        return ('too_small', area)

    try:
        dp = client.simGetVehiclePose(vehicle_name=det.name).position
        drone_world = np.array([dp.x_val, dp.y_val, dp.z_val])
        expected_dist = float(np.linalg.norm(drone_world - ego_pos))
    except Exception:
        return ('no_pose', None)

    roi = depth[y1:y2, x1:x2]
    tol = max(tol_min, tol_frac * expected_dist)
    # Occluder = pixel with a valid depth AND significantly closer than the drone
    occluder = (roi > 0.1) & (roi < expected_dist - tol)
    occluder_fraction = float(occluder.sum()) / area
    if occluder_fraction > occluder_threshold:
        return ('occluded', occluder_fraction)
    return ('ok', occluder_fraction)


def is_visible(client, det, depth, ego_pos):
    """Boolean wrapper kept for backward compatibility."""
    return visibility_status(client, det, depth, ego_pos)[0] == 'ok'


def compute_drone_bbox_manual(drone_pos_world, drone_orient_quat,
                               ego_pos, ego_R,
                               extent=(0.85, 0.85, 0.20)):
    """Compute a tight 2D box by projecting the 8 corners of the mesh's local AABB.

    Does NOT use simGetDetections, whose AABB includes the inflated collision shape.
    The projection is done manually from the drone pose plus the camera intrinsics.

    Args:
        drone_pos_world: (x, y, z) in NED -- the drone's position
        drone_orient_quat: tuple (w, x, y, z) -- its orientation
        ego_pos: (x, y, z) ego no mundo
        ego_R: 3x3 rotation matrix body→world do ego
        extent: mesh half-extents in metres (x_half, y_half, z_half)
                Default (0.85, 0.85, 0.20) = Quadrotor1 ~4x (1.7m × 1.7m × 0.4m)

    Returns:
        (x_min, y_min, x_max, y_max) in pixels, or None if behind the camera.
    """
    ex, ey, ez = extent
    # 8 corners no frame local do drone (NED conventions)
    corners_local = np.array([
        [+ex, +ey, +ez], [-ex, +ey, +ez], [+ex, -ey, +ez], [-ex, -ey, +ez],
        [+ex, +ey, -ez], [-ex, +ey, -ez], [+ex, -ey, -ez], [-ex, -ey, -ez],
    ])
    # Local -> world rotation from the drone quaternion
    w, x, y, z = drone_orient_quat
    R_drone = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    pixels = []
    any_in_front = False
    for c in corners_local:
        corner_world = drone_pos_world + R_drone @ c
        proj = world_to_image(corner_world, ego_pos, ego_R)
        if proj is None:
            continue
        u, v, d = proj
        if d > 0.5:
            any_in_front = True
        pixels.append((u, v))
    if not any_in_front or len(pixels) < 2:
        return None
    us = [p[0] for p in pixels]
    vs = [p[1] for p in pixels]
    return (min(us), min(vs), max(us), max(vs))


def refine_bbox_via_depth(bbox_loose, depth, drone_dist,
                           tol_pct=0.10, tol_min=1.0, expand_pct=0.5):
    """Refine a loose AABB box into a tight box, using depth.

    The simGetDetections AABB is the projection of the 3D AABB vertices, so it comes out
    fat and off-centre. Depth filter: pixels at depth ~ drone_dist belong TO THE DRONE;
    the background has a different depth. The tight box is the min/max of those pixels.

    Args:
        bbox_loose: (x1, y1, x2, y2) bbox solto
        depth: array (H, W) depth planar em metros
        drone_dist: distance from the drone to the camera (relative_pose magnitude)
        tol_pct: tolerance as a percentage of the distance
        tol_min: minimum absolute tolerance (m)
        expand_pct: how much to expand the box before searching (margin)

    Returns:
        (bbox_tight, n_drone_pixels) — bbox_tight no formato (x1,y1,x2,y2).
        If refinement is not possible (too few pixels), returns bbox_loose.
    """
    H, W = depth.shape
    x1, y1, x2, y2 = [int(v) for v in bbox_loose]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return bbox_loose, 0
    mx = max(8, int((x2 - x1) * expand_pct))
    my = max(8, int((y2 - y1) * expand_pct))
    xs1 = max(0, x1 - mx); ys1 = max(0, y1 - my)
    xs2 = min(W, x2 + mx); ys2 = min(H, y2 + my)
    sub = depth[ys1:ys2, xs1:xs2]
    tol = max(tol_min, drone_dist * tol_pct)
    mask = (sub > drone_dist - tol) & (sub < drone_dist + tol)
    n = int(mask.sum())
    if n < 4:
        return bbox_loose, n
    ys, xs = np.where(mask)
    return (xs1 + int(xs.min()), ys1 + int(ys.min()),
            xs1 + int(xs.max()), ys1 + int(ys.max())), n


def build_labels(detections, depth, client=None, ego_pos=None, ego_R=None,
                 drone_world_poses=None, drone_world_quats=None,
                 stats=None, seg=None,
                 refine_bbox=False):  # off by default -- the manual box is already tight
    """
    Produce YOLO + 3D labels for one frame. Filters:
      1) Cross-check: the drone's 3D projection must agree with the API box centre
         (rejeita se API retornou bbox stale/de pose desatualizada). Tol: 25px.
      2) Visibility: the pixel at the box centre has depth ~ the drone's true distance.
         If MUCH smaller (something in front), the target is occluded.
      3) Se refine_bbox=True, refina o bbox AABB (gordo) → bbox apertado via depth
         filter (pixels at depth ~ the drone's distance).
    Uses the poses from the paused snapshot (drone_world_poses), so there are no races.
    """
    yolo_lines = []
    labels_3d = []
    viz_boxes = []

    # Filter: if ANY VisQuad_* appears in the detections, keep only the visual meshes
    # (the underlying multirotor is also detected as mesh "Quadrotor1" and would duplicate).
    has_visual = any(getattr(d, 'name', '').startswith("VisQuad_") for d in detections)
    if has_visual:
        detections = [d for d in detections if getattr(d, 'name', '').startswith("VisQuad_")]

    for det in detections:
        name = det.name if hasattr(det, 'name') else "drone"
        # Map VisQuad_DroneX -> DroneX (the visual object synchronised with the multirotor)
        if name.startswith("VisQuad_"):
            name = name[len("VisQuad_"):]

        # ====== 2D box = CENTRE of the AirSim AABB + size computed from distance ======
        # The AirSim AABB is inflated by the collision shape, BUT its CENTRE is correct.
        # So: keep the centre, replace the size by the projected size of the visual mesh.
        bbox = extract_bbox(det)
        if bbox is None:
            continue
        x1a, y1a, x2a, y2a = bbox
        cx = (x1a + x2a) / 2.0
        cy = (y1a + y2a) / 2.0

        # Minimal filter: reject only if the drone is behind the camera (fwd < 0).
        # det has already discarded drones outside the image (valid 2D box).
        if drone_world_poses and name in drone_world_poses and ego_pos is not None and ego_R is not None:
            d_global = drone_world_poses[name]
            rel_world = d_global - ego_pos
            rel_body = ego_R.T @ rel_world
            rel_cam_body = rel_body - CAM_OFFSET_BODY
            ang = -CAM_PITCH_RAD
            rc, rs = math.cos(ang), math.sin(ang)
            x_fwd = rc*rel_cam_body[0] + rs*rel_cam_body[2]
            if x_fwd < 0.5:
                if stats is not None: stats['behind_cam'] = stats.get('behind_cam', 0) + 1
                continue
        api_dist_for_size = None
        try:
            rp = det.relative_pose.position
            api_dist_for_size = float(np.sqrt(rp.x_val**2 + rp.y_val**2 + rp.z_val**2))
        except Exception:
            pass
        if api_dist_for_size is not None and api_dist_for_size > 0.5:
            # Projected physical size of the drone: w_px = (DRONE_W / dist) * FX
            # Medidos via simSpawnObject('Quadrotor1', scale=4x) + simGetDetections.box3D:
            #   extent X=0.48 m, Y=5.45 m, Z=1.49 m (propellers included)
            # W = max horizontal projection over any yaw = sqrt(X^2 + Y^2) ~ 5.5 m
            # H = Z + folga p/ tilt do drone em manobras
            DRONE_PHYS_W = 5.50
            DRONE_PHYS_H = 2.00
            w_px = DRONE_PHYS_W / api_dist_for_size * FX
            h_px = DRONE_PHYS_H / api_dist_for_size * FY
            bbox = (cx - w_px / 2.0, cy - h_px / 2.0,
                    cx + w_px / 2.0, cy + h_px / 2.0)

        x1, y1, x2, y2 = bbox
        x1 = int(max(0, x1)); y1 = int(max(0, y1))
        x2 = int(min(IMAGE_W - 1, x2)); y2 = int(min(IMAGE_H - 1, y2))
        bbox = (x1, y1, x2, y2)
        area = (x2 - x1) * (y2 - y1)
        if area < 25:
            if stats is not None: stats['too_small'] = stats.get('too_small', 0) + 1
            continue

        # ====== Distance comes straight from the API (relative_pose), not from depth ======
        api_dist = None
        try:
            rp = det.relative_pose.position
            api_dist = float(np.sqrt(rp.x_val**2 + rp.y_val**2 + rp.z_val**2))
        except Exception:
            pass

        # ====== MINIMAL filter: reject only STRONG occlusion (>=60% closer pixels) ======
        if api_dist is not None:
            depth_roi = depth[y1:y2, x1:x2]
            tol = max(3.0, 0.25 * api_dist)
            occluder_mask = (depth_roi > 0.1) & (depth_roi < api_dist - tol)
            occluder_frac = float(occluder_mask.sum()) / max(1, area)
            if occluder_frac > 0.60:
                if stats is not None: stats['occluded'] = stats.get('occluded', 0) + 1
                continue
        if stats is not None: stats['ok'] = stats.get('ok', 0) + 1

        # ====== Refinamento via depth: bbox AABB (gordo) → bbox apertado ======
        if refine_bbox and api_dist is not None:
            bbox_tight, n_drone_px = refine_bbox_via_depth(bbox, depth, api_dist)
            if n_drone_px >= 4:
                bbox = bbox_tight
                if stats is not None:
                    stats['refined'] = stats.get('refined', 0) + 1

        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(IMAGE_W - 1, x_max); y_max = min(IMAGE_H - 1, y_max)
        if x_max - x_min < 3 or y_max - y_min < 3:
            continue

        # YOLO (normalizado)
        xc = (x_min + x_max) / 2 / IMAGE_W
        yc = (y_min + y_max) / 2 / IMAGE_H
        bw = (x_max - x_min) / IMAGE_W
        bh = (y_max - y_min) / IMAGE_H
        if not (0 < bw < 1 and 0 < bh < 1):
            continue
        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # MANUAL 3D label (does not use det.relative_pose / det.box3D -- AirSim has a
        # pitch bug that offsets ~5 m in down, and box3D is undersized).
        # Frame: CV camera (x=right, y=down, z=fwd) -- the same as the raw point cloud.
        depth_str = "?"
        if (drone_world_poses and name in drone_world_poses
                and ego_pos is not None and ego_R is not None):
            d_global = drone_world_poses[name]
            d_quat = drone_world_quats.get(name, (1.0, 0.0, 0.0, 0.0))
            # quat -> yaw (the drone hovers with roll = pitch = 0)
            w, x, y, z = d_quat
            d_yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

            # Centre: world -> ego body -> camera (pitch + lever arm) -> reorder into CV
            rel_world = d_global - ego_pos
            rel_body = ego_R.T @ rel_world
            rel_cam_body = rel_body - CAM_OFFSET_BODY
            ang = -CAM_PITCH_RAD
            rc, rs = math.cos(ang), math.sin(ang)
            c_fwd = rc*rel_cam_body[0] + rs*rel_cam_body[2]
            c_right = rel_cam_body[1]
            c_down = -rs*rel_cam_body[0] + rc*rel_cam_body[2]
            center_cv = [float(c_right), float(c_down), float(c_fwd)]
            dist_cv = float(np.sqrt(c_right*c_right + c_down*c_down + c_fwd*c_fwd))

            # 8 corners of the body AABB (rotated by the drone yaw) -> world -> CV
            ex, ey, ez = DRONE_BBOX_HALF
            local = np.array([
                [+ex, +ey, +ez], [-ex, +ey, +ez], [-ex, -ey, +ez], [+ex, -ey, +ez],
                [+ex, +ey, -ez], [-ex, +ey, -ez], [-ex, -ey, -ez], [+ex, -ey, -ez],
            ])
            cd, sd = math.cos(d_yaw), math.sin(d_yaw)
            R_d = np.array([[cd, -sd, 0], [sd, cd, 0], [0, 0, 1]])
            corners_world = (R_d @ local.T).T + d_global
            corners_cv = []
            for cw in corners_world:
                rb = ego_R.T @ (cw - ego_pos) - CAM_OFFSET_BODY
                cf = rc*rb[0] + rs*rb[2]
                cr = rb[1]
                cd2 = -rs*rb[0] + rc*rb[2]
                corners_cv.append([float(cr), float(cd2), float(cf)])
            corners_arr = np.array(corners_cv)
            # Filter: reject if any corner is behind the camera (fwd <= 0);
            # the projection degenerates and the box looks like "2 points" in the PLY.
            if (corners_arr[:, 2] < 0.5).any():
                if stats is not None:
                    stats['corner_behind'] = stats.get('corner_behind', 0) + 1
                continue
            bmin_cv = corners_arr.min(axis=0).tolist()
            bmax_cv = corners_arr.max(axis=0).tolist()

            label_3d = {
                "class": "drone",
                "name": name,
                "bbox_2d": [x_min, y_min, x_max, y_max],
                "center": center_cv,
                "distance_m": dist_cv,
                "box3D_min": [float(v) for v in bmin_cv],
                "box3D_max": [float(v) for v in bmax_cv],
                "corners": [[float(v) for v in c] for c in corners_cv],
                "drone_yaw_rad": float(d_yaw),
            }
            labels_3d.append(label_3d)
            depth_str = f"{dist_cv:.1f}m"
        viz_boxes.append((x_min, y_min, x_max, y_max, depth_str))

    return yolo_lines, labels_3d, viz_boxes


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Urban UAM dataset collector for three environments")
    ap.add_argument("--env", required=True, choices=list(ENV_PROFILES.keys()),
                    help="environment currently running in AirSim")
    ap.add_argument("--frames", type=int, default=1000, help="target number of frames WITH detections")
    ap.add_argument("--output", type=str, default=None,
                    help="output directory (default: dataset_urban_<env>)")
    ap.add_argument("--ip", type=str, default=airsim_host(), help="IP address of the AirSim RPC server")
    ap.add_argument("--port", type=int, default=airsim_port())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaws_per_scenario", type=int, default=4,
                    help="ego yaw offsets per placement (more yaw = more varied frames)")
    ap.add_argument("--scenario_change_every", type=int, default=4,
                    help="re-position ego and drones every N frames")
    ap.add_argument("--weather_change_every", type=int, default=40,
                    help="change weather / time of day every N frames")
    ap.add_argument("--visual_scale", type=float, default=4.0,
                    help="scale of the visual Quadrotor1 attached to each drone (default 4x). "
                         "Use 1.0 to disable it (the native drone mesh stays visible).")
    ap.add_argument("--start_frame", type=int, default=0,
                    help="initial offset of the frame counter (to resume after a crash). "
                         "Generates from start_frame to start_frame+frames.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    profile = ENV_PROFILES[args.env]
    output_dir = Path(args.output or f"dataset_urban_{args.env}")

    print("=" * 72)
    print(f"  URBAN DATASET GENERATOR | env={profile['name']}")
    print(f"  {profile['description']}")
    print("=" * 72)

    # --- Connection ---
    try:
        client = airsim.MultirotorClient(ip=args.ip, port=args.port)
        client.confirmConnection()
        vehicles = client.listVehicles()
        print(f"Connected. Vehicles: {vehicles}")
    except Exception as e:
        print(f"ERROR: could not connect to {args.ip}:{args.port} -> {e}")
        sys.exit(1)

    if "Ego" not in vehicles:
        print("ERROR: settings.json must declare an 'Ego' vehicle with a 'front_center' camera.")
        sys.exit(1)

    drone_targets = [v for v in vehicles if v != "Ego"]
    if not drone_targets:
        print("ERROR: no target drone found (Drone3/Drone4/Intruder1 or similar are required).")
        sys.exit(1)
    print(f"Drones-alvo: {drone_targets}")

    # --- Arm + control ---
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception:
            pass

    # --- Safe initial spawn (above the environment skyline) ---
    print(f"Moving all the drones to the initial safe altitude "
          f"(z={profile['safe_init_altitude']}m)...")
    init_z = profile["safe_init_altitude"]
    for i, v in enumerate(vehicles):
        # Espalha horizontalmente em volta da origem
        ang = 2 * math.pi * i / len(vehicles)
        teleport(client, v, 30.0 * math.cos(ang), 30.0 * math.sin(ang), init_z)
    time.sleep(1.5)

    # --- Directory layout ---
    output_dir.mkdir(exist_ok=True)
    yolo_dir = output_dir / "yolo"
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    pn_dir = output_dir / "pointnet"
    (pn_dir / "point_clouds").mkdir(parents=True, exist_ok=True)
    (pn_dir / "point_clouds_lidar").mkdir(parents=True, exist_ok=True)
    (pn_dir / "labels_3d").mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"; vis_dir.mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    # --- Enlarged visual drone (Quadrotor1 at 4x attached to each multirotor) ---
    visualizer = None
    if args.visual_scale and args.visual_scale != 1.0:
        print(f"\n[Visual] Attaching the Quadrotor1 mesh at {args.visual_scale}x to the drones...")
        visualizer = DroneVisualizer(client, drone_targets,
                                     scale=args.visual_scale, sync_hz=30)
        visualizer.attach()
        visualizer.start_sync()
        time.sleep(0.5)

    # --- Setup detection ---
    legacy_clear_detection(client, CAMERA, IMG_TYPE, "Ego")
    legacy_set_radius(client, CAMERA, IMG_TYPE, DETECTION_RADIUS_CM, "Ego")
    if visualizer is not None:
        # Detecta os Quadrotor1 spawnados (mesh = "Quadrotor1")
        legacy_add_mesh(client, CAMERA, IMG_TYPE, "Quadrotor1", "Ego")
        print("[Detection] mesh filter: Quadrotor1 (visual)")
    else:
        for d in drone_targets:
            legacy_add_mesh(client, CAMERA, IMG_TYPE, d, "Ego")

    # Occlusion filter: depth heuristic (validated in v4/v6).
    # The segmentation-based filter was removed as unstable (the diff picks up anti-aliasing noise).
    global DRONE_SEG_COLOR
    DRONE_SEG_COLOR = None
    print("Occlusion filter: depth-based (visibility_status)")

    # --- Loop principal ---
    target_frames = args.start_frame + args.frames
    total = 0
    with_dets = args.start_frame
    without_dets = 0
    safe_failures = 0
    filter_stats = {}  # contagem global de motivos de descarte
    current_weather = "clear"
    current_hour = 12
    apply_weather(client, current_weather)
    apply_time_of_day(client, current_hour)

    print(f"\nTarget: {target_frames} frames WITH detections (frames without detections are skipped)")
    print(f"Output: {output_dir.absolute()}\n")

    t0 = time.time()
    iter_count = 0
    while with_dets < target_frames:
        iter_count += 1
        # Hard timeout: 2x the expected duration (~0.5 s/frame on average) -- avoids an infinite loop
        if time.time() - t0 > args.frames * 2.0 + 300:
            print("\nTimeout -- the environment is probably producing no detections. Exiting.")
            break

        # ---- 1) Trocar weather/time-of-day periodicamente ----
        if iter_count % args.weather_change_every == 1:
            current_weather = random.choice(WEATHER_PRESETS)
            current_hour = random.choice(TIME_OF_DAY_HOURS)
            apply_weather(client, current_weather)
            apply_time_of_day(client, current_hour)
            print(f"  [env-randomization] weather={current_weather}  hour={current_hour:02d}h")

        # ---- 2) Reposicionar ego ----
        ego_xyz = random.choice(profile["ego_vantage_points"])
        ego_yaw_rad = random.uniform(-math.pi, math.pi)
        teleport(client, "Ego", *ego_xyz, ego_yaw_rad)
        time.sleep(0.3)
        if not ego_safe(client):
            # Climb 20 m and try again
            teleport(client, "Ego", ego_xyz[0], ego_xyz[1], ego_xyz[2] - 20, ego_yaw_rad)
            time.sleep(0.2)

        # ---- 3) Reposicionar drones de forma collision-aware ----
        placed = []
        ego_pos = np.array(ego_xyz, dtype=float)
        for d in drone_targets:
            pose = find_safe_pose(client, d, ego_pos, ego_yaw_rad,
                                  profile["search_radius"], profile["drone_z_bounds"])
            if pose is None:
                safe_failures += 1
            else:
                placed.append((d, pose))
                # hoverAsync holds the drone in place (prevents gravity drift)
                try:
                    client.hoverAsync(vehicle_name=d)
                except Exception:
                    pass

        if not placed:
            continue

        # Freeze the ego as well
        try:
            client.hoverAsync(vehicle_name="Ego")
        except Exception:
            pass
        time.sleep(0.4)  # let physics and rendering settle

        # ---- 4) Capture several ego yaw variations ----
        yaw_offsets = random.sample(range(-25, 26, 5), k=min(args.yaws_per_scenario, 11))
        for off_deg in yaw_offsets:
            if with_dets >= target_frames:
                break

            # RE-TELEPORT everything each frame to eliminate accumulated drift
            # (hoverAsync does not hold the drones reliably).
            for dn, drone_pose in placed:
                teleport(client, dn, drone_pose[0], drone_pose[1], drone_pose[2], drone_pose[3])
            new_yaw = ego_yaw_rad + math.radians(off_deg)
            teleport(client, "Ego", ego_xyz[0], ego_xyz[1], ego_xyz[2], new_yaw)
            time.sleep(0.18)

            # ====== CRITICAL: pause the scene so image, detections and poses are CONSISTENT ======
            # Without the pause the drones drift ~1 m in the 0.5 s spanned by the queries,
            # displacing the boxes from the rendered drones.
            try:
                client.simPause(True)
            except Exception:
                pass

            try:
                bgr, depth, seg = capture_frame(client)
                if bgr is None or depth is None:
                    continue

                # Sanity check: washed-out image
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if float(gray.std()) < MIN_IMAGE_STD:
                    without_dets += 1
                    total += 1
                    filter_stats['low_contrast'] = filter_stats.get('low_contrast', 0) + 1
                    continue

                # Detections AND poses from the SAME paused scene
                detections = legacy_get_detections(client, CAMERA, IMG_TYPE, "Ego")

                # IMPORTANT: simGetVehiclePose returns a pose RELATIVE to the origin
                # declared in settings.json. The origin must be added back to obtain the
                # GLOBAL pose, which is what the rendered scene shows (DroneVisualizer is pinned to it).
                ego_pose_now = client.simGetVehiclePose(vehicle_name="Ego")
                ego_origin_arr = np.array(VEHICLE_ORIGINS.get("Ego", (0.0, 0.0, 0.0)))
                ego_pos_now = np.array([ego_pose_now.position.x_val,
                                         ego_pose_now.position.y_val,
                                         ego_pose_now.position.z_val]) + ego_origin_arr
                ego_R_now = quat_to_rot(ego_pose_now.orientation)

                # Cache each drone's GLOBAL pose (consistent snapshot -- position + orientation quat)
                drone_world_poses = {}
                drone_world_quats = {}
                for dn in drone_targets:
                    try:
                        dp = client.simGetVehiclePose(vehicle_name=dn)
                        dn_origin = np.array(VEHICLE_ORIGINS.get(dn, (0.0, 0.0, 0.0)))
                        drone_world_poses[dn] = np.array([dp.position.x_val,
                                                          dp.position.y_val,
                                                          dp.position.z_val]) + dn_origin
                        drone_world_quats[dn] = (dp.orientation.w_val,
                                                  dp.orientation.x_val,
                                                  dp.orientation.y_val,
                                                  dp.orientation.z_val)
                    except Exception:
                        pass

                # LiDAR scan (SensorLocalFrame = the camera frame after alignment)
                lidar_pts = None
                try:
                    ld = client.getLidarData(lidar_name="LidarFront", vehicle_name="Ego")
                    if ld and ld.point_cloud and len(ld.point_cloud) >= 3:
                        lidar_pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
                except Exception:
                    pass
            finally:
                try:
                    client.simPause(False)
                except Exception:
                    pass
            # ====== fim do bloco pausado ======

            # No synthesis. Re-teleporting each frame should let the API detect them all.

            if not detections:
                without_dets += 1
                total += 1
                continue

            yolo_lines, labels_3d, viz_boxes = build_labels(
                detections, depth, client=client,
                ego_pos=ego_pos_now, ego_R=ego_R_now,
                drone_world_poses=drone_world_poses,
                drone_world_quats=drone_world_quats,
                stats=filter_stats, seg=seg)
            if not yolo_lines:
                without_dets += 1
                total += 1
                continue

            # === Save ===
            frame_name = f"frame_{with_dets:06d}"
            is_train = with_dets % 5 != 0
            split = "train" if is_train else "val"

            cv2.imwrite(str(yolo_dir / "images" / split / f"{frame_name}.jpg"), bgr)
            with open(yolo_dir / "labels" / split / f"{frame_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

            pc = depth_to_pointcloud(depth)
            np.save(pn_dir / "point_clouds" / f"{frame_name}.npy", pc)
            # LiDAR point cloud (same frame as the camera under the new alignment)
            if lidar_pts is not None and len(lidar_pts) > 0:
                np.save(pn_dir / "point_clouds_lidar" / f"{frame_name}.npy", lidar_pts)

            # Filter: each box must contain >= 5 point-cloud points inside its AABB
            # (rejects degenerate boxes that show up as "2 points" in the PLY).
            if labels_3d and len(pc) > 0:
                filtered = []
                for l in labels_3d:
                    bmin = np.array(l['box3D_min']); bmax = np.array(l['box3D_max'])
                    inside = ((pc[:, 0] >= bmin[0]) & (pc[:, 0] <= bmax[0]) &
                              (pc[:, 1] >= bmin[1]) & (pc[:, 1] <= bmax[1]) &
                              (pc[:, 2] >= bmin[2]) & (pc[:, 2] <= bmax[2]))
                    n_in = int(inside.sum())
                    if n_in >= 5:
                        l['n_pts_inside'] = n_in
                        filtered.append(l)
                    else:
                        filter_stats['pts_in_bbox_lt5'] = filter_stats.get('pts_in_bbox_lt5', 0) + 1
                labels_3d = filtered

            if labels_3d:
                with open(pn_dir / "labels_3d" / f"{frame_name}.json", "w") as f:
                    json.dump(labels_3d, f, indent=2)

            # Visualisation every 50 frames
            if with_dets % 50 == 0:
                viz = bgr.copy()
                for (x1, y1, x2, y2, ds) in viz_boxes:
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(viz, f"D:{ds}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(viz, f"env={args.env}  w={current_weather}  h={current_hour:02d}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(str(vis_dir / f"{frame_name}.jpg"), viz)

            # PER-FRAME metadata (true GLOBAL poses, consistent snapshot)
            meta = {
                "frame_id": int(with_dets),
                "env": args.env,
                "ego_xyz_vehicle": list(map(float, ego_xyz)),  # intent vehicle-pose
                "ego_xyz_global": [float(v) for v in ego_pos_now.tolist()],
                "ego_yaw_rad": float(new_yaw),
                "weather": current_weather,
                "hour": int(current_hour),
                "split": split,
                "n_detections": len(detections),
                "n_labels": len(yolo_lines),
                "drone_positions_vehicle": [
                    {"name": n, "xyz_yaw": list(map(float, p))} for n, p in placed
                ],
                "drone_positions_global": [
                    {"name": n,
                     "xyz": [float(v) for v in drone_world_poses.get(n, [0,0,0])],
                     "quat_wxyz": list(drone_world_quats.get(n, (1,0,0,0)))}
                    for n in drone_world_poses
                ],
            }
            with open(output_dir / "metadata" / f"{frame_name}.json", "w") as f:
                json.dump(meta, f, indent=2)

            with_dets += 1
            total += 1

            if with_dets % 50 == 0:
                rate = with_dets / max(1, total) * 100
                elapsed = time.time() - t0
                fps = with_dets / max(1, elapsed)
                print(f"  [{with_dets:5d}/{target_frames}]  total={total}  "
                      f"hit_rate={rate:5.1f}%  fps={fps:.2f}  "
                      f"safe_failures={safe_failures}")

    # --- dataset.yaml for YOLO ---
    dataset_yaml = (
        f"path: {yolo_dir.absolute()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names: ['drone']\n"
    )
    with open(yolo_dir / "dataset.yaml", "w") as f:
        f.write(dataset_yaml)

    n_train = len(list((yolo_dir / "labels" / "train").glob("*.txt")))
    n_val = len(list((yolo_dir / "labels" / "val").glob("*.txt")))

    if visualizer is not None:
        print("\n[Visual] Desanexando Quadrotor1 visuais...")
        visualizer.stop_sync()
        visualizer.detach()

    print("\n" + "=" * 72)
    print(f"  DONE  ({profile['name']})")
    print("=" * 72)
    print(f"  Total iterations:     {total}")
    print(f"  Frames with labels:  {with_dets}  (train={n_train}  val={n_val})")
    print(f"  Frames without det.: {without_dets}")
    print(f"  Falhas de safe-pos:  {safe_failures}")
    if filter_stats:
        print(f"  Occlusion filter:")
        for k, v in sorted(filter_stats.items(), key=lambda kv: -kv[1]):
            print(f"    {k:>12}: {v}")
    print(f"  Output:               {output_dir.absolute()}")
    print()
    print("  To merge with the other environments afterwards:")
    print(f"    rsync -a {output_dir}/yolo/  dataset_urban_merged/yolo/")
    print(f"    rsync -a {output_dir}/pointnet/  dataset_urban_merged/pointnet/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by the user.")
