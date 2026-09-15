#!/usr/bin/env python3
"""ARCHIVED. Drone tracking pipeline, multi-threaded -- the first end-to-end pipeline.

Architecture (a single AirSim client, to avoid conflicts):

  Main thread (AirSim calls) --> queue --> tracking thread --> queue --> main (visualisation)
       |
       +-- also sends drone movement via moveOnPathAsync

  When YOLO and PointNet are ready, a detection thread is inserted between main
  and tracking, turning the RGB image plus depth into 3D detections. In this
  version the ground truth from simGetDetections is used directly.

Tracking thread: the 3D SORT tracker with Kalman filters (sort_tracker_3d.py).
Superseded by common/pipeline.py.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
import time
import math
import threading
import queue
from pathlib import Path

from archive.early_detectors.sort_tracker_3d import Sort3D


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(w, x, y, z):
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def pose_to_pos_rot(pose):
    pos = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    q = pose.orientation
    R = quaternion_to_rotation_matrix(q.w_val, q.x_val, q.y_val, q.z_val)
    return pos, R


# Camera constants
IMAGE_W, IMAGE_H = 1280, 720
FOV_H_DEG = 90
FX = IMAGE_W / (2 * np.tan(np.radians(FOV_H_DEG / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2


def project_world_to_image(pos_world, ego_pos, ego_R):
    rel_world = pos_world - ego_pos
    rel_cam = ego_R.T @ rel_world
    if rel_cam[0] <= 0.1:
        return None
    u = FX * rel_cam[1] / rel_cam[0] + CX
    v = FY * rel_cam[2] / rel_cam[0] + CY
    if 0 <= u < IMAGE_W and 0 <= v < IMAGE_H:
        return int(u), int(v)
    return None


# ---------------------------------------------------------------------------
# Drone path generation
# ---------------------------------------------------------------------------

def generate_drone_path(pattern, n_points=300):
    """
    Generate waypoints inside Ego's FOV.
    Ego at (0,0,-12) facing +X, FOV=90deg.
    Drones at ~30-50m distance, same altitude as Ego (z=-12).
    """
    points = []
    for i in range(n_points):
        t = i * (2 * math.pi / 60)  # repeat every 60 pts
        if pattern == "ellipse":
            x = 35.0 + 8.0 * math.cos(t)           # 27..43
            y = 8.0 * math.sin(t)                   # -8..8
            z = -12.0 + 0.5 * math.sin(t * 2)      # -12.5..-11.5
        elif pattern == "figure8":
            x = 30.0 + 6.0 * math.sin(t)           # 24..36
            y = -5.0 + 6.0 * math.sin(t * 2)       # -11..1
            z = -12.0 + 0.3 * math.cos(t * 1.5)    # ~-12
        elif pattern == "zigzag":
            x = 40.0 + 8.0 * math.cos(t * 0.7)    # 32..48
            y = 5.0 + 6.0 * math.sin(t * 2.5)     # -1..11
            z = -12.0 + 0.5 * math.sin(t)          # -12.5..-11.5
        else:
            raise ValueError(pattern)
        points.append(airsim.Vector3r(x, y, z))
    return points


def start_drone_paths(client):
    """Move drones to start position (.join()), then send full path."""
    configs = {
        "Drone3":    ("ellipse",  4.0),
        "Drone4":    ("figure8",  3.5),
        "Intruder1": ("zigzag",   5.0),
    }
    futures = {}
    for name, (pattern, speed) in configs.items():
        wp = generate_drone_path(pattern)
        # Blocking move to first waypoint (like generate_dataset_1000.py)
        print(f"  {name} -> start ({wp[0].x_val:.0f}, {wp[0].y_val:.0f}, {wp[0].z_val:.0f})")
        client.moveToPositionAsync(
            wp[0].x_val, wp[0].y_val, wp[0].z_val, speed, vehicle_name=name
        ).join()
        # Non-blocking: follow entire path
        futures[name] = client.moveOnPathAsync(wp, speed, vehicle_name=name)
    return futures


# ---------------------------------------------------------------------------
# AirSim data acquisition (runs in main thread - single client)
# ---------------------------------------------------------------------------

CAMERA = "front_center"
IMG_TYPE = airsim.ImageType.Scene
DRONE_NAMES = ["Drone3", "Drone4", "Intruder1"]


def setup_detection(client):
    client.simClearDetectionMeshNames(CAMERA, IMG_TYPE, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(CAMERA, IMG_TYPE, 15000, vehicle_name="Ego")
    for name in DRONE_NAMES:
        client.simAddDetectionFilterMeshName(CAMERA, IMG_TYPE, name, vehicle_name="Ego")


def acquire_frame(client):
    """Get one frame: RGB + detections + ego pose. Called from main thread."""
    resp = client.simGetImages([
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
    ])
    rgb_resp = resp[0]
    img = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
    img = img.reshape(rgb_resp.height, rgb_resp.width, 3)

    ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
    ego_pos, ego_R = pose_to_pos_rot(ego_pose)

    raw_dets = client.simGetDetections(CAMERA, IMG_TYPE, vehicle_name="Ego")

    detections = []
    for det in raw_dets:
        # 2D bbox from detection API (replaces YOLO output)
        bbox_2d = None
        if hasattr(det, 'box2D') and hasattr(det.box2D, 'min'):
            if hasattr(det.box2D.min, 'x_val'):
                bbox_2d = [
                    int(det.box2D.min.x_val), int(det.box2D.min.y_val),
                    int(det.box2D.max.x_val), int(det.box2D.max.y_val),
                ]
        name = det.name if hasattr(det, 'name') else "unknown"

        # 3D position: use simGetVehiclePose directly (NOT relative_pose,
        # which is in an unclear coordinate frame and has vertical offset).
        # This gives clean NED world coords relative to vehicle's home.
        try:
            drone_pose = client.simGetVehiclePose(vehicle_name=name)
            pos_world = np.array([
                drone_pose.position.x_val,
                drone_pose.position.y_val,
                drone_pose.position.z_val,
            ])
        except Exception:
            # Fallback to relative_pose if vehicle name not found
            rp = det.relative_pose.position
            rel_cam = np.array([rp.x_val, rp.y_val, rp.z_val])
            pos_world = ego_pos + ego_R @ rel_cam

        # 2D bbox center for visualization
        bbox_center_2d = None
        if bbox_2d:
            bbox_center_2d = (
                (bbox_2d[0] + bbox_2d[2]) // 2,
                (bbox_2d[1] + bbox_2d[3]) // 2,
            )
        detections.append({
            'name': name,
            'bbox_2d': bbox_2d,
            'bbox_center_2d': bbox_center_2d,
            'pos_3d_world': pos_world,
            'confidence': 1.0,
        })

    return {
        'timestamp': time.time(),
        'image_rgb': img,
        'detections': detections,
        'ego_pos': ego_pos,
        'ego_R': ego_R,
    }


# ---------------------------------------------------------------------------
# Tracking Thread (no AirSim calls - pure processing)
# ---------------------------------------------------------------------------

class TrackingThread(threading.Thread):
    def __init__(self, in_queue, out_queue, stop_event):
        super().__init__(daemon=True, name="TrackingThread")
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.tracker = Sort3D(
            max_age=15, min_hits=2, distance_threshold=8.0,
            dt=0.1, sigma_acc=3.0, sigma_pos=0.5, outlier_threshold=25.0,
        )
        self.prev_ts = None
        self.trajectories: dict[int, list[np.ndarray]] = {}

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            ts = frame['timestamp']
            dt = (ts - self.prev_ts) if self.prev_ts else 0.1
            dt = np.clip(dt, 0.01, 1.0)
            self.prev_ts = ts

            dets = []
            for d in frame['detections']:
                p = d['pos_3d_world']
                dets.append([p[0], p[1], p[2], d['confidence']])
            dets = np.array(dets) if dets else np.empty((0, 4))

            tracks = self.tracker.update(dets, dt=dt)

            # Match each track to its closest detection to get bbox_center_2d
            track_info = []
            for trk in tracks:
                tid = int(trk[3])
                pos = trk[:3].copy()
                kt = self.tracker.get_tracker_by_id(tid)
                vel = kt.get_velocity() if kt else np.zeros(3)
                speed = float(np.linalg.norm(vel))

                # Find closest detection for this track's 2D center
                best_center_2d = None
                best_dist = float('inf')
                for d in frame['detections']:
                    dist = np.linalg.norm(pos - d['pos_3d_world'])
                    if dist < best_dist:
                        best_dist = dist
                        best_center_2d = d.get('bbox_center_2d')

                # Store 2D center in trajectory (for drawing on image)
                if tid not in self.trajectories:
                    self.trajectories[tid] = []
                if best_center_2d is not None and best_dist < 10.0:
                    self.trajectories[tid].append(best_center_2d)
                if len(self.trajectories[tid]) > 200:
                    self.trajectories[tid] = self.trajectories[tid][-200:]

                track_info.append({
                    'id': tid, 'pos': pos,
                    'velocity': vel, 'speed': speed,
                    'center_2d': best_center_2d,
                })

            result = {
                'timestamp': ts,
                'image_rgb': frame['image_rgb'],
                'detections': frame['detections'],
                'ego_pos': frame['ego_pos'],
                'ego_R': frame['ego_R'],
                'tracks': track_info,
                'trajectories': {k: list(v) for k, v in self.trajectories.items()},
                'stats': {
                    'active_tracks': len(tracks),
                    'total_created': self.tracker.total_tracks_created,
                    'n_detections': len(frame['detections']),
                },
            }

            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put_nowait(result)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

COLOR_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255),
]


def draw_tracking_frame(result):
    img = result['image_rgb'].copy()
    tracks = result['tracks']
    trajectories = result['trajectories']
    stats = result['stats']

    # Tracked points (no bboxes - tracker only produces 3D points)
    for trk in tracks:
        tid = trk['id']
        pos = trk['pos']
        speed = trk['speed']
        center_2d = trk.get('center_2d')
        color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]

        if center_2d is None:
            continue
        cx, cy = center_2d

        # Draw tracked point as circle at bbox center
        cv2.circle(img, (cx, cy), 6, color, -1)
        cv2.circle(img, (cx, cy), 8, color, 2)

        # Info label
        label_y = max(cy - 35, 15)
        overlay = img.copy()
        cv2.rectangle(overlay, (cx + 12, label_y - 10),
                      (cx + 160, label_y + 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.putText(img, f"ID:{tid}", (cx + 15, label_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(img, f"({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})m",
                    (cx + 15, label_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(img, f"{speed:.1f}m/s",
                    (cx + 15, label_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    # Trajectories (stored as 2D bbox centers - perfectly aligned)
    for tid, traj in trajectories.items():
        if len(traj) < 2:
            continue
        color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]
        recent = traj[-80:]
        for i in range(len(recent) - 1):
            alpha = (i + 1) / len(recent)
            c = tuple(int(alpha * ch) for ch in color)
            cv2.line(img, recent[i], recent[i + 1], c, 2)

    # HUD
    cv2.putText(img, f"Detections: {stats['n_detections']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Active tracks: {stats['active_tracks']}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Total tracks: {stats['total_created']}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DRONE TRACKING PIPELINE")
    print("  Detection (ground truth) + SORT 3D Tracker")
    print("=" * 60)

    # --- Single AirSim client for everything ---
    client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"Connected. Vehicles: {vehicles}")

    # Arm & takeoff
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception:
            pass

    print("Taking off...")
    for v in vehicles:
        try:
            client.takeoffAsync(vehicle_name=v).join()
        except Exception:
            pass
    time.sleep(3)

    # Ego to vantage point
    print("Positioning Ego...")
    client.moveToPositionAsync(0, 0, -12, 3, vehicle_name="Ego").join()
    time.sleep(1)

    # Setup detection filter (same client)
    setup_detection(client)

    # Send drones on paths (blocking .join() to start, then async path)
    print("\nSending drones on paths...")
    drone_futures = start_drone_paths(client)
    print("Drones moving!\n")

    # --- Queues & threads ---
    det_queue = queue.Queue(maxsize=5)
    trk_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    trk_thread = TrackingThread(det_queue, trk_queue, stop_event)
    trk_thread.start()
    print("Tracking thread started.")

    # Video output
    output_dir = Path("tracking_output")
    output_dir.mkdir(exist_ok=True)
    video_path = output_dir / "tracking.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (IMAGE_W, IMAGE_H)
    )

    frame_count = 0
    print(f"Recording to {video_path}  |  Press Ctrl+C to stop\n")

    try:
        while True:
            # --- Acquire data from AirSim (main thread, single client) ---
            try:
                frame_data = acquire_frame(client)
            except Exception as e:
                print(f"[Acquire] Error: {e}")
                time.sleep(0.05)
                continue

            # Push to tracking thread
            if det_queue.full():
                try:
                    det_queue.get_nowait()
                except queue.Empty:
                    pass
            det_queue.put_nowait(frame_data)

            # --- Read tracking results (non-blocking) ---
            try:
                result = trk_queue.get_nowait()
            except queue.Empty:
                continue

            vis = draw_tracking_frame(result)
            video_writer.write(vis)
            frame_count += 1

            if frame_count % 10 == 0:
                s = result['stats']
                print(f"  Frame {frame_count:5d}  |  "
                      f"det={s['n_detections']}  tracks={s['active_tracks']}  "
                      f"total={s['total_created']}")

    except KeyboardInterrupt:
        print("\nStopping...")

    # Shutdown
    stop_event.set()
    trk_thread.join(timeout=3)
    video_writer.release()

    # Save trajectories
    traj_path = output_dir / "trajectories.npz"
    traj_data = {}
    for tid, pts in trk_thread.trajectories.items():
        traj_data[f"track_{tid}"] = np.array(pts)
    if traj_data:
        np.savez(str(traj_path), **traj_data)
        print(f"Trajectories saved: {traj_path}")

    # Land
    print("Landing...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v).join()
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except Exception:
            pass

    print(f"\nDone! {frame_count} frames recorded.")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
