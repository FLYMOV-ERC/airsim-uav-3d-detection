#!/usr/bin/env python3
"""End-to-end inference pipeline (multi-threaded).

Threads:
  A. AcquisitionThread  — AirSim RGB + Depth + ego pose  ─→ q_raw
  B. DetectionThread    — YOLO bboxes + PointNet++ (per bbox) ─→ q_det
  C. FilterThread       -- MultiEKFTracker (association + EKF)   -> q_tracks
  M. Main               -- overlay (box + ID + 3D position + trail) -> MP4 + JSON

Pipeline data flow:
  bgr, depth → PC_cv
  per bbox: paint(PC_cv, bbox, yolo_conf) → sample 4096 → normalize → PointNet
  PointNet output → center_cv → cv_to_frd → spherical (d, phi, theta) → measurement
  the filter consumes the measurement list -> EKF predict + associate + update per track
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import airsim_host, airsim_port

import argparse
import asyncio
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# The AirSim client is optional: this module is imported by the offline half of
# the repository (pc_cache, the trainers, dump_voxel, the harness) purely for the
# camera model and the frame conversions.  See common/airsim_optional.py.
from common.airsim_optional import airsim, ImageResponse, require_airsim

from tracking.ekf_3d import EKFParams, rotation_matrix
from tracking.multi_ekf import MultiEKFTracker, MultiTrackerParams
from detection.base import Detector3D
from archive.early_detectors.detector_painted_v4 import PaintedPointNetDetector
from archive.early_detectors.detector_frustum_v1 import FrustumPointNetDetector
from detection.frustum.detector import FrustumPN2Detector
from archive.early_detectors.detector_depth_direct import DepthDirectDetector
from common.drone_visual import DroneVisualizer


def make_detector(method: str, **kwargs) -> Detector3D:
    if method == "painted":
        return PaintedPointNetDetector(**kwargs)
    elif method == "frustum":
        return FrustumPointNetDetector(**kwargs)
    elif method == "frustum_pn2":
        return FrustumPN2Detector(**kwargs)
    elif method == "depth":
        return DepthDirectDetector(**kwargs)
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Camera model and frame conversions
#
# These live in common/geometry.py, which has no AirSim and no Ultralytics
# dependency.  They are re-exported here because fifteen modules import them
# from this one.
# ---------------------------------------------------------------------------
from common.geometry import (                      # noqa: E402,F401
    IMAGE_W, IMAGE_H, FX, FY, CX, CY, CAMERA,
    CAMERA_PITCH_OFFSET, CAM_OFFSET_BODY, MIN_DEPTH, MAX_DEPTH,
    quat_to_rpy, airsim_pose_to_x_plat, depth_to_pointcloud_cv,
    cv_to_frd, frd_to_spherical, project_global_to_pixel,
)


# ---------------------------------------------------------------------------
# cosys-airsim <-> Microsoft AirSim 1.8 shim (4-argument simGetImages signature)
# ---------------------------------------------------------------------------
def legacy_get_images(client, requests, veh: str = "Ego", external: bool = False):
    require_airsim()
    raw = client.client.call('simGetImages', requests, veh, external)
    return [ImageResponse.from_msgpack(r) for r in raw]


# ─────────────────────────────────────────────────────────────────────────────
# Thread A -- AirSim acquisition
# ─────────────────────────────────────────────────────────────────────────────
class AcquisitionThread(threading.Thread):
    def __init__(self, ip: str, port: int, q_out: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True, name="Acq")
        self.ip = ip
        self.port = port
        self.q_out = q_out
        self.stop_event = stop_event
        self.frame_count = 0

    def run(self):
        require_airsim()
        # msgpack-rpc needs one client per thread (connections are not shareable)
        asyncio.set_event_loop(asyncio.new_event_loop())
        client = airsim.MultirotorClient(ip=self.ip, port=self.port)
        client.confirmConnection()
        print("[Acq] thread-local AirSim client connected")

        while not self.stop_event.is_set():
            try:
                client.simPause(True)
                try:
                    responses = legacy_get_images(client, [
                        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
                        airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, True, False),
                    ], "Ego")
                    pose = client.simGetVehiclePose("Ego")
                    try:
                        kine = client.getMultirotorState("Ego").kinematics_estimated
                        vel = (kine.linear_velocity.x_val,
                               kine.linear_velocity.y_val,
                               kine.linear_velocity.z_val)
                    except Exception:
                        vel = (0.0, 0.0, 0.0)
                finally:
                    client.simPause(False)
            except Exception as e:
                print(f"[Acq] error: {e}")
                time.sleep(0.1)
                continue

            if (not responses[0].image_data_uint8) or (not responses[1].image_data_float):
                continue

            rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
                responses[0].height, responses[0].width, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(
                responses[1].height, responses[1].width)
            if depth.shape != (IMAGE_H, IMAGE_W):
                depth = cv2.resize(depth, (IMAGE_W, IMAGE_H))
            pc_cv = depth_to_pointcloud_cv(depth, max_depth=250.0)

            x_plat = airsim_pose_to_x_plat(pose, velocity_vec=vel)
            pkt = {
                'timestamp': time.time(),
                'frame': self.frame_count,
                'bgr': bgr,
                'depth': depth,
                'pc_cv': pc_cv,
                'x_plat': x_plat,
            }
            _push_drop_old(self.q_out, pkt)
            self.frame_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Thread B -- detection (YOLO followed by PointNet)
# ─────────────────────────────────────────────────────────────────────────────
class DetectionThread(threading.Thread):
    def __init__(self, yolo: YOLO, detector: Detector3D,
                 q_in: queue.Queue, q_out: queue.Queue, stop_event: threading.Event,
                 yolo_conf: float = 0.25):
        super().__init__(daemon=True, name="Det")
        self.yolo = yolo
        self.det3d = detector
        self.q_in = q_in
        self.q_out = q_out
        self.stop_event = stop_event
        self.yolo_conf = yolo_conf

    def run(self):
        while not self.stop_event.is_set():
            try:
                pkt = self.q_in.get(timeout=0.5)
            except queue.Empty:
                continue

            bgr = pkt['bgr']
            pc_cv = pkt['pc_cv']
            depth = pkt.get('depth')

            # ── YOLO ──
            try:
                results = self.yolo.predict(bgr, conf=self.yolo_conf, imgsz=1280, verbose=False)
            except Exception as e:
                print(f"[Det] YOLO error: {e}")
                continue

            bboxes = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    bboxes.append({
                        'bbox': tuple(map(float, xyxy[i])),
                        'conf': float(confs[i]),
                    })

            # -- 3D detector per box + conversion to spherical --
            measurements = []
            for b in bboxes:
                pred = self.det3d.predict(pc_cv, depth, b['bbox'], b['conf'])
                if pred is None or not pred['is_drone']:
                    continue
                p_frd = cv_to_frd(pred['center_cv'])
                d, phi, theta = frd_to_spherical(p_frd)
                r = float(max(pred['size_cv']) / 2.0)
                r = float(np.clip(r, 0.1, 10.0))
                measurements.append({
                    'd': d, 'phi': phi, 'theta': theta, 'r': r,
                    'bbox': list(b['bbox']),
                    'yolo_conf': b['conf'],
                    'pn_prob': pred['is_drone_prob'],
                    'center_cv': pred['center_cv'].tolist(),
                    'size_cv': pred['size_cv'].tolist(),
                    'method': pred.get('method', '?'),
                })

            out = {
                'timestamp': pkt['timestamp'],
                'frame': pkt['frame'],
                'bgr': bgr,
                'x_plat': pkt['x_plat'],
                'measurements': measurements,
                'n_yolo': len(bboxes),
            }
            _push_drop_old(self.q_out, out)


# ─────────────────────────────────────────────────────────────────────────────
# Thread C -- multi-target EKF filter
# ─────────────────────────────────────────────────────────────────────────────
class FilterThread(threading.Thread):
    def __init__(self, q_in: queue.Queue, q_out: queue.Queue, stop_event: threading.Event,
                 tracker_params: MultiTrackerParams):
        super().__init__(daemon=True, name="Flt")
        self.q_in = q_in
        self.q_out = q_out
        self.stop_event = stop_event
        self.tracker = MultiEKFTracker(tracker_params)
        self.prev_ts = None

    def run(self):
        while not self.stop_event.is_set():
            try:
                pkt = self.q_in.get(timeout=0.5)
            except queue.Empty:
                continue

            ts = pkt['timestamp']
            dt = (ts - self.prev_ts) if self.prev_ts else 0.1
            dt = float(np.clip(dt, 0.01, 1.0))
            self.prev_ts = ts

            meas = [
                (m['d'], m['phi'], m['theta'], m['r'], m['yolo_conf'])
                for m in pkt['measurements']
            ]
            tracks = self.tracker.update(meas, pkt['x_plat'], dt)

            # Snapshot the histories (avoids a race: the tracker may mutate them afterwards)
            histories = {t.id: [p.tolist() for p in t.history[-60:]] for t in tracks}

            # Associate the 2D box of the nearest measurement (in 3D distance) to the track,
            # so the box is drawn in that track's colour
            track_bboxes = {}
            for t in tracks:
                best_dist, best_bbox = float('inf'), None
                for m in pkt['measurements']:
                    md, mp = m['d'], np.array(m['center_cv'])
                    # Distance in the CV frame: well approximated by the distance in FRD
                    p_cv_pred = np.array([
                        # Predicted track position in CV: world -> cam (FRD) -> CV
                        (rotation_matrix(pkt['x_plat'][6], pkt['x_plat'][7], pkt['x_plat'][8])
                         @ (np.array(t.position_global()) - pkt['x_plat'][:3]))[i] for i in (1, 2, 0)
                    ])
                    d_xyz = float(np.linalg.norm(p_cv_pred - mp))
                    if d_xyz < best_dist:
                        best_dist, best_bbox = d_xyz, m['bbox']
                if best_dist < 8.0 and best_bbox is not None:
                    track_bboxes[t.id] = best_bbox

            out = {
                'timestamp': ts,
                'frame': pkt['frame'],
                'bgr': pkt['bgr'],
                'x_plat': pkt['x_plat'],
                'measurements': pkt['measurements'],
                'tracks': [t.to_dict() for t in tracks],
                'histories': histories,
                'track_bboxes': track_bboxes,
                'stats': self.tracker.statistics(),
            }
            _push_drop_old(self.q_out, out)


# ─────────────────────────────────────────────────────────────────────────────
# Queue helpers
# ─────────────────────────────────────────────────────────────────────────────
def _push_drop_old(q: queue.Queue, item):
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
COLOR_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255),
]


def draw_overlay(pkt: dict) -> np.ndarray:
    img = pkt['bgr'].copy()
    x_plat = pkt['x_plat']
    measurements = pkt['measurements']
    tracks = pkt['tracks']
    histories = pkt['histories']
    track_bboxes = pkt['track_bboxes']

    # "Raw" boxes (grey) for measurements without a confirmed track
    matched_bboxes = set(tuple(b) for b in track_bboxes.values())
    for m in measurements:
        bb = tuple(m['bbox'])
        if bb in matched_bboxes:
            continue
        x1, y1, x2, y2 = [int(v) for v in m['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.putText(img, f"YOLO{m['yolo_conf']:.2f}|PN{m['pn_prob']:.2f}",
                    (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # Tracks
    for t in tracks:
        tid = t['id']
        color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]
        pos = np.array(t['position'])

        # Bbox do track (se existir)
        bbox = track_bboxes.get(tid)
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Marker at the projected centre
        u, v, depth = project_global_to_pixel(pos, x_plat)
        if u is None:
            continue
        u = int(np.clip(u, 0, IMAGE_W - 1))
        v = int(np.clip(v, 0, IMAGE_H - 1))
        cv2.circle(img, (u, v), 8, color, 2)
        cv2.circle(img, (u, v), 3, color, -1)

        # Projected trail
        traj = histories.get(tid, [])
        prev_pt = None
        for p_g in traj:
            up, vp, dp = project_global_to_pixel(np.array(p_g), x_plat)
            if up is None:
                prev_pt = None
                continue
            up = int(np.clip(up, 0, IMAGE_W - 1))
            vp = int(np.clip(vp, 0, IMAGE_H - 1))
            if prev_pt is not None:
                cv2.line(img, prev_pt, (up, vp), color, 2)
            prev_pt = (up, vp)

        # Labels
        spd = float(np.linalg.norm(t['velocity']))
        lines = [
            f"ID {tid}  ({t['hits']} hits)",
            f"({pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f})m",
            f"v={spd:.1f}m/s  r={t['radius']:.1f}m",
            f"d={depth:.1f}m",
        ]
        ly = max(v - 60, 12)
        lx = u + 14
        overlay = img.copy()
        cv2.rectangle(overlay, (lx - 4, ly - 14),
                      (lx + 200, ly + 4 + 14 * len(lines)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        for k, line in enumerate(lines):
            c = color if k == 0 else (255, 255, 255)
            cv2.putText(img, line, (lx, ly + k * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    # HUD
    stats = pkt['stats']
    hud = [
        f"Frame: {pkt['frame']}",
        f"Detections (PN positive): {len(measurements)}",
        f"Tracks: active={stats['active']}  confirmed={stats['confirmed']}  total={stats['total_created']}",
        f"Ego NED: ({x_plat[0]:+.1f},{x_plat[1]:+.1f},{x_plat[2]:+.1f})  "
        f"rpy=({np.degrees(x_plat[6]):.0f},{np.degrees(x_plat[7]):.0f},{np.degrees(x_plat[8]):.0f})°",
    ]
    for k, l in enumerate(hud):
        cv2.putText(img, l, (10, 24 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, l, (10, 24 + k * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Drones (live routes)
# ─────────────────────────────────────────────────────────────────────────────
def generate_drone_path(pattern: str, n_points: int = 300):
    require_airsim()
    points = []
    for i in range(n_points):
        t = i * (2 * math.pi / 60)
        if pattern == "ellipse":
            x = 35.0 + 8.0 * math.cos(t)
            y = 8.0 * math.sin(t)
            z = -12.0 + 0.5 * math.sin(t * 2)
        elif pattern == "figure8":
            x = 30.0 + 6.0 * math.sin(t)
            y = -5.0 + 6.0 * math.sin(t * 2)
            z = -12.0 + 0.3 * math.cos(t * 1.5)
        elif pattern == "zigzag":
            x = 40.0 + 8.0 * math.cos(t * 0.7)
            y = 5.0 + 6.0 * math.sin(t * 2.5)
            z = -12.0 + 0.5 * math.sin(t)
        else:
            raise ValueError(pattern)
        points.append(airsim.Vector3r(x, y, z))
    return points


def start_drones(client, drone_configs: dict):
    futures = {}
    for name, (pattern, speed) in drone_configs.items():
        wp = generate_drone_path(pattern)
        print(f"  {name} -> start ({wp[0].x_val:+.0f}, {wp[0].y_val:+.0f}, {wp[0].z_val:+.0f}) speed={speed}")
        try:
            client.moveToPositionAsync(wp[0].x_val, wp[0].y_val, wp[0].z_val,
                                       speed, vehicle_name=name).join()
        except Exception as e:
            print(f"    moveToPositionAsync({name}) failed: {e}")
            continue
        futures[name] = client.moveOnPathAsync(wp, speed, vehicle_name=name)
    return futures


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="runs/drone/drone_urban_v1/weights/best.pt")
    ap.add_argument("--method", choices=["painted", "frustum", "frustum_pn2", "depth"],
                    default="painted",
                    help="3D detector to run after YOLO")
    ap.add_argument("--pointnet", default=None,
                    help="override the neural detector checkpoint (per-method default otherwise)")
    ap.add_argument("--ip", default=airsim_host())
    ap.add_argument("--port", type=int, default=airsim_port())
    ap.add_argument("--output_dir", default="inference_output")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--ego_alt", type=float, default=-12.0)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--pn_thr", type=float, default=0.5)
    ap.add_argument("--gate_chi2", type=float, default=13.28)
    ap.add_argument("--max_age", type=int, default=10)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--no_takeoff", action="store_true",
                    help="skip arm/takeoff/moveOnPath (assume the drones are already flying)")
    ap.add_argument("--no_drone_paths", action="store_true",
                    help="do not send waypoints to the drones, only observe")
    ap.add_argument("--visual_scale", type=float, default=4.0,
                    help="scale of the visual Quadrotor1 (VisQuad) attached to each drone. "
                         "Must match what YOLO saw during training (4x). 1.0 disables it.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  END-TO-END INFERENCE PIPELINE")
    print("  YOLO + PointNet++ Painted + EKF 3D Multi-target Tracker")
    print("=" * 64)

    # ── Modelos ──
    print(f"\n[Models] Loading YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)
    method_defaults = {
        "painted":     "runs/pointnet2_painted/v4_balanced/best.pt",
        "frustum":     "runs/pointnet/drone_detector/best.pt",
        "frustum_pn2": "runs/pointnet2_frustum/v1/best.pt",
        "depth":       None,
    }
    ckpt = args.pointnet or method_defaults[args.method]
    print(f"[Models] 3D detector method: {args.method}  ckpt={ckpt or '(none)'}")
    if args.method in ("painted", "frustum", "frustum_pn2"):
        detector = make_detector(args.method, checkpoint_path=ckpt, cls_threshold=args.pn_thr)
    else:
        detector = make_detector("depth")

    # ── AirSim ──
    require_airsim()
    print(f"\n[AirSim] Connecting {args.ip}:{args.port}")
    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"[AirSim] Vehicles: {vehicles}")

    if not args.no_takeoff:
        print("[AirSim] Arming + takeoff...")
        for v in vehicles:
            try:
                client.enableApiControl(True, v)
                client.armDisarm(True, v)
            except Exception:
                pass
        for v in vehicles:
            try:
                client.takeoffAsync(vehicle_name=v).join()
            except Exception as e:
                print(f"  takeoff({v}) failed: {e}")
        time.sleep(2)
        print(f"[AirSim] Ego → (0, 0, {args.ego_alt})")
        client.moveToPositionAsync(0, 0, args.ego_alt, 3, vehicle_name="Ego").join()
        time.sleep(1)

        if not args.no_drone_paths:
            drone_cfg = {
                "Drone3":    ("ellipse",  4.0),
                "Drone4":    ("figure8",  3.5),
                "Intruder1": ("zigzag",   5.0),
            }
            drone_cfg = {k: v for k, v in drone_cfg.items() if k in vehicles}
            if drone_cfg:
                print(f"\n[AirSim] Drones on paths: {list(drone_cfg.keys())}")
                start_drones(client, drone_cfg)
                time.sleep(1)

    # -- Visual 4x (VisQuad): YOLO was trained on this mesh; without it nothing is detected --
    viz = None
    if args.visual_scale and args.visual_scale != 1.0:
        target_drones = [v for v in vehicles if v != "Ego"]
        if target_drones:
            print(f"\n[Visual] Anexando Quadrotor1 {args.visual_scale}x a {target_drones}")
            viz = DroneVisualizer(client, target_drones, scale=args.visual_scale, sync_hz=30)
            viz.attach()
            viz.start_sync()
            time.sleep(1)

    # ── Threads & queues ──
    q_raw = queue.Queue(maxsize=2)
    q_det = queue.Queue(maxsize=2)
    q_trk = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    t_acq = AcquisitionThread(args.ip, args.port, q_raw, stop_event)
    t_det = DetectionThread(yolo, detector, q_raw, q_det, stop_event, yolo_conf=args.yolo_conf)
    tracker_params = MultiTrackerParams(
        chi2_gate=args.gate_chi2,
        max_age=args.max_age,
        min_hits=args.min_hits,
        ekf_params=EKFParams(),
    )
    t_flt = FilterThread(q_det, q_trk, stop_event, tracker_params=tracker_params)
    t_acq.start(); t_det.start(); t_flt.start()
    print("[Pipeline] Threads started: Acquisition + Detection + Filter\n")

    # ── Writers ──
    video_path = out / "pipeline_output.mp4"
    json_path = out / "tracks_log.json"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             10, (IMAGE_W, IMAGE_H))
    json_log = []
    t_start = time.time()
    frame_count = 0

    try:
        while time.time() - t_start < args.duration:
            try:
                pkt = q_trk.get(timeout=1.0)
            except queue.Empty:
                continue
            vis = draw_overlay(pkt)
            writer.write(vis)
            frame_count += 1

            log_meas = []
            for m in pkt['measurements']:
                log_meas.append({k: v for k, v in m.items() if k not in ('center_cv', 'size_cv')})
            json_log.append({
                'timestamp': pkt['timestamp'],
                'frame': pkt['frame'],
                'ego': pkt['x_plat'].tolist(),
                'n_measurements': len(pkt['measurements']),
                'measurements': log_meas,
                'tracks': pkt['tracks'],
                'stats': pkt['stats'],
            })

            if frame_count % 10 == 0:
                elapsed = max(time.time() - t_start, 1e-3)
                s = pkt['stats']
                print(f"  Frame {frame_count:5d}  fps={frame_count/elapsed:.1f}  "
                      f"det={len(pkt['measurements'])}  "
                      f"tracks={s['active']}  confirmed={s['confirmed']}  total={s['total_created']}")
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")

    print("\n[Pipeline] Stopping threads...")
    stop_event.set()
    for t in [t_acq, t_det, t_flt]:
        t.join(timeout=3)
    writer.release()
    if viz is not None:
        viz.stop_sync(); viz.detach()

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"non-serializable: {type(o)}")

    with open(json_path, "w") as f:
        json.dump(json_log, f, indent=2, default=_json_default)
    print(f"[Done] {frame_count} frames")
    print(f"       video: {video_path}")
    print(f"       log:   {json_path}")

    if not args.no_takeoff:
        print("[AirSim] Landing...")
        for v in vehicles:
            try:
                client.landAsync(vehicle_name=v).join()
                client.armDisarm(False, v)
                client.enableApiControl(False, v)
            except Exception:
                pass


if __name__ == "__main__":
    main()
