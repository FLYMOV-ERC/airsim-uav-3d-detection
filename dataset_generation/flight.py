#!/usr/bin/env python3
"""Live flight driver: guarantees the targets stay airborne, plus the YOLO -> frustum PointNet detector.

- Takes everything off and raises the ego to a safe altitude.
- The targets orbit slowly IN FRONT of the ego, continuously re-commanded from a thread
  so they never fall (this is what fixes targets dropping into the Coastline terrain).
- VisQuad 4x (via the corrected DroneVisualizer) so YOLO sees the mesh it was trained on.
- Loop: capture RGB + depth, YOLO at 1280, frustum PointNet per box, overlay and video.
"""
import sys, time, math, threading, argparse
import numpy as np, cv2
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import airsim_host, airsim_port
import cosysairsim as airsim
from cosysairsim.types import ImageResponse
from common.drone_visual import DroneVisualizer, VEHICLE_ORIGINS
from detection.frustum.detector import FrustumPN2Detector

IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2))); FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2


def depth_to_pc_cv(depth, max_d=250.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    m = (depth > 0.1) & (depth < max_d)
    z = depth[m]; uu = u[m].astype(np.float32); vv = v[m].astype(np.float32)
    return np.stack([(uu-CX)*z/FX, (vv-CY)*z/FY, z], -1).astype(np.float32)


def cv_to_frd(p):
    return np.array([p[2], p[0], p[1]], dtype=np.float32)


def _yaw_quat(yaw):
    return airsim.Quaternionr(0, 0, math.sin(yaw/2), math.cos(yaw/2))


class FlightThread(threading.Thread):
    """Kinematic flight: teleports each target along a slow orbit and HOLDS the ego
    in the air (vehicle frame) at ~15 Hz. No physics, no takeoff -- nothing falls or snags on terrain."""
    def __init__(self, ip, port, drones, centers, ego_z, stop_event, radius=6.0, w=0.25,
                 ego_move=True):
        super().__init__(daemon=True)
        self.ip, self.port = ip, port
        self.drones = drones; self.centers = centers; self.ego_z = ego_z
        self.stop_event = stop_event; self.radius = radius; self.w = w
        self.ego_move = ego_move

    def run(self):
        c = airsim.MultirotorClient(ip=self.ip, port=self.port); c.confirmConnection()
        t0 = time.time()
        while not self.stop_event.is_set():
            t = (time.time() - t0) * self.w  # rad lento
            # The EGO MOVES (moving platform): a slow 3D oscillation plus a small yaw sweep,
            # small amplitude, to keep the targets inside the FOV; exercises the tracker's global transform
            we = (time.time() - t0) * 0.20
            if self.ego_move:
                ex = 4.0*math.sin(we*0.5); ey = 5.0*math.sin(we); ez = self.ego_z + 3.0*math.sin(we*0.7)
                eyaw = math.radians(8.0)*math.sin(we*0.3)
            else:
                ex=ey=0.0; ez=self.ego_z; eyaw=0.0
            try:
                c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(ex, ey, ez), _yaw_quat(eyaw)),
                                    True, vehicle_name="Ego")
            except Exception:
                pass
            for i, d in enumerate(self.drones):
                cx, cy, cz = self.centers[d]   # vehicle frame
                x = cx + self.radius * math.cos(t + i*2.1)
                y = cy + self.radius * math.sin(t + i*2.1)
                yaw = t + i*2.1 + math.pi/2  # nose along the direction of travel
                try:
                    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, cz), _yaw_quat(yaw)),
                                        True, vehicle_name=d)
                except Exception:
                    pass
            self.stop_event.wait(1/15.0)  # ~15Hz → movimento suave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=airsim_host()); ap.add_argument("--port", type=int, default=airsim_port())
    ap.add_argument("--duration", type=float, default=50.0)
    ap.add_argument("--yolo", default="runs/drone/drone_synth_v1/weights/best.pt")
    ap.add_argument("--pointnet", default="runs/pointnet2_frustum/v4/best.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--pn_thr", type=float, default=0.45)
    ap.add_argument("--fuse_w", type=float, default=0.5,
                    help="weight of YOLO in the fusion: score = w*yolo + (1-w)*pn. Accepted if score >= fuse_thr.")
    ap.add_argument("--fuse_thr", type=float, default=0.5,
                    help="threshold on the fused YOLO x PointNet score")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO input size (smaller = higher fps)")
    ap.add_argument("--out", default="inference_output_flying")
    args = ap.parse_args()

    from pathlib import Path
    from ultralytics import YOLO
    Path(args.out).mkdir(exist_ok=True)
    print("[Models] YOLO + Frustum PN2")
    yolo = YOLO(args.yolo)
    det = FrustumPN2Detector(checkpoint_path=args.pointnet, cls_threshold=args.pn_thr)

    c = airsim.MultirotorClient(ip=args.ip, port=args.port); c.confirmConnection()
    vehicles = c.listVehicles()
    drones = [v for v in vehicles if v != "Ego"]
    print(f"[AirSim] vehicles={vehicles}")

    # Ego: teleport to a safe altitude (vehicle z = -20 -> global -25). No takeoff.
    EGO_Z = -20.0
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, EGO_Z), _yaw_quat(0)), True, "Ego")
    time.sleep(0.5)

    # Orbit centres IN FRONT of the ego (global). The camera is pitched -15 deg (looking down),
    # so the targets have to sit BELOW the ego (~dist*tan15) to be centred in the frame.
    ego_g = np.array([0.0, 0.0, EGO_Z + VEHICLE_ORIGINS['Ego'][2]])  # global ego ~ (0,0,-25)
    # A less negative global z means lower. At 30 m, 15 deg down is about 8 m below the ego.
    # SPREAD OUT (left/centre/right, different ranges) so the boxes do not overlap.
    targets_global = {
        'Drone3':    (ego_g[0]+22, ego_g[1]-14, ego_g[2]+5),   # near, left
        'Drone4':    (ego_g[0]+45, ego_g[1]+16, ego_g[2]+12),  # far, right
        'Intruder1': (ego_g[0]+33, ego_g[1]+1,  ego_g[2]+8),   # mid, centre
    }
    centers = {}
    for d in drones:
        gx, gy, gz = targets_global.get(d, (ego_g[0]+30, ego_g[1], ego_g[2]))
        ox, oy, oz = VEHICLE_ORIGINS.get(d, (0,0,0))
        centers[d] = (gx-ox, gy-oy, gz-oz)  # vehicle frame
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*centers[d]), _yaw_quat(0)), True, d)
    time.sleep(0.5)

    # VisQuad 4x
    viz = DroneVisualizer(c, drones, scale=4.0, sync_hz=30)
    viz.attach(); viz.start_sync(); time.sleep(1)

    # Continuous flight thread
    stop = threading.Event()
    fly = FlightThread(args.ip, args.port, drones, centers, EGO_Z, stop, radius=6.0, w=0.25)
    fly.start()
    print(f"[Fly] targets orbiting in front of the ego. Detecting for {args.duration}s...")

    frames_buf = []  # buffer, so the video is written at the REAL fps (otherwise it plays fast)
    cap_client = airsim.MultirotorClient(ip=args.ip, port=args.port); cap_client.confirmConnection()
    t_start = time.time(); n_frames = 0; n_yolo_tot = 0; n_det_tot = 0
    dets_summary = []
    try:
        while time.time() - t_start < args.duration:
            raw = cap_client.client.call('simGetImages', [
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
            pc = depth_to_pc_cv(depth)

            res = yolo.predict(bgr, conf=args.yolo_conf, imgsz=args.imgsz, verbose=False)
            vis = bgr.copy(); n_yolo = 0; n_det = 0
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxy = res[0].boxes.xyxy.cpu().numpy(); confs = res[0].boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    n_yolo += 1
                    bb = tuple(map(float, xyxy[i])); conf = float(confs[i])
                    x1,y1,x2,y2 = [int(v) for v in bb]
                    pred = det.predict(pc, depth, bb, conf)
                    pn = pred['is_drone_prob'] if pred else 0.0
                    # YOLO x PN fusion: a confident YOLO compensates for a timid PointNet
                    fused = args.fuse_w * conf + (1 - args.fuse_w) * pn
                    accept = (pred is not None) and (fused >= args.fuse_thr)
                    if accept:
                        n_det += 1
                        d3 = float(np.linalg.norm(cv_to_frd(pred['center_cv'])))
                        col = (0,255,0)
                        cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
                        cv2.putText(vis,f"DRONE {d3:.0f}m F{fused:.2f}(Y{conf:.2f} P{pn:.2f})",(x1,max(0,y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
                        dets_summary.append((d3, conf, pn, fused))
                    else:
                        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,140,255),1)
                        cv2.putText(vis,f"F{fused:.2f}(Y{conf:.2f} P{pn:.2f})",(x1,max(0,y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,140,255),1)
            cv2.putText(vis,f"t={time.time()-t_start:.0f}s yolo={n_yolo} det3d={n_det}",(10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            frames_buf.append(vis); n_frames += 1; n_yolo_tot += n_yolo; n_det_tot += n_det
            if n_frames % 5 == 0:
                print(f"  f{n_frames} yolo={n_yolo} det3d={n_det}")
    finally:
        elapsed = max(time.time() - t_start, 1e-3)
        stop.set(); fly.join(timeout=3)
        # Write at the REAL fps -> a real-time video, not a sped-up one
        real_fps = max(1.0, n_frames / elapsed)
        writer = cv2.VideoWriter(str(Path(args.out)/"flying.mp4"),
                                 cv2.VideoWriter_fourcc(*'mp4v'), real_fps, (IMAGE_W, IMAGE_H))
        for f in frames_buf:
            writer.write(f)
        writer.release()
        print(f"[Video] {n_frames} frames @ {real_fps:.1f} fps (real time)")
        viz.stop_sync(); viz.detach()

    print(f"\n[Done] {n_frames} frames | YOLO total={n_yolo_tot} | det3D total={n_det_tot}")
    if dets_summary:
        ds = np.array(dets_summary)
        print(f"  dist 3D: media={ds[:,0].mean():.1f}m range=[{ds[:,0].min():.0f},{ds[:,0].max():.0f}]")
        print(f"  yolo_conf: media={ds[:,1].mean():.2f}  pn_prob: media={ds[:,2].mean():.2f}  fused: media={ds[:,3].mean():.2f}")
    print(f"  video: {Path(args.out)/'flying.mp4'}")


if __name__ == "__main__":
    main()
