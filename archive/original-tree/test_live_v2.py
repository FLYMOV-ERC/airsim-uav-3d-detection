#!/usr/bin/env python3
"""Test live v2: usa GLOBAL poses + fix DroneVisualizer.
Gera PLY com PC + AirSim bbox + manual bbox.
"""
import cosysairsim as airsim
from cosysairsim.types import ImageResponse, DetectionInfo, Pose
import numpy as np, cv2, time, math, sys
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
from drone_visual import DroneVisualizer, vehicle_to_global

IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2
CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = np.radians(-15.0)


def yq(y):
    return airsim.Quaternionr(0, 0, math.sin(y/2), math.cos(y/2))


def depth_to_pc(depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > 0.1) & (depth < 250)
    z = depth[valid]; u_v = u[valid].astype(np.float32); v_v = v[valid].astype(np.float32)
    x = (u_v - CX) * z / FX
    y = (v_v - CY) * z / FY
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def write_ply(path, points, colors):
    n = len(points)
    with open(path, 'wb') as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode('ascii'))
        for p, c in zip(points, colors):
            f.write(np.array(p, dtype=np.float32).tobytes())
            f.write(np.array(c, dtype=np.uint8).tobytes())


def make_wf(corners, color, pts_per_edge=200):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0, 1, pts_per_edge)[:, None]
        pts.append(corners[a] * (1 - t) + corners[b] * t)
    pts = np.vstack(pts)
    return pts, np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))


def make_sphere(center, color, r=0.4, n=500):
    pts = np.array([np.array(center) + (lambda v: v / (np.linalg.norm(v)+1e-9) * r)(np.random.randn(3)) for _ in range(n)])
    return pts, np.tile(np.array(color, dtype=np.uint8), (n, 1))


def aabb_to_corners(center, size):
    cx, cy, cz = center
    sx, sy, sz = max(size[0], 0.3), max(size[1], 0.3), max(size[2], 0.3)
    return np.array([
        [cx-sx/2, cy-sy/2, cz-sz/2], [cx+sx/2, cy-sy/2, cz-sz/2],
        [cx+sx/2, cy+sy/2, cz-sz/2], [cx-sx/2, cy+sy/2, cz-sz/2],
        [cx-sx/2, cy-sy/2, cz+sz/2], [cx+sx/2, cy-sy/2, cz+sz/2],
        [cx+sx/2, cy+sy/2, cz+sz/2], [cx-sx/2, cy+sy/2, cz+sz/2],
    ])


def main():
    c = airsim.MultirotorClient(ip='172.19.80.1', port=41451)
    c.confirmConnection()

    # Limpa qualquer VisQuad anterior
    for v in c.listVehicles():
        if v == 'Ego': continue
        try: c.simDestroyObject(f'VisQuad_{v}')
        except: pass
        # Move targets longe
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(-5000,-5000,2000), yq(0)), True, v)
    time.sleep(0.3)

    # Sobe ego: pra global (0, 0, -25), ego origin (0,0,-5), vehicle = (0, 0, -20)
    c.enableApiControl(True, 'Ego'); c.armDisarm(True, 'Ego')
    try: c.takeoffAsync(vehicle_name='Ego').join()
    except: pass
    c.moveToPositionAsync(0, 0, -20, 3, vehicle_name='Ego').join()
    time.sleep(0.5)

    # Attach VisQuad (com FIX)
    viz = DroneVisualizer(c, ['Drone3'], scale=4.0, sync_hz=30)
    viz.attach()
    viz.start_sync()
    time.sleep(1)

    # Posiciona Drone3 (vehicle frame): origin Drone3 é (15,-5,-5)
    # Pra ter drone GLOBAL em (40, 0, -30), seto vehicle (40-15, 0-(-5), -30-(-5)) = (25, 5, -25)
    TARGET_GLOBAL = (40.0, 0.0, -30.0)
    ox, oy, oz = (15.0, -5.0, -5.0)
    c.simSetVehiclePose(airsim.Pose(
        airsim.Vector3r(TARGET_GLOBAL[0]-ox, TARGET_GLOBAL[1]-oy, TARGET_GLOBAL[2]-oz), yq(0)),
        True, 'Drone3')
    time.sleep(1.5)

    # Confirma poses globais
    def obj_pose(name):
        return Pose.from_msgpack(c.client.call('simGetObjectPose', name))

    ego_g = obj_pose('Ego')
    drone_g = obj_pose('Drone3')
    visq_g = obj_pose('VisQuad_Drone3')
    print(f"Ego global:    ({ego_g.position.x_val:.2f}, {ego_g.position.y_val:.2f}, {ego_g.position.z_val:.2f})")
    print(f"Drone3 global: ({drone_g.position.x_val:.2f}, {drone_g.position.y_val:.2f}, {drone_g.position.z_val:.2f})")
    print(f"VisQuad global:({visq_g.position.x_val:.2f}, {visq_g.position.y_val:.2f}, {visq_g.position.z_val:.2f})")

    # Capture
    c.client.call('simClearDetectionMeshNames', 'front_center', airsim.ImageType.Scene, 'Ego', False)
    c.client.call('simSetDetectionFilterRadius', 'front_center', airsim.ImageType.Scene, 15000, 'Ego', False)
    c.client.call('simAddDetectionFilterMeshName', 'front_center', airsim.ImageType.Scene, 'Quadrotor1', 'Ego', False)

    c.simPause(True)
    try:
        raw = c.client.call('simGetImages', [
            airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
            airsim.ImageRequest('front_center', airsim.ImageType.DepthPlanar, True, False),
        ], 'Ego', False)
        dets_raw = c.client.call('simGetDetections', 'front_center', airsim.ImageType.Scene, 'Ego', False)
    finally:
        c.simPause(False)

    rgb = np.frombuffer(ImageResponse.from_msgpack(raw[0]).image_data_uint8, np.uint8).reshape(720,1280,3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_resp = ImageResponse.from_msgpack(raw[1])
    depth = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(depth_resp.height, depth_resp.width)
    if depth.shape != (720, 1280):
        depth = cv2.resize(depth, (1280, 720))
    pc = depth_to_pc(depth)
    cv2.imwrite('/tmp/live_v2.png', bgr)

    dets = [DetectionInfo.from_msgpack(r) for r in dets_raw]
    dets = [d for d in dets if d.name.startswith('VisQuad_')]
    print(f"\nVisQuad detections: {len(dets)}")
    if not dets:
        viz.stop_sync(); viz.detach(); return
    d = dets[0]
    rp = d.relative_pose.position
    print(f"AirSim rel_pose: ({rp.x_val:.2f}, {rp.y_val:.2f}, {rp.z_val:.2f})")
    print(f"AirSim box3D_min:({d.box3D.min.x_val:.2f}, {d.box3D.min.y_val:.2f}, {d.box3D.min.z_val:.2f})")
    print(f"AirSim box3D_max:({d.box3D.max.x_val:.2f}, {d.box3D.max.y_val:.2f}, {d.box3D.max.z_val:.2f})")

    # CALCULO MANUAL com pose GLOBAL real
    ego_p = np.array([ego_g.position.x_val, ego_g.position.y_val, ego_g.position.z_val])
    drone_p = np.array([drone_g.position.x_val, drone_g.position.y_val, drone_g.position.z_val])
    rel_world = drone_p - ego_p
    # Ego yaw=0, R_ego = identidade
    rel_body = rel_world  # ego yaw=0 → no rotation
    rel_cam_body = rel_body - CAM_OFFSET_BODY
    a = -CAM_PITCH_RAD  # R_y(+15°)
    rc, rs = np.cos(a), np.sin(a)
    manual_fwd = rc * rel_cam_body[0] + rs * rel_cam_body[2]
    manual_right = rel_cam_body[1]
    manual_down = -rs * rel_cam_body[0] + rc * rel_cam_body[2]
    print(f"\nMANUAL cv (right, down, fwd): ({manual_right:.2f}, {manual_down:.2f}, {manual_fwd:.2f})")
    print(f"DIFF AirSim vs Manual:")
    airsim_cv = np.array([rp.y_val, rp.z_val, rp.x_val])  # FRD → CV reorder
    manual_cv = np.array([manual_right, manual_down, manual_fwd])
    print(f"  AirSim CV: {airsim_cv.round(2).tolist()}")
    print(f"  Manual CV: {manual_cv.round(2).tolist()}")
    print(f"  DIFF:      {(airsim_cv - manual_cv).round(2).tolist()}")

    # Generate PLY
    out = Path('viz_live')
    out.mkdir(exist_ok=True)
    if len(pc) > 80000:
        idx = np.random.choice(len(pc), 80000, replace=False)
        pc = pc[idx]
    z_min, z_max = pc[:,2].min(), pc[:,2].max()
    dn = (pc[:,2] - z_min) / (z_max - z_min + 1e-6)
    pc_cols = np.column_stack([150+80*(1-dn), 150+80*(1-dn), 180+60*(1-dn)]).clip(0,255).astype(np.uint8)

    all_pts = pc; all_cols = pc_cols

    # AirSim center (vermelho) + bbox (vermelho)
    sp, sc = make_sphere(airsim_cv, (255, 0, 0))
    all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])
    # AirSim bbox vem em frame body (FRD relative). Converter pra CV:
    air_min_cv = np.array([d.box3D.min.y_val, d.box3D.min.z_val, d.box3D.min.x_val])
    air_max_cv = np.array([d.box3D.max.y_val, d.box3D.max.z_val, d.box3D.max.x_val])
    air_center_cv = (air_min_cv + air_max_cv) / 2
    air_size_cv = np.abs(air_max_cv - air_min_cv)
    print(f"AirSim box CV: center={air_center_cv.round(2).tolist()} size={air_size_cv.round(2).tolist()}")
    wp_a, wc_a = make_wf(aabb_to_corners(air_center_cv, air_size_cv), (255, 0, 0))
    all_pts = np.vstack([all_pts, wp_a]); all_cols = np.vstack([all_cols, wc_a])

    # Manual center (azul) + bbox MANUAL com extents Quadrotor1 4x
    sp, sc = make_sphere(manual_cv, (0, 100, 255))
    all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])
    # Quadrotor1 4x: body NED extents medidos via detection box3D (3.01, 3.93, 2.79)
    # Margem 20% pra cobrir rotores spinning (excluidos do AABB AirSim).
    DRONE_HALF = np.array([3.01, 3.93, 2.79]) * 1.20 / 2
    local_corners = np.array([
        [+DRONE_HALF[0], +DRONE_HALF[1], +DRONE_HALF[2]],
        [-DRONE_HALF[0], +DRONE_HALF[1], +DRONE_HALF[2]],
        [-DRONE_HALF[0], -DRONE_HALF[1], +DRONE_HALF[2]],
        [+DRONE_HALF[0], -DRONE_HALF[1], +DRONE_HALF[2]],
        [+DRONE_HALF[0], +DRONE_HALF[1], -DRONE_HALF[2]],
        [-DRONE_HALF[0], +DRONE_HALF[1], -DRONE_HALF[2]],
        [-DRONE_HALF[0], -DRONE_HALF[1], -DRONE_HALF[2]],
        [+DRONE_HALF[0], -DRONE_HALF[1], -DRONE_HALF[2]],
    ])
    # Drone yaw=0 → corners world = drone_global + local
    drone_global_p = np.array([drone_g.position.x_val, drone_g.position.y_val, drone_g.position.z_val])
    corners_world = local_corners + drone_global_p
    # Para cada corner: world→ego body (yaw=0)→cv camera (pitch+offset)
    def world_to_cv(p_world):
        rel = p_world - ego_p
        rel_cam = rel - CAM_OFFSET_BODY
        ang = -CAM_PITCH_RAD
        rc, rs = np.cos(ang), np.sin(ang)
        fwd = rc * rel_cam[0] + rs * rel_cam[2]
        right = rel_cam[1]
        down = -rs * rel_cam[0] + rc * rel_cam[2]
        return np.array([right, down, fwd])
    corners_cv = np.array([world_to_cv(c) for c in corners_world])
    wp_m, wc_m = make_wf(corners_cv, (0, 100, 255))
    all_pts = np.vstack([all_pts, wp_m]); all_cols = np.vstack([all_cols, wc_m])

    write_ply(out / 'live_v2.ply', all_pts, all_cols)
    print(f"\nPLY: {out / 'live_v2.ply'}")
    viz.stop_sync(); viz.detach()


if __name__ == "__main__":
    main()
