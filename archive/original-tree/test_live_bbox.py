#!/usr/bin/env python3
"""Test live RIGOROSO:
  1. Move Ego pra posição conhecida (alto, yaw=0, sem gravidade caindo)
  2. Move Drone3 pra exatamente 15m forward, 0 right, mesma altura
  3. Atacha VisQuad_Drone3 4x
  4. Captura PC + AirSim detection + meu cálculo manual
  5. Gera PLY com PC + AirSim bbox (red) + manual bbox (blue) + ponto puro pose drone (yellow)
"""
import cosysairsim as airsim
from cosysairsim.types import ImageResponse, DetectionInfo
import numpy as np, cv2, time, math, json
from pathlib import Path
import sys
sys.path.insert(0, '/home/ericyos/airsim')
from drone_visual import DroneVisualizer

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
    cols = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))
    return pts, cols


def make_sphere(center, color, r=0.3, n=500):
    pts = []
    for _ in range(n):
        v = np.random.randn(3); v = v/(np.linalg.norm(v)+1e-9)*r
        pts.append(np.array(center) + v)
    pts = np.array(pts)
    return pts, np.tile(np.array(color, dtype=np.uint8), (n, 1))


def aabb_to_corners(center, size):
    cx, cy, cz = center; sx, sy, sz = max(size[0], 0.3), max(size[1], 0.3), max(size[2], 0.3)
    return np.array([
        [cx-sx/2, cy-sy/2, cz-sz/2], [cx+sx/2, cy-sy/2, cz-sz/2],
        [cx+sx/2, cy+sy/2, cz-sz/2], [cx-sx/2, cy+sy/2, cz-sz/2],
        [cx-sx/2, cy-sy/2, cz+sz/2], [cx+sx/2, cy-sy/2, cz+sz/2],
        [cx+sx/2, cy+sy/2, cz+sz/2], [cx-sx/2, cy+sy/2, cz+sz/2],
    ])


def main():
    c = airsim.MultirotorClient(ip='172.19.80.1', port=41451)
    c.confirmConnection()

    # Move drones longe
    for v in c.listVehicles():
        if v == 'Ego': continue
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(-5000,-5000,2000), yq(0)), True, v)
    time.sleep(0.3)

    # Ego em (0, 0, -30) yaw=0
    EGO_Z = -30.0
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, EGO_Z), yq(0)), True, 'Ego')
    time.sleep(0.5)

    # Drone3 em (15, 0, -30) — 15m forward, mesma altura
    DRONE_POS = np.array([15.0, 0.0, EGO_Z])
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*DRONE_POS), yq(0)), True, 'Drone3')
    time.sleep(0.5)

    # Re-set ego pra garantir
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, EGO_Z), yq(0)), True, 'Ego')
    time.sleep(0.3)

    # Attach VisQuad
    viz = DroneVisualizer(c, ['Drone3'], scale=4.0, sync_hz=30)
    viz.attach()
    viz.start_sync()
    time.sleep(1.5)

    # Re-set tudo ANTES da captura (gravidade pode ter movido)
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, EGO_Z), yq(0)), True, 'Ego')
    c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*DRONE_POS), yq(0)), True, 'Drone3')
    time.sleep(0.5)
    # Sync VisQuad pose
    obj_pose = c.simGetVehiclePose('Drone3')
    c.simSetObjectPose('VisQuad_Drone3', obj_pose, True)
    time.sleep(0.3)

    # Setup detection
    c.client.call('simClearDetectionMeshNames', 'front_center', airsim.ImageType.Scene, 'Ego', False)
    c.client.call('simSetDetectionFilterRadius', 'front_center', airsim.ImageType.Scene, 15000, 'Ego', False)
    c.client.call('simAddDetectionFilterMeshName', 'front_center', airsim.ImageType.Scene, 'Quadrotor1', 'Ego', False)
    time.sleep(0.3)

    # ATOMIC capture pausado
    c.simPause(True)
    try:
        ego_pose_real = c.simGetVehiclePose('Ego')
        drone_pose_real = c.simGetVehiclePose('Drone3')
        raw = c.client.call('simGetImages', [
            airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
            airsim.ImageRequest('front_center', airsim.ImageType.DepthPlanar, True, False),
        ], 'Ego', False)
        dets_raw = c.client.call('simGetDetections', 'front_center', airsim.ImageType.Scene, 'Ego', False)
    finally:
        c.simPause(False)

    print(f"Ego REAL pose: ({ego_pose_real.position.x_val:.2f}, {ego_pose_real.position.y_val:.2f}, {ego_pose_real.position.z_val:.2f})")
    eq = ego_pose_real.orientation
    print(f"  quat: w={eq.w_val:.3f} x={eq.x_val:.3f} y={eq.y_val:.3f} z={eq.z_val:.3f}")
    print(f"Drone3 REAL pose: ({drone_pose_real.position.x_val:.2f}, {drone_pose_real.position.y_val:.2f}, {drone_pose_real.position.z_val:.2f})")

    rgb = np.frombuffer(ImageResponse.from_msgpack(raw[0]).image_data_uint8, np.uint8).reshape(720,1280,3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_resp = ImageResponse.from_msgpack(raw[1])
    depth = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(depth_resp.height, depth_resp.width)
    if depth.shape != (720, 1280):
        depth = cv2.resize(depth, (1280, 720))
    pc = depth_to_pc(depth)
    print(f"PC: {len(pc)} pts  Z range [{pc[:,2].min():.2f}..{pc[:,2].max():.2f}]")

    cv2.imwrite('/tmp/live_test.png', bgr)

    dets = [DetectionInfo.from_msgpack(r) for r in dets_raw]
    dets = [d for d in dets if d.name.startswith('VisQuad_')]
    print(f"VisQuad detections: {len(dets)}")
    if not dets:
        print("ERRO: VisQuad_Drone3 não detectado!")
        viz.stop_sync(); viz.detach()
        return

    d = dets[0]
    rp = d.relative_pose.position
    print(f"\nAirSim rel_pose: ({rp.x_val:.2f}, {rp.y_val:.2f}, {rp.z_val:.2f})")
    if hasattr(d, 'box3D'):
        print(f"AirSim box3D_min: ({d.box3D.min.x_val:.2f}, {d.box3D.min.y_val:.2f}, {d.box3D.min.z_val:.2f})")
        print(f"AirSim box3D_max: ({d.box3D.max.x_val:.2f}, {d.box3D.max.y_val:.2f}, {d.box3D.max.z_val:.2f})")

    # CÁLCULO MANUAL: drone está em (15, 0, -30), ego em (0,0,-30)
    # rel_world = drone - ego = (15, 0, 0)
    # body NED (ego yaw=0): rel_body = (15, 0, 0) — direto forward
    # CV camera (pitch -15, offset 0.35, 0, -0.5):
    #   subtract offset: (15-0.35, 0, 0-(-0.5)) = (14.65, 0, 0.5)
    #   apply pitch +15: x_cam = c*14.65 + s*0.5 = 14.15+0.13=14.28
    #                    y_cam = 0
    #                    z_cam = -s*14.65 + c*0.5 = -3.79+0.48 = -3.31
    #   CV reorder (right, down, fwd): (0, -3.31, 14.28)
    ego_real = np.array([ego_pose_real.position.x_val, ego_pose_real.position.y_val, ego_pose_real.position.z_val])
    drone_real = np.array([drone_pose_real.position.x_val, drone_pose_real.position.y_val, drone_pose_real.position.z_val])
    rel_world = drone_real - ego_real
    # ego com yaw atual — vou usar quat completo
    q = eq
    # Hamilton quat → rot matrix (body→world)
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    R_ego = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    rel_body = R_ego.T @ rel_world
    # subtract camera offset (camera está em CAM_OFFSET_BODY no body)
    rel_cam_body = rel_body - CAM_OFFSET_BODY
    # apply pitch R_y(+15°) to rotate body → camera frame
    a = -CAM_PITCH_RAD
    rc, rs = np.cos(a), np.sin(a)
    x_cam_fwd = rc * rel_cam_body[0] + rs * rel_cam_body[2]
    y_cam_right = rel_cam_body[1]
    z_cam_down = -rs * rel_cam_body[0] + rc * rel_cam_body[2]
    # CV reorder (right, down, fwd)
    manual_cv = np.array([y_cam_right, z_cam_down, x_cam_fwd])
    print(f"\nMy manual cv: ({manual_cv[0]:.2f}, {manual_cv[1]:.2f}, {manual_cv[2]:.2f})")
    print(f"AirSim cv (apenas reorder FRD→CV): ({rp.y_val:.2f}, {rp.z_val:.2f}, {rp.x_val:.2f})")

    # Gera PLY
    out = Path('viz_live')
    out.mkdir(exist_ok=True)
    # PC colorido por depth
    if len(pc) > 80000:
        idx = np.random.choice(len(pc), 80000, replace=False)
        pc = pc[idx]
    z_min, z_max = pc[:,2].min(), pc[:,2].max()
    dn = (pc[:,2] - z_min) / (z_max - z_min + 1e-6)
    pc_cols = np.column_stack([150+80*(1-dn), 150+80*(1-dn), 180+60*(1-dn)]).clip(0,255).astype(np.uint8)

    all_pts = pc; all_cols = pc_cols
    # AirSim center: rel_pose direto (reorder FRD→CV)
    airsim_cv = np.array([rp.y_val, rp.z_val, rp.x_val])
    sp, sc = make_sphere(airsim_cv, (255, 0, 0))
    all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])
    # Bbox AirSim
    bmin = np.array([d.box3D.min.y_val, d.box3D.min.z_val, d.box3D.min.x_val])
    bmax = np.array([d.box3D.max.y_val, d.box3D.max.z_val, d.box3D.max.x_val])
    center_air = (bmin + bmax) / 2
    size_air = np.abs(bmax - bmin)
    corners_air = aabb_to_corners(center_air, size_air)
    wp, wc = make_wf(corners_air, (255, 60, 60))
    all_pts = np.vstack([all_pts, wp]); all_cols = np.vstack([all_cols, wc])
    # Bbox manual: usa pose drone real
    # 8 corners locais (Quadrotor1 4x: fwd=0.48, right=5.45, down=1.49)
    hx, hy, hz = 0.48/2, 5.45/2, 1.49/2
    local = np.array([
        [+hx,+hy,+hz],[-hx,+hy,+hz],[-hx,-hy,+hz],[+hx,-hy,+hz],
        [+hx,+hy,-hz],[-hx,+hy,-hz],[-hx,-hy,-hz],[+hx,-hy,-hz],
    ])
    # quat drone (yaw=0 since I set it)
    dq = drone_pose_real.orientation
    dw, dx, dy, dz = dq.w_val, dq.x_val, dq.y_val, dq.z_val
    R_drone = np.array([
        [1-2*(dy*dy+dz*dz), 2*(dx*dy-dw*dz), 2*(dx*dz+dw*dy)],
        [2*(dx*dy+dw*dz), 1-2*(dx*dx+dz*dz), 2*(dy*dz-dw*dx)],
        [2*(dx*dz-dw*dy), 2*(dy*dz+dw*dx), 1-2*(dx*dx+dy*dy)],
    ])
    corners_world = (R_drone @ local.T).T + drone_real
    # World → CV camera
    corners_cv = []
    for cw in corners_world:
        rb = R_ego.T @ (cw - ego_real)
        rcb = rb - CAM_OFFSET_BODY
        xc = rc * rcb[0] + rs * rcb[2]
        yc = rcb[1]
        zc = -rs * rcb[0] + rc * rcb[2]
        corners_cv.append([yc, zc, xc])
    corners_cv = np.array(corners_cv)
    wp, wc = make_wf(corners_cv, (0, 200, 255))
    all_pts = np.vstack([all_pts, wp]); all_cols = np.vstack([all_cols, wc])
    sp, sc = make_sphere(manual_cv, (0, 200, 255))
    all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])

    write_ply(out / 'live_test.ply', all_pts, all_cols)
    print(f"\nPLY saved: {out / 'live_test.ply'}")
    print("Legend: VERMELHO=AirSim  AZUL=Manual (pose+extent conhecido)")

    viz.stop_sync(); viz.detach()


if __name__ == "__main__":
    main()
