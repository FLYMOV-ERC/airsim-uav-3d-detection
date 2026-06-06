#!/usr/bin/env python3
"""Gera vários PLYs com diferentes posições de drone + ego yaw, para validar bbox manual."""
import cosysairsim as airsim
from cosysairsim.types import ImageResponse, DetectionInfo, Pose
import numpy as np, cv2, time, math, sys
from pathlib import Path
sys.path.insert(0, '/home/ericyos/airsim')
from drone_visual import DroneVisualizer

IMAGE_W, IMAGE_H = 1280, 720
FX = IMAGE_W / (2 * np.tan(np.radians(90 / 2)))
FY = FX; CX, CY = IMAGE_W / 2, IMAGE_H / 2
CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = np.radians(-15.0)
DRONE_HALF = np.array([3.01, 3.93, 2.79]) * 1.38 / 2  # margem 38% (20% + 15% adicional)


def yq(y):
    return airsim.Quaternionr(0, 0, math.sin(y/2), math.cos(y/2))


def quat_to_yaw(q):
    return math.atan2(2*(q.w_val*q.z_val + q.x_val*q.y_val),
                      1 - 2*(q.y_val*q.y_val + q.z_val*q.z_val))


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def depth_to_pc(depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > 0.1) & (depth < 300)
    z = depth[valid]; u_v = u[valid].astype(np.float32); v_v = v[valid].astype(np.float32)
    return np.stack([(u_v - CX) * z / FX, (v_v - CY) * z / FY, z], axis=-1).astype(np.float32)


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


def make_wf(corners, color, n=200):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        t = np.linspace(0, 1, n)[:, None]
        pts.append(corners[a] * (1 - t) + corners[b] * t)
    pts = np.vstack(pts)
    return pts, np.tile(np.array(color, dtype=np.uint8), (len(pts), 1))


def make_sphere(c0, color, r=0.4, n=400):
    pts = np.array([c0 + (lambda v: v/(np.linalg.norm(v)+1e-9)*r)(np.random.randn(3)) for _ in range(n)])
    return pts, np.tile(np.array(color, dtype=np.uint8), (n, 1))


def world_to_cv(p_world, ego_pos, ego_yaw):
    rel_world = np.array(p_world) - np.array(ego_pos)
    R_ego = Rz(ego_yaw)
    rel_body = R_ego.T @ rel_world
    rel_cam = rel_body - CAM_OFFSET_BODY
    a = -CAM_PITCH_RAD
    rc, rs = math.cos(a), math.sin(a)
    return np.array([rel_cam[1], -rs*rel_cam[0]+rc*rel_cam[2], rc*rel_cam[0]+rs*rel_cam[2]])


def make_bbox_corners_world(drone_pos, drone_yaw):
    local = np.array([
        [+DRONE_HALF[0], +DRONE_HALF[1], +DRONE_HALF[2]], [-DRONE_HALF[0], +DRONE_HALF[1], +DRONE_HALF[2]],
        [-DRONE_HALF[0], -DRONE_HALF[1], +DRONE_HALF[2]], [+DRONE_HALF[0], -DRONE_HALF[1], +DRONE_HALF[2]],
        [+DRONE_HALF[0], +DRONE_HALF[1], -DRONE_HALF[2]], [-DRONE_HALF[0], +DRONE_HALF[1], -DRONE_HALF[2]],
        [-DRONE_HALF[0], -DRONE_HALF[1], -DRONE_HALF[2]], [+DRONE_HALF[0], -DRONE_HALF[1], -DRONE_HALF[2]],
    ])
    R = Rz(drone_yaw)
    return (R @ local.T).T + np.array(drone_pos)


def obj_pose(c, name):
    return Pose.from_msgpack(c.client.call('simGetObjectPose', name))


def main():
    c = airsim.MultirotorClient(ip='172.19.80.1', port=41451)
    c.confirmConnection()
    # Limpa
    for v in c.listVehicles():
        if v == 'Ego': continue
        try: c.simDestroyObject(f'VisQuad_{v}')
        except: pass
        c.simSetVehiclePose(airsim.Pose(airsim.Vector3r(-5000,-5000,2000), yq(0)), True, v)
    time.sleep(0.3)

    # Sobe ego
    c.enableApiControl(True, 'Ego'); c.armDisarm(True, 'Ego')
    try: c.takeoffAsync(vehicle_name='Ego').join()
    except: pass

    viz = DroneVisualizer(c, ['Drone3'], scale=4.0, sync_hz=30)
    viz.attach(); viz.start_sync()
    time.sleep(1)

    # 6 scenarios: variando ego_yaw, drone_yaw, distância, azimute
    # ego_global, drone_global, drone_yaw
    SCENARIOS = [
        # (ego_pos, ego_yaw_deg, drone_pos, drone_yaw_deg, label)
        ((0, 0, -25), 0,    (40, 0, -25),     0,    "s1_centered"),
        ((0, 0, -25), 0,    (30, 8, -22),    45,    "s2_right_high_yaw45"),
        ((0, 0, -25), 0,    (50, -12, -30), -30,    "s3_left_low_yaw-30"),
        ((0, 0, -25), 45,   (35, 35, -23),    0,    "s4_egoyaw45_drone_NE"),
        ((0, 0, -30), -30,  (25, -20, -28),   60,   "s5_egoyaw-30_droneSW"),
        ((0, 0, -20), 0,    (15, 0, -18),     90,   "s6_close_yaw90"),
    ]

    ORIGIN_DRONE3 = (15.0, -5.0, -5.0)
    ORIGIN_EGO = (0.0, 0.0, -5.0)
    out = Path('viz_live_multi')
    out.mkdir(exist_ok=True)

    for ego_g, ego_yaw_deg, drone_g, drone_yaw_deg, label in SCENARIOS:
        print(f"\n=== {label} ===")
        ego_yaw = math.radians(ego_yaw_deg)
        drone_yaw = math.radians(drone_yaw_deg)

        # Pose ego (vehicle frame)
        c.moveToPositionAsync(ego_g[0]-ORIGIN_EGO[0], ego_g[1]-ORIGIN_EGO[1], ego_g[2]-ORIGIN_EGO[2],
                              5, vehicle_name='Ego').join()
        time.sleep(0.3)
        c.simSetVehiclePose(airsim.Pose(
            airsim.Vector3r(ego_g[0]-ORIGIN_EGO[0], ego_g[1]-ORIGIN_EGO[1], ego_g[2]-ORIGIN_EGO[2]),
            yq(ego_yaw)), True, 'Ego')
        time.sleep(0.3)

        # Pose drone (vehicle frame)
        c.simSetVehiclePose(airsim.Pose(
            airsim.Vector3r(drone_g[0]-ORIGIN_DRONE3[0], drone_g[1]-ORIGIN_DRONE3[1], drone_g[2]-ORIGIN_DRONE3[2]),
            yq(drone_yaw)), True, 'Drone3')
        time.sleep(1.2)

        # Confirma globais
        ego_p_obj = obj_pose(c, 'Ego')
        drone_p_obj = obj_pose(c, 'Drone3')
        visq_p_obj = obj_pose(c, 'VisQuad_Drone3')
        ego_p = np.array([ego_p_obj.position.x_val, ego_p_obj.position.y_val, ego_p_obj.position.z_val])
        drone_p = np.array([drone_p_obj.position.x_val, drone_p_obj.position.y_val, drone_p_obj.position.z_val])
        visq_p = np.array([visq_p_obj.position.x_val, visq_p_obj.position.y_val, visq_p_obj.position.z_val])
        actual_ego_yaw = quat_to_yaw(ego_p_obj.orientation)
        actual_drone_yaw = quat_to_yaw(visq_p_obj.orientation)
        print(f"  Ego G:    ({ego_p[0]:.1f}, {ego_p[1]:.1f}, {ego_p[2]:.1f})  yaw={math.degrees(actual_ego_yaw):.1f}")
        print(f"  Drone G:  ({drone_p[0]:.1f}, {drone_p[1]:.1f}, {drone_p[2]:.1f})")
        print(f"  VisQuad G:({visq_p[0]:.1f}, {visq_p[1]:.1f}, {visq_p[2]:.1f})  yaw={math.degrees(actual_drone_yaw):.1f}")

        # Capture
        c.simPause(True)
        try:
            raw = c.client.call('simGetImages', [
                airsim.ImageRequest('front_center', airsim.ImageType.Scene, False, False),
                airsim.ImageRequest('front_center', airsim.ImageType.DepthPlanar, True, False),
            ], 'Ego', False)
        finally:
            c.simPause(False)

        rgb = np.frombuffer(ImageResponse.from_msgpack(raw[0]).image_data_uint8, np.uint8).reshape(720,1280,3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dr = ImageResponse.from_msgpack(raw[1])
        depth = np.array(dr.image_data_float, dtype=np.float32).reshape(dr.height, dr.width)
        if depth.shape != (720, 1280): depth = cv2.resize(depth, (1280, 720))
        pc = depth_to_pc(depth)
        cv2.imwrite(f'/tmp/{label}.png', bgr)

        # Bbox manual usando ACTUAL ego/drone yaw (lido após sim)
        corners_world = make_bbox_corners_world(visq_p, actual_drone_yaw)
        corners_cv = np.array([world_to_cv(cn, visq_p*0 + ego_p, actual_ego_yaw) for cn in corners_world])
        center_cv = world_to_cv(visq_p, ego_p, actual_ego_yaw)

        # Build PLY
        if len(pc) > 80000:
            idx = np.random.choice(len(pc), 80000, replace=False); pc = pc[idx]
        z_min, z_max = pc[:,2].min(), pc[:,2].max()
        dn = (pc[:,2]-z_min)/(z_max-z_min+1e-6)
        pc_cols = np.column_stack([150+80*(1-dn), 150+80*(1-dn), 180+60*(1-dn)]).clip(0,255).astype(np.uint8)

        all_pts = pc; all_cols = pc_cols
        sp, sc = make_sphere(center_cv, (0, 200, 255), r=0.4)
        all_pts = np.vstack([all_pts, sp]); all_cols = np.vstack([all_cols, sc])
        wp, wc = make_wf(corners_cv, (0, 100, 255))
        all_pts = np.vstack([all_pts, wp]); all_cols = np.vstack([all_cols, wc])

        write_ply(out / f'{label}.ply', all_pts, all_cols)
        print(f"  center_cv: {center_cv.round(2).tolist()}")
        print(f"  PLY: {label}.ply")

    viz.stop_sync(); viz.detach()


if __name__ == "__main__":
    main()
