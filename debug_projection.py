#!/usr/bin/env python3
"""
Diagnostic: find why the projected 3D point doesn't match the 2D bbox center.
Compares: relative_pose projection vs box2D center vs box3D center.
"""
import cosysairsim as airsim
import numpy as np
import time

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

# Setup
for v in client.listVehicles():
    client.enableApiControl(True, v)
    client.armDisarm(True, v)
    client.takeoffAsync(vehicle_name=v).join()
time.sleep(2)

# Ego at origin, drones in front at same height
print("\nPositioning drones...")
client.moveToPositionAsync(0, 0, -12, 3, vehicle_name="Ego").join()
print("  Ego at (0, 0, -12)")
client.moveToPositionAsync(20, 0, -12, 3, vehicle_name="Drone3").join()
print("  Drone3 at (20, 0, -12)")
client.moveToPositionAsync(20, 5, -12, 3, vehicle_name="Drone4").join()
print("  Drone4 at (20, 5, -12)")

# Reset yaw to face +X
client.rotateToYawAsync(0, vehicle_name="Ego").join()
print("  Ego yaw reset to 0")

time.sleep(3)

# Verify actual positions
for v in ["Ego", "Drone3", "Drone4"]:
    p = client.simGetVehiclePose(vehicle_name=v)
    print(f"  {v} actual: ({p.position.x_val:.1f}, {p.position.y_val:.1f}, {p.position.z_val:.1f})")

# Detection setup
camera = "front_center"
img_type = airsim.ImageType.Scene
client.simClearDetectionMeshNames(camera, img_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera, img_type, 15000, vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera, img_type, "Drone3", vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera, img_type, "Drone4", vehicle_name="Ego")
print("\nDetection filter configured. Waiting...")
time.sleep(1)

# Camera intrinsics
IMAGE_W, IMAGE_H = 1280, 720
FOV_H = 90
fx = IMAGE_W / (2 * np.tan(np.radians(FOV_H / 2)))
fy = fx
cx, cy = IMAGE_W / 2, IMAGE_H / 2
print(f"Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# Get detections
dets = client.simGetDetections(camera, img_type, vehicle_name="Ego")
print(f"\nDetections: {len(dets)}")

for det in dets:
    print(f"\n{'='*60}")
    print(f"Name: {det.name}")

    # Print all attributes to find box3D
    attrs = [a for a in dir(det) if not a.startswith('_')]
    print(f"Attributes: {attrs}")

    # --- relative_pose ---
    rp = det.relative_pose.position
    rel = np.array([rp.x_val, rp.y_val, rp.z_val])
    print(f"\nrelative_pose.position: X={rel[0]:.3f}, Y={rel[1]:.3f}, Z={rel[2]:.3f}")

    # Project relative_pose to 2D
    if rel[0] > 0:
        u_rel = fx * rel[1] / rel[0] + cx
        v_rel = fy * rel[2] / rel[0] + cy
        print(f"relative_pose projected: u={u_rel:.1f}, v={v_rel:.1f}")

    # --- box2D ---
    b2 = det.box2D
    b2_min = (b2.min.x_val, b2.min.y_val)
    b2_max = (b2.max.x_val, b2.max.y_val)
    b2_cx = (b2_min[0] + b2_max[0]) / 2
    b2_cy = (b2_min[1] + b2_max[1]) / 2
    print(f"\nbox2D: min={b2_min}, max={b2_max}")
    print(f"box2D center: u={b2_cx:.1f}, v={b2_cy:.1f}")

    # --- box3D (try different field names) ---
    for field in ['box3D', 'box3d', 'bounding_box_3d', 'bbox3D']:
        if hasattr(det, field):
            b3 = getattr(det, field)
            print(f"\n{field} found! Type: {type(b3)}")
            print(f"  Attributes: {[a for a in dir(b3) if not a.startswith('_')]}")
            if hasattr(b3, 'min') and hasattr(b3, 'max'):
                b3_center = np.array([
                    (b3.min.x_val + b3.max.x_val) / 2,
                    (b3.min.y_val + b3.max.y_val) / 2,
                    (b3.min.z_val + b3.max.z_val) / 2,
                ])
                print(f"  center: {b3_center}")
                if b3_center[0] > 0:
                    u_b3 = fx * b3_center[1] / b3_center[0] + cx
                    v_b3 = fy * b3_center[2] / b3_center[0] + cy
                    print(f"  projected: u={u_b3:.1f}, v={v_b3:.1f}")

    # --- OFFSET ANALYSIS ---
    print(f"\n--- OFFSET ---")
    print(f"box2D center:       u={b2_cx:.1f}, v={b2_cy:.1f}")
    if rel[0] > 0:
        print(f"relative_pose proj: u={u_rel:.1f}, v={v_rel:.1f}")
        print(f"  OFFSET:           du={u_rel - b2_cx:.1f}, dv={v_rel - b2_cy:.1f} pixels")

    # --- Also check simGetObjectPose for comparison ---
    try:
        obj_pose = client.simGetObjectPose(det.name)
        ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
        ego_pos = np.array([ego_pose.position.x_val, ego_pose.position.y_val, ego_pose.position.z_val])
        obj_pos = np.array([obj_pose.position.x_val, obj_pose.position.y_val, obj_pose.position.z_val])
        print(f"\nsimGetObjectPose: {obj_pos}")
        print(f"simGetVehiclePose(Ego): {ego_pos}")
        print(f"Difference (world): {obj_pos - ego_pos}")
    except Exception as e:
        print(f"simGetObjectPose error: {e}")

# Land
for v in client.listVehicles():
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass
