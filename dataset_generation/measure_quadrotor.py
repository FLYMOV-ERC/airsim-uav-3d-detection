#!/usr/bin/env python3
"""Measure the true extents of the Quadrotor1 mesh at scale 4x, via simGetMeshPositionVertexBuffers.

This is the source of the 3.01 x 3.93 x 2.79 m figure quoted in Section 7.2.2.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import time

c = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
c.confirmConnection()

# Limpa
try: c.simDestroyObject('TEST_QUAD')
except: pass
time.sleep(0.3)

# Spawn at (0,0,-50), high enough not to hit the ground
pose = airsim.Pose(airsim.Vector3r(0, 0, -50), airsim.Quaternionr(0, 0, 0, 1))
scale_v = airsim.Vector3r(4.0, 4.0, 4.0)
actual = c.simSpawnObject('TEST_QUAD', 'Quadrotor1', pose, scale_v,
                          physics_enabled=False, is_blueprint=False)
print(f"Spawned: {actual}")
time.sleep(1.0)

# Read the object pose
from cosysairsim.types import Pose
p = Pose.from_msgpack(c.client.call('simGetObjectPose', actual))
print(f"Pose obj: ({p.position.x_val:.2f}, {p.position.y_val:.2f}, {p.position.z_val:.2f})")

# METHOD 1: simGetMeshPositionVertexBuffers -- read the real vertices
try:
    meshes = c.simGetMeshPositionVertexBuffers()
    print(f"Total meshes: {len(meshes)}")
    matching = [m for m in meshes if 'TEST_QUAD' in m.name or 'Quadrotor' in m.name]
    print(f"Matching meshes: {[m.name for m in matching]}")
    # Gather every vertex of every matching mesh (the drone has several sub-meshes: body, rotors)
    all_verts = []
    for found in matching:
        verts = np.array(found.vertices, dtype=np.float32).reshape(-1, 3)
        all_verts.append(verts)
        print(f"  {found.name}: n_verts={len(verts)}")
    if all_verts:
        verts = np.vstack(all_verts)
    # Unreal coordinates are in cm. Convert to metres (/100) and to NED (X=fwd, Y=right, Z=-up)
        verts_ned = np.column_stack([verts[:, 0], verts[:, 1], -verts[:, 2]]) / 100.0
        # Centra relativo a pose drone (0,0,-50)
        verts_ned_rel = verts_ned - np.array([0, 0, -50])
        mn = verts_ned_rel.min(axis=0)
        mx = verts_ned_rel.max(axis=0)
        size = mx - mn
        print(f"  AABB (relative to drone center) m:")
        print(f"    min: {mn.round(3).tolist()}")
        print(f"    max: {mx.round(3).tolist()}")
        print(f"    SIZE (fwd, right, down): {size.round(3).tolist()}")
except Exception as e:
    print(f"vertex method: {e}")

# METHOD 2: use the detection box3D of TEST_QUAD itself, seen from the ego
print("\n--- Method 2: detection box3D ---")
c.client.call('simClearDetectionMeshNames', 'front_center', airsim.ImageType.Scene, 'Ego', False)
c.client.call('simSetDetectionFilterRadius', 'front_center', airsim.ImageType.Scene, 15000, 'Ego', False)
c.client.call('simAddDetectionFilterMeshName', 'front_center', airsim.ImageType.Scene, 'TEST_QUAD', 'Ego', False)
# Place the ego near test_quad with a clear line of sight
c.enableApiControl(True, 'Ego'); c.armDisarm(True, 'Ego')
try: c.takeoffAsync(vehicle_name='Ego').join()
except: pass
c.moveToPositionAsync(-15, 0, -45, 3, vehicle_name='Ego').join()
time.sleep(1)

from cosysairsim.types import DetectionInfo
dets_raw = c.client.call('simGetDetections', 'front_center', airsim.ImageType.Scene, 'Ego', False)
dets = [DetectionInfo.from_msgpack(r) for r in dets_raw]
print(f"Detections: {len(dets)}")
for d in dets:
    bm = d.box3D.min; bx = d.box3D.max
    size = np.array([bx.x_val - bm.x_val, bx.y_val - bm.y_val, bx.z_val - bm.z_val])
    print(f"  {d.name}: size body (fwd, right, down) = {np.abs(size).round(2).tolist()}")

# Limpa
c.simDestroyObject('TEST_QUAD')
print("\nDone.")
