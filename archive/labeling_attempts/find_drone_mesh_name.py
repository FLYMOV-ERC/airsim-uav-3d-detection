#!/usr/bin/env python3
"""ARCHIVED. Narrow down the exact mesh name of the drones.

Tests detection and inspects the name returned. Part of the route-(ii)
investigation of Section 7.2.3.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import time
import sys

print("\n" + "="*70)
print("FINDING THE EXACT MESH NAME OF THE DRONES")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"Connected. Vehicles: {vehicles}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Prepara e posiciona drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass

time.sleep(3)

# Place the drones well separated and visible
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(10, -5, -10, 3, vehicle_name="Drone3").join()
client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone4").join()
client.moveToPositionAsync(10, 5, -10, 3, vehicle_name="Intruder1").join()
time.sleep(2)

print("\nTEST 1: detect EVERYTHING and inspect the names")
print("-" * 50)

# Configure it to detect EVERYTHING
camera_name = "front_center"
image_type = airsim.ImageType.Scene

client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera_name, image_type, "*", vehicle_name="Ego")

time.sleep(0.1)

# Detecta
detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

if detections:
    print(f"Total detections: {len(detections)}")
    print("\nAnalysing each detection:")

    drone_names = []

    for i, det in enumerate(detections):
        print(f"\n   detection {i+1}:")

        # Nome
        name = det.name if hasattr(det, 'name') else "NO NAME"
        print(f"      Nome: '{name}'")

        # Box size
        if hasattr(det, 'box2D'):
            if hasattr(det.box2D.min, 'x_val'):
                width = det.box2D.max.x_val - det.box2D.min.x_val
                height = det.box2D.max.y_val - det.box2D.min.y_val
            else:
                width = det.box2D.max.x - det.box2D.min.x
                height = det.box2D.max.y - det.box2D.min.y

            print(f"      size: {width:.0f}x{height:.0f} px")

            # Try to tell whether it is a drone from its size
            if 10 < width < 400 and 10 < height < 400:
                print(f"      -> likely a DRONE (small or medium size)")
                drone_names.append(name)
            elif width > 1000 or height > 500:
                print(f"      -> likely GROUND or MOUNTAIN (too large)")
            else:
                print(f"      → Objeto desconhecido")

        # 3D position, where available
        if hasattr(det, 'relative_pose'):
            x = det.relative_pose.position.x_val
            y = det.relative_pose.position.y_val
            z = det.relative_pose.position.z_val
            print(f"      position: ({x:.1f}, {y:.1f}, {z:.1f})")

    if drone_names:
        print(f"\nLikely drone mesh names: {list(set(drone_names))}")

print("\nTEST 2: try specific patterns")
print("-" * 50)

# Very specific patterns to try
test_patterns = [
    # Based on the vehicle names
    "Drone3",
    "Drone4",
    "Intruder1",
    "Ego",

    # Combinations
    "Drone*",
    "Intruder*",
    "*3*",
    "*4*",

    # Candidate mesh / blueprint names
    "BP_FlyingPawn",
    "BP_FlyingPawn_C",
    "FlyingPawn",
    "SimpleFlight",
    "Multirotor",

    # Test whether the names are the vehicles themselves
    "Vehicle*",
    "*Vehicle*",
]

best_pattern = None
best_count = 0

for pattern in test_patterns:
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    time.sleep(0.05)

    detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")
    count = len(detections) if detections else 0

    if count > 0:
        print(f"   '{pattern}': {count} detections")

        # Check whether something small was detected (likely a drone)
        small_objects = 0
        for det in detections:
            if hasattr(det, 'box2D'):
                if hasattr(det.box2D.min, 'x_val'):
                    width = det.box2D.max.x_val - det.box2D.min.x_val
                    height = det.box2D.max.y_val - det.box2D.min.y_val
                else:
                    width = det.box2D.max.x - det.box2D.min.x
                    height = det.box2D.max.y - det.box2D.min.y

                if width < 400 and height < 400:
                    small_objects += 1

        if small_objects > 0:
            print(f"      -> {small_objects} small objects (likely drones)")

            if small_objects > best_count:
                best_count = small_objects
                best_pattern = pattern

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

if best_pattern and best_count > 0:
    print(f"BEST PATTERN FOUND: '{best_pattern}'")
    print(f"   detects {best_count} likely drones")
    print(f"\nUse this pattern in the dataset generator")
else:
    print("No pattern was found that detects ONLY drones")
    print("\n   Solutions:")
    print("   1. Configure mesh names no Unreal Engine")
    print("   2. detect EVERYTHING and filter by size")
    print("   3. use the drones' known positions")

# Pousa
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass