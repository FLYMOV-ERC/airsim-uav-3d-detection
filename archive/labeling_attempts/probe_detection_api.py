#!/usr/bin/env python3
"""ARCHIVED. Detailed per-target recall measurement of the simGetDetections API.

The source of the "~36% of in-FOV drones missed" figure that the dissertation
quotes as "roughly a third" in route (ii) of Section 7.2.3.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys

print("\n" + "="*70)
print(" TESTE DETALHADO: API simGetDetections")
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

# Test 1: check that the functions exist
print("\nTEST 1: checking which functions are available...")
detection_functions = [
    'simGetDetections',
    'simSetDetectionFilterRadius',
    'simAddDetectionFilterMeshName',
    'simClearDetectionMeshNames',
    'simGetMeshPositionVertexBuffers'
]

for func in detection_functions:
    if hasattr(client, func):
        print(f"   {func} is available")
    else:
        print(f"   {func} is NOT available")

# Prepara drones
print("\n Preparando drones...")
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("Taking off...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Place the drones for the test
print("\n Posicionando drones...")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
if "Drone3" in vehicles:
    client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone3").join()
    print("   Drone3 placed 10 m ahead")
if "Drone4" in vehicles:
    client.moveToPositionAsync(15, 3, -10, 3, vehicle_name="Drone4").join()
    print("   Drone4 placed 15 m ahead, 3 m to the right")
time.sleep(2)

# Test 2: different detection configurations
print("\nTEST 2: trying different configurations...")

# Candidate mesh names for the drones
mesh_patterns = [
    "*",  # every object
    "Drone*",  # Drone pattern
    "drone*",  # lowercase
    "SimpleFlight*",  # vehicle type
    "Multirotor*",  # generic type
    "Vehicle*",  # generic vehicle
    "Quadrotor*",  # Quadrotor
    "BP_FlyingPawn*",  # the default AirSim blueprint
    ".*",  # regex matching everything
    "Drone3",  # specific name
    "Intruder*",  # Intruder
]

# Camera settings
camera_configs = [
    ("front_center", airsim.ImageType.Scene),
    ("0", airsim.ImageType.Scene),
    ("front_center", airsim.ImageType.DepthPlanar),
]

best_config = None
max_detections = 0

for camera_name, image_type in camera_configs:
    print(f"\n   camera: {camera_name}, type: {image_type}")

    for pattern in mesh_patterns:
        try:
            # Clear the previous filters
            client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")

            # Set the detection radius (cm)
            radius_cm = 10000  # 100 metros
            client.simSetDetectionFilterRadius(camera_name, image_type, radius_cm, vehicle_name="Ego")

            # Add a mesh-name pattern
            client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

            # Tenta detectar
            detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

            if detections and len(detections) > 0:
                print(f"      pattern '{pattern}': {len(detections)} detections")

                for det in detections[:3]:  # show up to three detections
                    print(f"         - Nome: {det.name if hasattr(det, 'name') else 'N/A'}")
                    if hasattr(det, 'box2D'):
                        print(f"           Box2D: min({det.box2D.min.x:.0f}, {det.box2D.min.y:.0f}), max({det.box2D.max.x:.0f}, {det.box2D.max.y:.0f})")
                    if hasattr(det, 'relative_pose'):
                        print(f"           Relative position: ({det.relative_pose.position.x_val:.1f}, {det.relative_pose.position.y_val:.1f}, {det.relative_pose.position.z_val:.1f})")

                if len(detections) > max_detections:
                    max_detections = len(detections)
                    best_config = (camera_name, image_type, pattern)
            else:
                print(f"      pattern '{pattern}': 0 detections")

        except Exception as e:
            print(f"      error with pattern '{pattern}': {e}")

# Test 3: use the best configuration found
if best_config:
    print(f"\nTEST 3: using the best configuration found...")
    camera_name, image_type, pattern = best_config
    print(f"   best config: camera='{camera_name}', type={image_type}, pattern='{pattern}'")

    # Set up with the best configuration
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    # Capture the image and the detections
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name if camera_name != "0" else "front_center",
                           airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Get the detections
        detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

        # Desenha bounding boxes
        for det in detections:
            if hasattr(det, 'box2D'):
                x_min = int(det.box2D.min.x)
                y_min = int(det.box2D.min.y)
                x_max = int(det.box2D.max.x)
                y_max = int(det.box2D.max.y)

                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if hasattr(det, 'name'):
                    cv2.putText(img_bgr, det.name, (x_min, y_min-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite("test_detection_result.png", img_bgr)
        print(f"   Image saved: test_detection_result.png")
        print(f"   total detections: {len(detections)}")

else:
    print("\nNo configuration produced any detection")

# Test 4: try detection with no filters (detect EVERYTHING)
print("\nTEST 4: detection with no specific filters...")
try:
    # Clear all the filters
    client.simClearDetectionMeshNames("front_center", airsim.ImageType.Scene, vehicle_name="Ego")

    # Define raio grande
    client.simSetDetectionFilterRadius("front_center", airsim.ImageType.Scene, 50000, vehicle_name="Ego")

    # Add no specific filter - try to detect everything
    detections = client.simGetDetections("front_center", airsim.ImageType.Scene, vehicle_name="Ego")

    print(f"   detections with no filter: {len(detections) if detections else 0}")

except Exception as e:
    print(f"   error: {e}")

# Test 5: check whether an alternative API exists
print("\nTEST 5: looking for alternative APIs...")

# Candidate related methods
possible_methods = dir(client)
detection_related = [m for m in possible_methods if 'detect' in m.lower() or 'object' in m.lower() or 'mesh' in m.lower()]

if detection_related:
    print("   Related methods found:")
    for method in detection_related:
        print(f"      - {method}")
else:
    print("   No alternative method found")

# Pousa
print("\nLanding...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("CONCLUSIONS:")
print("="*70)

if max_detections > 0:
    print(f"DETECTION WORKED")
    print(f"   best configuration: {best_config}")
    print(f"   maximum number of detections: {max_detections}")
else:
    print("DETECTION FAILED")
    print("""
   Possible reasons:
   1. this AirSim build does not expose the complete API
   2. the drones have no detectable mesh names
   3. additional configuration is needed in Unreal Engine
   4. the API may be disabled in this build

   Solutions:
   1. build AirSim from the latest source
   2. use an alternative method (known positions)
   3. Configurar mesh names no Unreal Engine
   """)