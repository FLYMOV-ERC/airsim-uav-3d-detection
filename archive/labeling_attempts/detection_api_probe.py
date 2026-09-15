#!/usr/bin/env python3
"""ARCHIVED. Systematic probe of the simGetDetections API against the official documentation.

Tries different mesh names and configurations in an attempt to detect the drones.
This is the literal source of the dissertation's "more than twenty mesh-name
filter patterns" in route (ii) of Section 7.2.3.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys
import json

print("\n" + "="*70)
print("CORRECT IMPLEMENTATION OF THE simGetDetections API")
print("="*70)
print("Based on the official AirSim documentation")

# Conecta
try:
    client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"Connected. Vehicles available: {vehicles}")
except Exception as e:
    print(f"Connection error: {e}")
    sys.exit(1)

# Arm all the drones
print("\n Preparando drones...")
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        print(f"{v} preparado")
    except Exception as e:
        print(f"   error arming {v}: {e}")

# Take everything off
print("\nTaking off all the drones...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
        print(f"{v} decolou")
    except Exception as e:
        print(f"   error taking off {v}: {e}")

time.sleep(3)

# Place the drones in a visible formation
print("\nPlacing the drones in formation...")
positions = {
    "Ego": {"x": 0, "y": 0, "z": -10},
    "Drone3": {"x": 10, "y": -3, "z": -10},
    "Drone4": {"x": 10, "y": 0, "z": -10},
    "Intruder1": {"x": 10, "y": 3, "z": -10}
}

for vehicle, pos in positions.items():
    if vehicle in vehicles:
        try:
            client.moveToPositionAsync(pos["x"], pos["y"], pos["z"], 5, vehicle_name=vehicle).join()
            print(f"{vehicle} posicionado em ({pos['x']}, {pos['y']}, {pos['z']})")
        except:
            pass

time.sleep(2)

# IMPORTANT: per the documentation, the correct parameters are:
# - camera_name: either a name or an ID ("0", "1", ...)
# - image_type: the image type (Scene, DepthPlanar, ...)
# - vehicle_name: name of the vehicle carrying the camera

print("\n" + "="*70)
print("TESTING DETECTION UNDER DIFFERENT CONFIGURATIONS")
print("="*70)

# Mesh names to try, from the documentation and the examples
mesh_patterns_to_test = [
    # Generic patterns
    "*",                    # detects everything
    ".*",                   # regex matching everything

    # Patterns based on BP_FlyingPawn (the default AirSim blueprint)
    "BP_FlyingPawn",        # exact name, no wildcard
    "BP_FlyingPawn*",       # with a wildcard
    "BP_FlyingPawn.*",      # with a regex
    "*FlyingPawn*",         # Wildcard em ambos lados

    # Patterns based on SimpleFlight (the vehicle type)
    "SimpleFlight",
    "SimpleFlight*",
    "*SimpleFlight*",

    # Patterns based on the vehicle names
    "Drone*",               # Detecta Drone3, Drone4
    "Intruder*",            # Detecta Intruder1
    "*drone*",              # Case insensitive
    "*intruder*",

    # Component-based patterns
    "Pawn",
    "Pawn*",
    "*Pawn*",
    "Vehicle",
    "Vehicle*",

    # Unreal-specific patterns
    "SM_*",                 # Static Mesh prefix
    "BP_*",                 # Blueprint prefix
    "*_C",                  # Class instance suffix
]

# Camera settings to try
camera_configs = [
    ("0", airsim.ImageType.Scene),          # numeric ID
    ("front_center", airsim.ImageType.Scene),  # camera name
]

results = {}
best_config = None
max_detections = 0

for camera_name, image_type in camera_configs:
    print(f"\nTrying camera '{camera_name}', type {image_type}")

    for pattern in mesh_patterns_to_test:
        try:
            # STEP 1: clear the previous filters
            client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")

            # STEP 2: set the detection radius (in centimetres)
            radius_cm = 100 * 100  # 100 metros = 10000 cm
            client.simSetDetectionFilterRadius(camera_name, image_type, radius_cm, vehicle_name="Ego")

            # STEP 3: add the mesh-name pattern
            if pattern:  # if not empty, add the filter
                client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

            # Short pause, to be sure the filter took effect
            time.sleep(0.05)

            # STEP 4: get the detections
            detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

            num_detections = len(detections) if detections else 0

            if num_detections > 0:
                print(f"   pattern '{pattern}': {num_detections} detections")

                # Show the details of the detections
                for i, det in enumerate(detections[:3]):  # show up to three
                    print(f"      detection {i+1}:")

                    # Nome do objeto
                    if hasattr(det, 'name'):
                        print(f"         Nome: {det.name}")

                    # Bounding box 2D
                    if hasattr(det, 'box2D'):
                        if hasattr(det.box2D.min, 'x_val'):
                            print(f"         Box2D: ({det.box2D.min.x_val:.0f}, {det.box2D.min.y_val:.0f}) a ({det.box2D.max.x_val:.0f}, {det.box2D.max.y_val:.0f})")
                        else:
                            print(f"         Box2D: ({det.box2D.min.x:.0f}, {det.box2D.min.y:.0f}) a ({det.box2D.max.x:.0f}, {det.box2D.max.y:.0f})")

                    # Relative position
                    if hasattr(det, 'relative_pose'):
                        print(f"         Relative position: ({det.relative_pose.position.x_val:.1f}, {det.relative_pose.position.y_val:.1f}, {det.relative_pose.position.z_val:.1f})")

                # Update the best configuration
                if num_detections > max_detections:
                    max_detections = num_detections
                    best_config = (camera_name, image_type, pattern)

                # Save the result
                key = f"{camera_name}_{pattern}"
                results[key] = num_detections

        except Exception as e:
            print(f"   error with pattern '{pattern}': {str(e)[:100]}")

# If anything was detected, capture a screenshot with the visualisation
if best_config and max_detections > 0:
    print("\n" + "="*70)
    print("CAPTURING A VISUALISATION WITH THE BEST CONFIGURATION")
    print("="*70)

    camera_name, image_type, pattern = best_config
    print(f"Best configuration found:")
    print(f"   camera: '{camera_name}'")
    print(f"   Tipo: {image_type}")
    print(f"   pattern: '{pattern}'")
    print(f"   detections: {max_detections}")

    # Set up with the best configuration
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 100 * 100, vehicle_name="Ego")
    if pattern:
        client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    time.sleep(0.1)

    # Capture the image
    actual_camera = "front_center" if camera_name == "0" else camera_name
    response = client.simGetImages([
        airsim.ImageRequest(actual_camera, airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")[0]

    if response.image_data_uint8:
        # Convert to OpenCV format
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(response.height, response.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Get the detections again
        detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

        # Desenha bounding boxes
        for det in detections:
            if hasattr(det, 'box2D'):
                # Try the different ways of accessing the values
                try:
                    if hasattr(det.box2D.min, 'x_val'):
                        x_min = int(det.box2D.min.x_val)
                        y_min = int(det.box2D.min.y_val)
                        x_max = int(det.box2D.max.x_val)
                        y_max = int(det.box2D.max.y_val)
                    else:
                        x_min = int(det.box2D.min.x)
                        y_min = int(det.box2D.min.y)
                        x_max = int(det.box2D.max.x)
                        y_max = int(det.box2D.max.y)

                    # Draw the rectangle
                    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Adiciona label
                    if hasattr(det, 'name'):
                        cv2.putText(img_bgr, det.name, (x_min, y_min-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"   error drawing the box: {e}")

        # Save the image
        cv2.imwrite("detection_working.png", img_bgr)
        print("\nImage saved: detection_working.png")

else:
    print("\n" + "="*70)
    print("NO CONFIGURATION DETECTED ANY OBJECT")
    print("="*70)

# Save the results
with open("detection_results.json", "w") as f:
    json.dump({
        "results": results,
        "best_config": {
            "camera": best_config[0] if best_config else None,
            "image_type": str(best_config[1]) if best_config else None,
            "pattern": best_config[2] if best_config else None,
            "max_detections": max_detections
        }
    }, f, indent=2)

print("\nResults saved to detection_results.json")

# Land all the drones
print("\nLanding all the drones...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
        print(f"{v} pousou")
    except:
        pass

print("\n" + "="*70)
print("ANALYSIS OF THE RESULTS")
print("="*70)

if max_detections > 0:
    print("DETECTION WORKED")
    print(f"\nBest configuration:")
    print(f"   - camera: '{best_config[0]}'")
    print(f"   - image type: {best_config[1]}")
    print(f"   - mesh pattern: '{best_config[2]}'")
    print(f"   - number of detections: {max_detections}")
    print("\nUse this configuration in the dataset generator")
else:
    print("DETECTION FAILED UNDER EVERY CONFIGURATION")
    print("\nPossible solutions:")
    print("1. Configure mesh names no Unreal Engine:")
    print("   - add tags to the BP_FlyingPawn blueprints")
    print("   • Configure Stencil IDs")
    print("   - enable detection on the blueprints")
    print("\n2. use a more recent AirSim build:")
    print("   - build from source")
    print("   - check that the API is enabled")
    print("\n3. use the alternative method:")
    print("   - the drones' known positions (ground truth)")
    print("   - manual 3D-to-2D projection")