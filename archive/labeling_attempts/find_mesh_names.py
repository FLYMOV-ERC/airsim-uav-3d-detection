#!/usr/bin/env python3
"""ARCHIVED. Enumerate the mesh names of every object in the scene.

Part of the route-(ii) investigation of Section 7.2.3: understanding how the
detection API could be made to see the drones.
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
print("FINDING THE MESH NAMES OF THE OBJECTS")
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

# Test 1: list the scene objects
print("\nTEST 1: listing the scene objects...")
try:
    # Tenta listar objetos da cena
    if hasattr(client, 'simListSceneObjects'):
        scene_objects = client.simListSceneObjects()
        print(f"   Total de objetos na cena: {len(scene_objects) if scene_objects else 0}")
        if scene_objects:
            print("\n   Primeiros 20 objetos:")
            for obj in scene_objects[:20]:
                print(f"      - {obj}")
            if len(scene_objects) > 20:
                print(f"      ... and {len(scene_objects) - 20} more objects")

            # Look for objects that might be drones
            drone_candidates = []
            keywords = ['drone', 'Drone', 'vehicle', 'Vehicle', 'multi', 'Multi',
                       'quad', 'Quad', 'rotor', 'Rotor', 'intruder', 'Intruder',
                       'flying', 'Flying', 'pawn', 'Pawn']

            for obj in scene_objects:
                for keyword in keywords:
                    if keyword in obj:
                        drone_candidates.append(obj)
                        break

            if drone_candidates:
                print(f"\n   candidate drones found ({len(drone_candidates)}):")
                for candidate in drone_candidates:
                    print(f"      - {candidate}")
    else:
        print("   simListSceneObjects is not available")

except Exception as e:
    print(f"   error: {e}")

# Test 2: try the object tags
print("\nTEST 2: listing the object tags...")
try:
    if hasattr(client, 'simListSceneObjectsTags'):
        tags = client.simListSceneObjectsTags()
        if tags:
            print(f"   Tags encontradas: {tags}")
    else:
        print("   simListSceneObjectsTags is not available")
except Exception as e:
    print(f"   error: {e}")

# Test 3: try simGetMeshPositionVertexBuffers
print("\nTEST 3: trying simGetMeshPositionVertexBuffers...")
test_names = vehicles + ["Cylinder", "Ground", "Cube", "SimpleFlight", "Multirotor", "Quadrotor"]
for name in test_names:
    try:
        result = client.simGetMeshPositionVertexBuffers(name)
        if result:
            print(f"Mesh '{name}' encontrado!")
            # Se encontrou, vamos testar se consegue detectar
            break
    except Exception as e:
        # Expected error for names that do not exist
        pass

# Test 4: prepare the drones and test detection with a wildcard
print("\nTEST 4: preparing the drones for a detection test...")

# Prepara e posiciona drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

print("   Taking off...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Place the drones clearly visible
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
if "Drone3" in vehicles:
    client.moveToPositionAsync(5, 0, -10, 3, vehicle_name="Drone3").join()
if "Drone4" in vehicles:
    client.moveToPositionAsync(7, 2, -10, 3, vehicle_name="Drone4").join()
if "Intruder1" in vehicles:
    client.moveToPositionAsync(10, -2, -10, 3, vehicle_name="Intruder1").join()

time.sleep(2)

# Test 5: detection with specific configurations
print("\nTEST 5: testing detection under different configurations...")

camera_name = "front_center"
image_type = airsim.ImageType.Scene

# Patterns to try (more specific)
test_patterns = [
    "",  # empty - detects everything
    ".*",  # regex matching everything
    "*",  # wildcard matching everything
    "BP_FlyingPawn",  # the default AirSim blueprint (no wildcard)
    "BP_FlyingPawn*",  # blueprint with a wildcard
    "BP_FlyingPawn.*",  # blueprint with a regex
    "SimpleFlight",  # vehicle type
    "SimpleFlight*",
    "Pawn",
    "Pawn*",
    "Vehicle",
    "Vehicle*"
]

results = {}

for pattern in test_patterns:
    try:
        # Clear the filters
        client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")

        # Define raio grande
        client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")

        # If the pattern is not empty, add the filter
        if pattern:
            client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

        # Short pause, to be sure the filter took effect
        time.sleep(0.1)

        # Tenta detectar
        detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

        num_detections = len(detections) if detections else 0
        results[pattern] = num_detections

        if num_detections > 0:
            print(f"   pattern '{pattern if pattern else '(empty)'}': {num_detections} detections")
            # Show the details of the first detection
            det = detections[0]
            if hasattr(det, 'name'):
                print(f"      first object: {det.name}")
        else:
            print(f"   pattern '{pattern if pattern else '(empty)'}': 0 detections")

    except Exception as e:
        print(f"   error with pattern '{pattern}': {e}")
        results[pattern] = -1

# Save the results
with open("mesh_detection_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to mesh_detection_results.json")

# If anything was detected, take a screenshot
best_pattern = max(results, key=lambda k: results[k] if results[k] >= 0 else -1)
if results[best_pattern] > 0:
    print(f"\nCapturing a screenshot with pattern '{best_pattern}'...")

    # Set up with the best pattern
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    if best_pattern:
        client.simAddDetectionFilterMeshName(camera_name, image_type, best_pattern, vehicle_name="Ego")

    # Capture the image
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
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
                x_min = int(det.box2D.min.x_val)
                y_min = int(det.box2D.min.y_val)
                x_max = int(det.box2D.max.x_val)
                y_max = int(det.box2D.max.y_val)

                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if hasattr(det, 'name'):
                    cv2.putText(img_bgr, det.name, (x_min, y_min-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite("mesh_detection_success.png", img_bgr)
        print("   Image saved: mesh_detection_success.png")

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
print("RESUMO:")
print("="*70)

# Show a summary of the results
successful_patterns = {k: v for k, v in results.items() if v > 0}
if successful_patterns:
    print("PATTERNS THAT WORKED:")
    for pattern, count in sorted(successful_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"   '{pattern if pattern else '(empty)'}': {count} detections")
else:
    print("NO PATTERN DETECTED ANY OBJECT")
    print("\n   This suggests that:")
    print("   1. the drones have no mesh names configured for detection")
    print("   2. or they need special configuration in Unreal Engine")
    print("   3. or the API needs different initialisation")