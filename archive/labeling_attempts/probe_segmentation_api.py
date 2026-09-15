#!/usr/bin/env python3
"""ARCHIVED. Probe of ImageType.Segmentation and stencil-ID assignment.

The concrete test behind the failure of route (i) in Section 7.2.3.
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
print("DEEP INVESTIGATION: SEGMENTATION IN AIRSIM")
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

# Test 1: check whether an API exists for mesh IDs or object IDs
print("\nTEST 1: exploring the segmentation APIs...")
try:
    # Try to read information about the objects in the scene
    # AirSim uses mesh IDs for the segmentation
    print("   Trying to read object information...")

    # Arm and take off only a few drones, for the test
    for v in ["Ego", "Drone3"]:
        if v in vehicles:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
            client.takeoffAsync(vehicle_name=v).join()

    time.sleep(3)

    # Posiciona drones
    client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
    client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone3").join()

    time.sleep(2)

except Exception as e:
    print(f"   error: {e}")

# Test 2: capture the different image types
print("\nTEST 2: available image types...")
image_types = [
    (airsim.ImageType.Scene, "Scene (RGB)"),
    (airsim.ImageType.DepthPlanar, "DepthPlanar"),
    (airsim.ImageType.DepthPerspective, "DepthPerspective"),
    (airsim.ImageType.DepthVis, "DepthVis"),
    (airsim.ImageType.DisparityNormalized, "DisparityNormalized"),
    (airsim.ImageType.Segmentation, "Segmentation"),
    (airsim.ImageType.SurfaceNormals, "SurfaceNormals"),
    (airsim.ImageType.Infrared, "Infrared")
]

for img_type, name in image_types:
    try:
        response = client.simGetImages([
            airsim.ImageRequest("front_center", img_type, False, False)
        ], vehicle_name="Ego")[0]

        if response.image_data_uint8:
            print(f"   {name}: available")
        else:
            print(f"{name}: no data")
    except Exception as e:
        print(f"   {name}: error - {e}")

# Test 3: analyse the segmentation in detail
print("\nTEST 3: detailed analysis of the segmentation...")

# Capture the segmentation
response = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
], vehicle_name="Ego")[0]

if response.image_data_uint8:
    img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    seg_img = img_1d.reshape(response.height, response.width, 3)

    # Analyse the unique colours and their IDs
    unique_colors = np.unique(seg_img.reshape(-1, 3), axis=0)

    print(f"\n   total unique colours/IDs: {len(unique_colors)}")

    # AirSim maps object IDs to colours with a specific formula
    # ID = R + G*256 + B*256*256
    print("\n   Mapping of colours to object IDs:")

    for color in unique_colors:
        # Calcula o ID do objeto baseado na cor
        object_id = int(color[0]) + int(color[1])*256 + int(color[2])*256*256

        # Conta pixels
        mask = cv2.inRange(seg_img, color, color)
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (seg_img.shape[0] * seg_img.shape[1])) * 100

        print(f"\n   Cor RGB{tuple(color)}:")
        print(f"      Object ID: {object_id}")
        print(f"      Pixels: {pixel_count:,} ({percentage:.3f}%)")

        # Try to identify it from its characteristics
        if object_id == 0:
            print("      → Tipo: Background/Sky")
        elif percentage > 20:
            print("      → Tipo: Terreno/Grande objeto")
        elif 0.01 < percentage < 2.0:
            print("      -> type: possible drone / small object")

            # Check whether it is a dark object (drones usually are)
            if np.sum(color) < 100:
                print("      -> DRONE CANDIDATE (dark and small)")

# Test 4: try alternative methods
print("\nTEST 4: alternative detection methods...")

# Method 1: use the vehicle poses to validate the detections
print("\n   Method 1: validation against the drones' known positions")
for v in vehicles:
    if v != "Ego":
        try:
            pose = client.simGetVehiclePose(vehicle_name=v)
            print(f"      {v}: position ({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {pose.position.z_val:.1f})")
        except:
            pass

# Method 2: check whether an object-annotation API exists
print("\n   Method 2: looking for annotation APIs...")
try:
    # Try the various APIs that may exist
    if hasattr(client, 'simGetSegmentationObjectID'):
        print("      simGetSegmentationObjectID is available")
    else:
        print("      simGetSegmentationObjectID not found")

    if hasattr(client, 'simSetSegmentationObjectID'):
        print("      simSetSegmentationObjectID is available")
    else:
        print("      simSetSegmentationObjectID not found")

    if hasattr(client, 'simGetObjectPose'):
        print("      simGetObjectPose is available")
    else:
        print("      simGetObjectPose not found")

except Exception as e:
    print(f"      error: {e}")

# Test 5: check the mesh-name configuration
print("\nTEST 5: per-mesh segmentation configuration...")
print("   IMPORTANT: in Unreal Engine you must:")
print("   1. add a stencil ID to the drone meshes")
print("   2. set Custom Depth-Stencil in the mesh properties")
print("   3. assign a unique ID to each object type")
print("   4. AirSim converts those IDs into segmentation colours")

# Save the visualisation for analysis
print("\nSaving the visualisations for analysis...")

# RGB
rgb_response = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")[0]

if rgb_response.image_data_uint8:
    img_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    rgb_img = img_1d.reshape(rgb_response.height, rgb_response.width, 3)
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Build the comparison visualisation
    h, w = seg_img.shape[:2]
    vis = np.zeros((h, w*2, 3), dtype=np.uint8)
    vis[:, :w] = rgb_bgr
    vis[:, w:] = seg_img

    cv2.putText(vis, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Segmentation", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite("airsim_segmentation_analysis.png", vis)
    print("   Saved: airsim_segmentation_analysis.png")

# Pousa
print("\n Finalizando...")
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
print("""
1. SEGMENTATION IN AIRSIM:
   - uses object IDs mapped to RGB colours
   - ID = R + G*256 + B*256*256
   - every unique object gets a unique colour

2. CURRENT PROBLEM:
   - the drones may have no segmentation IDs configured
   - or they share an ID with the terrain

3. POSSIBLE SOLUTIONS:

   A) NO UNREAL ENGINE:
      - assign unique stencil IDs to the drones
      - Configurar Custom Depth-Stencil
      - make sure each drone has a different ID

   B) IN CODE:
      - use the drones' known positions (ground truth)
      - combine segmentation with depth for validation
      - use feature-based detection (dark colour + small size)

   C) THE SIMPLEST ALTERNATIVE:
      - use the drones' 3D bounding boxes (where the API provides them)
      - project to 2D using the camera parameters

4. RECOMMENDATION:
   - set the stencil IDs in Unreal for a definitive solution
   - or use depth plus the known pose as a workaround
""")