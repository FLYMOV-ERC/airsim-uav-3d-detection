#!/usr/bin/env python3
"""ARCHIVED. Deeper diagnosis of the drone/background colour collision in the segmentation.

Kept with the numbered segmentation chain it belongs to (Section 7.2.3, route (i)).
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys

print("\n" + "="*70)
print("DEBUG: IDENTIFYING THE DRONE COLOURS")
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

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\nTaking off...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Place the drones close and well separated, for clear identification
print("\nPlacing the drones for identification...")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()

# Place each drone in a different position, to identify its colour
if "Drone3" in vehicles:
    client.moveToPositionAsync(5, -3, -10, 3, vehicle_name="Drone3").join()
    print("   Drone3: 5 m ahead, 3 m left")

if "Drone4" in vehicles:
    client.moveToPositionAsync(5, 0, -10, 3, vehicle_name="Drone4").join()
    print("   Drone4: 5 m ahead, centred")

if "Intruder1" in vehicles:
    client.moveToPositionAsync(5, 3, -10, 3, vehicle_name="Intruder1").join()
    print("   Intruder1: 5 m ahead, 3 m right")

time.sleep(2)

# Capture the images
print("\nCapturing the images...")
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
], vehicle_name="Ego")

# Process the RGB image
img_bgr = None
if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Process the segmentation
seg_image = None
if responses[1].image_data_uint8:
    seg_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
    seg_image = seg_1d.reshape(responses[1].height, responses[1].width, 3)

if img_bgr is not None and seg_image is not None:
    print("\nAnalysing the colours in the segmentation...")

    # Find every unique colour
    unique_colors = np.unique(seg_image.reshape(-1, 3), axis=0)

    print(f"\nTotal unique colours: {len(unique_colors)}")
    print("\nCores encontradas (RGB):")

    # Analyse each colour
    color_info = []
    for i, color in enumerate(unique_colors):
        # Conta pixels desta cor
        mask = cv2.inRange(seg_image, color, color)
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (seg_image.shape[0] * seg_image.shape[1])) * 100

        color_info.append({
            'color': color,
            'count': pixel_count,
            'percentage': percentage
        })

        print(f"\n   Cor {i}: RGB{tuple(color)}")
        print(f"      Pixels: {pixel_count:,} ({percentage:.2f}%)")

        # Guess what each colour most likely is
        if np.array_equal(color, [0, 0, 0]):
            print("      -> likely background / empty")
        elif percentage > 30:
            print("      -> likely sky or terrain (too large)")
        elif percentage < 0.01:
            print("      -> likely noise (too small)")
        elif 0.05 < percentage < 5:
            print("      -> likely a DRONE (plausible size)")

    # Build the visualisation
    h, w = seg_image.shape[:2]
    vis = np.zeros((h*2, w*2, 3), dtype=np.uint8)

    # Put the RGB image in the top-left corner
    vis[:h, :w] = img_bgr

    # Put the segmentation in the top-right corner
    vis[:h, w:] = seg_image

    # Build the drone-colour visualisation (bottom corner)
    drone_colors = []
    y_offset = h + 20

    for info in sorted(color_info, key=lambda x: x['percentage']):
        color = info['color']
        pct = info['percentage']

        # Keep the candidate drone colours
        if 0.05 < pct < 5 and not np.array_equal(color, [0, 0, 0]):
            drone_colors.append(color)

            # Build a mask for this colour
            mask = cv2.inRange(seg_image, color, color)

            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Pega o maior contorno
                largest = max(contours, key=cv2.contourArea)
                x, y, w_bbox, h_bbox = cv2.boundingRect(largest)

                # Draw onto the visualisation
                color_bgr = tuple(int(c) for c in color[::-1])  # RGB to BGR
                cv2.rectangle(vis[:h, w:], (x, y), (x+w_bbox, y+h_bbox), color_bgr, 2)

                # Adiciona texto
                cv2.putText(vis, f"RGB{tuple(color)}", (w + x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

                # Mostra patch da cor
                cv2.rectangle(vis, (10, y_offset), (100, y_offset + 30), color_bgr, -1)
                cv2.putText(vis, f"Drone? {pct:.2f}%", (110, y_offset + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 40

    print(f"\nLikely drone colours: {len(drone_colors)}")
    for color in drone_colors:
        print(f"   RGB{tuple(color)}")

    # Save the visualisation
    cv2.imwrite("debug_segmentation.png", vis)
    print("\nVisualisation saved to debug_segmentation.png")

    # Mostra
    cv2.imshow("Debug Segmentation", vis)
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Pousa
print("\nLanding...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\nDebug finished")