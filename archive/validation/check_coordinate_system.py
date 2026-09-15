#!/usr/bin/env python3
"""ARCHIVED. Verify the AirSim coordinate system and validate the sign of the Y axis.

The NED-world / CV-optical frame study behind Section 7.2.1's frame statement, and
behind the R-versus-R-transpose projection bug recorded in docs/methodology.md.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import time

print("\n" + "="*70)
print("CHECKING THE AIRSIM COORDINATE SYSTEM")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
client.confirmConnection()
vehicles = client.listVehicles()

print("\n Sistema de Coordenadas do AirSim:")
print("-" * 50)
print("AirSim uses NED (North-East-Down):")
print("   - X = North (forward)")
print("   - Y = East (right)")
print("   - Z = Down")
print("\nIn the camera:")
print("   - X = forward")
print("   - Y = right")
print("   - Z = down")

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass

time.sleep(3)

print("\n TESTE DE POSICIONAMENTO:")
print("-" * 50)

# Test 1: drone on the LEFT (negative world Y)
print("\nTest 1: Drone3 to the LEFT of the camera")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(10, -5, -10, 3, vehicle_name="Drone3").join()  # Y=-5 (left)
time.sleep(2)

ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
drone3_pose = client.simGetVehiclePose(vehicle_name="Drone3")

ego_pos = [ego_pose.position.x_val, ego_pose.position.y_val, ego_pose.position.z_val]
drone3_pos = [drone3_pose.position.x_val, drone3_pose.position.y_val, drone3_pose.position.z_val]

relative_pos = [drone3_pos[0] - ego_pos[0],
                drone3_pos[1] - ego_pos[1],
                drone3_pos[2] - ego_pos[2]]

print(f"   Ego position: X={ego_pos[0]:.1f}, Y={ego_pos[1]:.1f}, Z={ego_pos[2]:.1f}")
print(f"   Drone3 position: X={drone3_pos[0]:.1f}, Y={drone3_pos[1]:.1f}, Z={drone3_pos[2]:.1f}")
print(f"   relative position: X={relative_pos[0]:.1f}, Y={relative_pos[1]:.1f}, Z={relative_pos[2]:.1f}")

if relative_pos[1] < 0:
    print("   negative Y = drone on the LEFT (correct in NED)")
else:
    print("   positive Y = drone on the RIGHT (incorrect)")

# Test 2: drone on the RIGHT (positive world Y)
print("\nTest 2: Drone4 to the RIGHT of the camera")
client.moveToPositionAsync(10, 5, -10, 3, vehicle_name="Drone4").join()  # Y=+5 (right)
time.sleep(2)

drone4_pose = client.simGetVehiclePose(vehicle_name="Drone4")
drone4_pos = [drone4_pose.position.x_val, drone4_pose.position.y_val, drone4_pose.position.z_val]
relative_pos2 = [drone4_pos[0] - ego_pos[0],
                 drone4_pos[1] - ego_pos[1],
                 drone4_pos[2] - ego_pos[2]]

print(f"   Drone4 position: X={drone4_pos[0]:.1f}, Y={drone4_pos[1]:.1f}, Z={drone4_pos[2]:.1f}")
print(f"   relative position: X={relative_pos2[0]:.1f}, Y={relative_pos2[1]:.1f}, Z={relative_pos2[2]:.1f}")

if relative_pos2[1] > 0:
    print("   positive Y = drone on the RIGHT (correct in NED)")
else:
    print("   negative Y = drone on the LEFT (incorrect)")

# Test 3: check in the image
print("\nCHECK IN THE 2D IMAGE:")
print("-" * 50)

# Camera parameters
image_width = 1280
image_height = 720
FOV_H = 90
fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = fx
cx = image_width / 2
cy = image_height / 2

# Project Drone3 (on the left)
if relative_pos[0] > 0:  # if it is ahead
    u1 = fx * relative_pos[1] / relative_pos[0] + cx
    v1 = fy * (-relative_pos[2]) / relative_pos[0] + cy

    print(f"Drone3 (Y={relative_pos[1]:.1f}):")
    print(f"   Pixel X = {u1:.0f} (de 0 a {image_width})")
    if u1 < cx:
        print(f"   it is on the LEFT of the image (X < {cx}) -- correct")
    else:
        print(f"   it is on the RIGHT of the image (X > {cx}) -- incorrect")

# Project Drone4 (on the right)
if relative_pos2[0] > 0:  # if it is ahead
    u2 = fx * relative_pos2[1] / relative_pos2[0] + cx
    v2 = fy * (-relative_pos2[2]) / relative_pos2[0] + cy

    print(f"\nDrone4 (Y={relative_pos2[1]:.1f}):")
    print(f"   Pixel X = {u2:.0f} (de 0 a {image_width})")
    if u2 > cx:
        print(f"   it is on the RIGHT of the image (X > {cx}) -- correct")
    else:
        print(f"   it is on the LEFT of the image (X < {cx}) -- incorrect")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("""
No AirSim (sistema NED):
   - negative Y = LEFT
   - positive Y = RIGHT
   - negative Z = UP (altitude)
   - positive Z = DOWN

This is CORRECT and expected.

For training PointNet:
   The 3D coordinates are correct
   A negative Y for drones to the left is normal
   The model will learn this convention

To convert to another convention:
   - ROS uses ENU (East-North-Up)
   - Unity uses a left-handed system
   - but for training, use it as it is
""")

# Pousa
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass