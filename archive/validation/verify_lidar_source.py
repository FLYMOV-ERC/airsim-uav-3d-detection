#!/usr/bin/env python3
"""ARCHIVED. Verify exactly where the working point cloud comes from.

The only archived .py file containing the literal LiDAR channel parameters, which
makes it the code-side record of the sensor configuration the dissertation names.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import json

print("\n" + "="*60)
print("LIDAR SENSOR CHECK")
print("="*60)

# Conecta
print("\nConnecting to AirSim...")
client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
client.confirmConnection()

vehicles = client.listVehicles()
print(f"Drones available: {vehicles}")

# Check the ego configuration
print("\nCHECKING THE SENSORS ON THE 'Ego' DRONE:")
print("-"*40)

# Tenta descobrir sensores LiDAR configurados
possible_lidar_names = [
    "LidarSensor1", "LidarSensor2",
    "LidarFront", "LidarBack", "LidarTop", "LidarBottom",
    "Lidar1", "Lidar2",
    "LiDAR", "LIDAR",
    "lidar", "front_lidar"
]

lidar_found = []

for lidar_name in possible_lidar_names:
    try:
        # Tenta pegar dados do LiDAR
        lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")

        if lidar_data and hasattr(lidar_data, 'point_cloud'):
            points = np.array(lidar_data.point_cloud, dtype=np.float32)

            if len(points) > 3:
                points = points.reshape(-1, 3)
                lidar_found.append({
                    'name': lidar_name,
                    'points': len(points),
                    'data': lidar_data
                })
                print(f"'{lidar_name}' FOUND - {len(points)} points")
    except:
        pass

if not lidar_found:
    print("NO LiDAR sensor found on the ego drone")
else:
    print(f"\n SENSORES LIDAR ENCONTRADOS: {len(lidar_found)}")

    for lidar_info in lidar_found:
        print(f"\n Sensor: '{lidar_info['name']}'")
        print(f"   points: {lidar_info['points']}")

        # Show the sensor information
        data = lidar_info['data']
        if hasattr(data, 'pose'):
            print(f"   position on the drone:")
            print(f"      X: {data.pose.position.x_val:.2f}")
            print(f"      Y: {data.pose.position.y_val:.2f}")
            print(f"      Z: {data.pose.position.z_val:.2f}")

        if hasattr(data, 'time_stamp'):
            print(f"   Timestamp: {data.time_stamp}")

# Check the configuration file
print("\nCHECKING THE CONFIGURATION (settings.json):")
print("-"*40)

# Read the local settings.json
try:
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    if 'Vehicles' in settings and 'Ego' in settings['Vehicles']:
        ego_config = settings['Vehicles']['Ego']

        if 'Sensors' in ego_config:
            print("Sensores configurados no Ego:")
            for sensor_name, sensor_config in ego_config['Sensors'].items():
                sensor_type = sensor_config.get('SensorType', 'Unknown')
                if sensor_type == 6:  # LiDAR
                    print(f"{sensor_name} (LiDAR)")
                    print(f"      Canais: {sensor_config.get('NumberOfChannels', 'N/A')}")
                    print(f"      points/s: {sensor_config.get('PointsPerSecond', 'N/A')}")
                    print(f"      FOV Vertical: {sensor_config.get('VerticalFOVLower', 'N/A')}° a {sensor_config.get('VerticalFOVUpper', 'N/A')}°")
                    print(f"      FOV Horizontal: {sensor_config.get('HorizontalFOVStart', 'N/A')}° a {sensor_config.get('HorizontalFOVEnd', 'N/A')}°")
                    print(f"      position: X={sensor_config.get('X', 0)}, Y={sensor_config.get('Y', 0)}, Z={sensor_config.get('Z', 0)}")
        else:
            print("No sensor is configured on the ego")
    else:
        print("Ego not found in the configuration")

except FileNotFoundError:
    print("settings.json not found locally")

# Testa captura real
print("\n TESTE DE CAPTURA REAL:")
print("-"*40)

if lidar_found:
    lidar_name = lidar_found[0]['name']
    print(f"Using sensor: '{lidar_name}'")

    # Prepara e decola Ego
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")
    client.takeoffAsync(vehicle_name="Ego").join()

    import time
    time.sleep(3)

    # Captura dados
    print("\nCapturando dados LiDAR...")
    lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")

    if lidar_data and hasattr(lidar_data, 'point_cloud'):
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3)

        print(f"\nPOINT CLOUD CAPTURED:")
        print(f"   Total points: {len(points)}")
        print(f"   Forma do array: {points.shape}")
        print(f"   X range: {points[:,0].min():.1f} m to {points[:,0].max():.1f} m")
        print(f"   Y range: {points[:,1].min():.1f} m to {points[:,1].max():.1f} m")
        print(f"   Z range: {points[:,2].min():.1f} m to {points[:,2].max():.1f} m")

        # Save a sample
        np.save('amostra_lidar.npy', points)
        print(f"\nSample saved to 'lidar_sample.npy'")

        # Show the first few points
        print(f"\nFirst five points (X, Y, Z):")
        for i in range(min(5, len(points))):
            print(f"   Ponto {i}: [{points[i,0]:.2f}, {points[i,1]:.2f}, {points[i,2]:.2f}]")

    # Pousa
    client.landAsync(vehicle_name="Ego").join()
    client.armDisarm(False, "Ego")
    client.enableApiControl(False, "Ego")

else:
    print("No LiDAR is configured on the drone")
    print("\nThe point cloud may be generated by:")
    print("   1. De um sensor LiDAR virtual do AirSim")
    print("   2. converted from the depth map")
    print("   3. from a sensor configured on Windows but not visible here")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)

if lidar_found:
    print(f"SIM! O drone Ego tem {len(lidar_found)} sensor(es) LiDAR acoplado(s)!")
    print(f"   Nome: '{lidar_found[0]['name']}'")
    print(f"   Producing {lidar_found[0]['points']} 3D points per capture")
    print("\nThe LiDAR IS physically simulated on the drone,")
    print("   capturing the surrounding 3D environment")
else:
    print("No LiDAR detected - check the configuration on the Windows side")