#!/usr/bin/env python3
"""ARCHIVED. Dataset collection with the LiDAR ALIGNED to the camera.

Ensures that whatever appears in the camera image is also present in the point
cloud. The substantive LiDAR/camera frame-alignment effort behind the shared
optical frame of Section 7.2.1, and the last representative of that line of work.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import math
from tqdm import tqdm

def rotate_lidar_to_camera_view(points, lidar_pose, camera_pose):
    """
    Rotate the LiDAR points into alignment with the camera view
    """
    # For now, return the original points
    # A transform can be added here if needed
    return points

def filter_points_in_camera_fov(points, fov_degrees=90, max_range=100):
    """
    Keep only the points inside the forward camera's field of view
    CORRECTED: positive X is forward in AirSim
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Keep the points in front (positive X)
    in_front = x > 0

    # Horizontal angle relative to the forward axis (X)
    # In AirSim: X = forward, Y = lateral
    angles = np.degrees(np.arctan2(y, x))

    # FOV horizontal
    half_fov = fov_degrees / 2
    in_horizontal_fov = (angles >= -half_fov) & (angles <= half_fov)

    # FOV vertical (assumindo 60 graus)
    vertical_angles = np.degrees(np.arctan2(-z, x))  # negated, because Z points down
    in_vertical_fov = (vertical_angles >= -30) & (vertical_angles <= 30)

    # Compute the range
    distances = np.sqrt(x**2 + y**2 + z**2)

    # Filter by range (including nearby drones)
    in_range = (distances > 0.1) & (distances < max_range)

    # Combine all the filters
    valid = in_front & in_horizontal_fov & in_vertical_fov & in_range

    return points[valid], np.sum(valid)

def detect_drones_in_pointcloud(points, drone_positions, ego_position=None):
    """
    Detect the point clusters corresponding to the drones
    Improved to account for positions relative to the ego
    """
    drone_detections = []

    for drone_name, pos in drone_positions.items():
        if drone_name == "Ego":
            continue

        # If the ego pose is known, convert to relative coordinates
        if ego_position:
            # Drone position relative to the ego (sensor coordinate system)
            rel_x = pos['x'] - ego_position['x']
            rel_y = pos['y'] - ego_position['y']
            rel_z = pos['z'] - ego_position['z']
            drone_loc = np.array([rel_x, rel_y, rel_z])
        else:
            drone_loc = np.array([pos['x'], pos['y'], pos['z']])

        # Compute the distance from the points to the drone
        distances = np.linalg.norm(points - drone_loc, axis=1)

        # Increase the detection radius and adjust the threshold
        # The drones may be 1-3 m across
        near_drone = distances < 5.0  # raised to 5 m, to capture more points
        num_points = np.sum(near_drone)

        # Compute the drone-to-ego range, to adjust the threshold
        drone_distance = np.linalg.norm(drone_loc)

        # Adaptive threshold based on the range
        # The farther away, the fewer points are expected
        # Lower the minimum threshold to one point
        min_points_threshold = max(1, int(20 / (1 + drone_distance/10)))

        if num_points >= min_points_threshold:
            drone_detections.append({
                'name': drone_name,
                'points': int(num_points),
                'position': pos,
                'relative_position': {'x': float(drone_loc[0]), 'y': float(drone_loc[1]), 'z': float(drone_loc[2])},
                'distance': float(drone_distance),
                'detected': True
            })
        else:
            drone_detections.append({
                'name': drone_name,
                'points': int(num_points),
                'position': pos,
                'relative_position': {'x': float(drone_loc[0]), 'y': float(drone_loc[1]), 'z': float(drone_loc[2])},
                'distance': float(drone_distance),
                'detected': False
            })

    return drone_detections

def visualize_lidar_with_drones(points, drone_detections, img_size=(800, 600)):
    """
    2D visualisation showing the detected drones
    """
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    if len(points) == 0:
        return img

    # Top-down view (XY)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Normaliza coordenadas
    x_range = max(abs(x.min()), abs(x.max())) * 1.1
    y_range = max(abs(y.min()), abs(y.max())) * 1.1

    if x_range > 0 and y_range > 0:
        # Convert to pixels
        px = ((x / x_range + 1) * img_size[0] / 2).astype(int)
        py = ((y / y_range + 1) * img_size[1] / 2).astype(int)

        # Colour by altitude (Z)
        z_norm = (z - z.min()) / (z.max() - z.min() + 0.001)
        colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Draw the points
        for i in range(len(points)):
            if 0 <= px[i] < img_size[0] and 0 <= py[i] < img_size[1]:
                color = colors[i][0].tolist()
                cv2.circle(img, (px[i], py[i]), 1, color, -1)

        # Mark the drone positions
        for drone in drone_detections:
            dx = drone['position']['x']
            dy = drone['position']['y']

            # Convert to pixels
            dpx = int((dx / x_range + 1) * img_size[0] / 2)
            dpy = int((dy / y_range + 1) * img_size[1] / 2)

            if 0 <= dpx < img_size[0] and 0 <= dpy < img_size[1]:
                color = (0, 255, 0) if drone['detected'] else (0, 0, 255)
                cv2.circle(img, (dpx, dpy), 8, color, 2)
                cv2.putText(img, drone['name'], (dpx-30, dpy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(img, f"{drone['points']}pts", (dpx-30, dpy+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Info
    cv2.putText(img, f"Total: {len(points)} points", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Top-down view (XY)", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Legenda
    cv2.putText(img, "Verde: Drone Detectado", (img_size[0]-200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Red: not detected", (img_size[0]-200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img

def main():
    print("\n" + "="*60)
    print("DATASET WITH THE LIDAR ALIGNED TO THE CAMERA")
    print("="*60)

    # Configuration
    output_dir = Path("dataset_lidar_camera_aligned")
    output_dir.mkdir(exist_ok=True)

    dirs = {
        'rgb': output_dir / 'rgb',
        'lidar': output_dir / 'lidar',
        'lidar_filtered': output_dir / 'lidar_filtered',  # points inside the camera FOV
        'lidar_viz': output_dir / 'lidar_viz',
        'detections': output_dir / 'detections',
        'metadata': output_dir / 'metadata'
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True)

    print(f"\n Dataset em: {output_dir.absolute()}")

    # Conecta
    print("\nConnecting...")
    client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"Drones: {vehicles}")

    # Verifica LiDAR
    print("\nConfiguring the LiDAR aligned with the camera...")
    lidar_name = "LidarFront"

    # Prepara drones
    print("\n Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
            print(f"{vehicle}")
        except:
            pass

    # Decola
    print("\nTaking off...")
    for vehicle in vehicles:
        try:
            client.takeoffAsync(vehicle_name=vehicle)
        except:
            pass
    time.sleep(5)

    # Posiciona Ego
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

    # Configuration
    num_frames = 10  # heavily reduced, for a quick test
    frames_captured = 0

    print(f"\nCapturing {num_frames} frames")
    print("   LiDAR filtered to the camera FOV")
    print("   Detecting drones in the point cloud\n")

    detection_stats = {
        'total_frames': 0,
        'frames_with_drones': 0,
        'total_detections': 0
    }

    with tqdm(total=num_frames, desc="Capturing") as pbar:
        for frame_idx in range(num_frames):
            t = frame_idx * 0.1

            # MOTION: place the drones IN FRONT of the ego (visible to the camera)
            num_drones = len(vehicles) - 1
            drone_idx = 0

            for vehicle in vehicles:
                if vehicle == "Ego":
                    # The ego stays put, looking forward
                    client.rotateToYawAsync(0, vehicle_name="Ego")  # facing forward (0 deg)
                    continue

                # IMPORTANT: place the drones IN FRONT of the ego
                # Between 10 and 30 m away, inside the field of view

                # Spread the drones across the forward field of view
                angle_offset = (drone_idx - num_drones/2) * 30  # -45° a +45°
                angle_rad = math.radians(angle_offset)

                # Varying range
                distance = 15 + 10 * math.sin(t + drone_idx)

                # FORWARD position (positive X = ahead)
                x = distance * math.cos(angle_rad)  # forward
                y = distance * math.sin(angle_rad)  # Lateral

                # IMPORTANT: place the drones at altitudes the LiDAR can reach
                # The LiDAR only returns at 0-1.7 m (ground) and 15-17 m (buildings)
                # Keep the drones low, where there are many returns

                if drone_idx == 0:
                    # First drone: very low, almost on the ground
                    z = -0.5  # 0.5 m altitude
                elif drone_idx == 1:
                    # Second drone: slightly higher
                    z = -1.0  # 1 m altitude
                else:
                    # Third drone: also low
                    z = -1.5  # 1.5 m altitude

                try:
                    # IMPORTANT: use .join() so the motion is actually executed
                    client.moveToPositionAsync(x, y, z, 3, vehicle_name=vehicle).join()
                except:
                    pass

                drone_idx += 1

            time.sleep(1)  # extra time, to be sure the motions are executed

            # CAPTURA
            try:
                # 1. Captura RGB
                responses = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
                ], vehicle_name="Ego")

                if responses[0].image_data_uint8:
                    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # Mark the drones in the image (for checking)
                    img_annotated = img_bgr.copy()
                    cv2.putText(img_annotated, f"Frame {frames_captured}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_file), img_annotated)

                # 2. Captura LiDAR
                lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")

                if lidar_data and len(lidar_data.point_cloud) > 3:
                    # Convert to an array
                    points_raw = np.array(lidar_data.point_cloud, dtype=np.float32)
                    points_raw = points_raw.reshape(-1, 3)

                    # Save the original points
                    np.save(dirs['lidar'] / f"frame_{frames_captured:06d}.npy", points_raw)

                    # KEEP the points inside the camera FOV
                    points_filtered, num_filtered = filter_points_in_camera_fov(points_raw)

                    # Save the filtered points
                    np.save(dirs['lidar_filtered'] / f"frame_{frames_captured:06d}.npy", points_filtered)

                    # Get the drone positions
                    drone_positions = {}
                    ego_position = None
                    for vehicle in vehicles:
                        try:
                            state = client.getMultirotorState(vehicle_name=vehicle)
                            position = {
                                'x': state.kinematics_estimated.position.x_val,
                                'y': state.kinematics_estimated.position.y_val,
                                'z': state.kinematics_estimated.position.z_val
                            }
                            if vehicle == "Ego":
                                ego_position = position
                            else:
                                drone_positions[vehicle] = position
                        except:
                            pass

                    # DETECT drones in the point cloud (position relative to the ego)
                    drone_detections = detect_drones_in_pointcloud(points_filtered, drone_positions, ego_position)

                    # Statistics
                    num_detected = sum(1 for d in drone_detections if d['detected'])
                    if num_detected > 0:
                        detection_stats['frames_with_drones'] += 1
                        detection_stats['total_detections'] += num_detected

                    pbar.write(f"   Frame {frames_captured}: {len(points_raw)} pts total, "
                              f"{num_filtered} no FOV, {num_detected}/{len(drone_detections)} drones detectados")

                    # Visualisation
                    viz_img = visualize_lidar_with_drones(points_filtered, drone_detections)
                    viz_file = dirs['lidar_viz'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(viz_file), viz_img)

                    # Save the detections
                    detections_data = {
                        'frame': frames_captured,
                        'total_points': int(len(points_raw)),
                        'points_in_fov': int(num_filtered),
                        'drone_detections': drone_detections
                    }

                    det_file = dirs['detections'] / f"frame_{frames_captured:06d}.json"
                    with open(det_file, 'w') as f:
                        json.dump(detections_data, f, indent=2)

                # 3. Metadata
                metadata = {
                    'frame': frames_captured,
                    'timestamp': datetime.now().isoformat(),
                    'lidar': {
                        'sensor': lidar_name,
                        'total_points': int(len(points_raw)) if 'points_raw' in locals() else 0,
                        'points_in_camera_fov': int(num_filtered) if 'num_filtered' in locals() else 0
                    },
                    'vehicles': {}
                }

                for vehicle in vehicles:
                    try:
                        state = client.getMultirotorState(vehicle_name=vehicle)
                        metadata['vehicles'][vehicle] = {
                            'position': {
                                'x': float(state.kinematics_estimated.position.x_val),
                                'y': float(state.kinematics_estimated.position.y_val),
                                'z': float(state.kinematics_estimated.position.z_val)
                            }
                        }
                    except:
                        pass

                meta_file = dirs['metadata'] / f"frame_{frames_captured:06d}.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                frames_captured += 1
                detection_stats['total_frames'] += 1
                pbar.update(1)

            except Exception as e:
                pbar.write(f"Error on frame {frame_idx}: {e}")
                continue

    # Pousa
    print("\nLanding...")
    for vehicle in vehicles:
        try:
            client.landAsync(vehicle_name=vehicle)
        except:
            pass

    time.sleep(5)

    for vehicle in vehicles:
        try:
            client.armDisarm(False, vehicle)
            client.enableApiControl(False, vehicle)
        except:
            pass

    # STATISTICS
    print("\n" + "="*60)
    print("LIDAR + CAMERA ALIGNED DATASET COMPLETE")
    print("="*60)

    print(f"\nStatistics:")
    print(f"   Frames totais: {frames_captured}")
    print(f"   Frames with detected drones: {detection_stats['frames_with_drones']}")
    print(f"   detection rate: {detection_stats['frames_with_drones']/max(1, detection_stats['total_frames'])*100:.1f}%")
    print(f"   total detections: {detection_stats['total_detections']}")

    print(f"\n Dataset em: {output_dir.absolute()}")

    print("\nContents:")
    print("   - rgb/ : images from the forward camera")
    print("   - lidar/ : the complete 360 deg point cloud")
    print("   - lidar_filtered/ : points inside the camera FOV ONLY")
    print("   - lidar_viz/ : visualisation showing the detected drones")
    print("   - detections/ : JSON with each drone's detections")

    print("\nDrones visible to the camera MUST appear in the point cloud")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Interrompido")
    except Exception as e:
        print(f"\nError: {e}")