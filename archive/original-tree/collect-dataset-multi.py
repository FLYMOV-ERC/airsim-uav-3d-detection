"""
collect-dataset-multi.py
Enhanced dataset collection with multiple drones and varied scenarios
Supports 4+ drones with diverse movement patterns and environmental conditions
"""

import os
import json
import time
import math
import argparse
import threading
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
from scene_variation import SceneVariation

# AirSim import with fallback
try:
    import airsim
except Exception:
    import cosysairsim as airsim


# ====================== Movement Patterns ======================

class DroneMovementPatterns:
    """Diverse movement patterns for multiple drones"""

    @staticmethod
    def formation_v(drones: List[str], leader_pos: Tuple[float, float, float],
                   spacing: float = 10.0, angle: float = 30.0) -> Dict[str, Tuple]:
        """V formation pattern"""
        positions = {}
        leader_x, leader_y, leader_z = leader_pos

        for i, drone in enumerate(drones):
            if i == 0:
                positions[drone] = leader_pos
            else:
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                offset_x = -row * spacing * math.cos(math.radians(angle))
                offset_y = side * row * spacing * math.sin(math.radians(angle))
                offset_z = random.uniform(-2, 2)  # Small vertical variation

                positions[drone] = (
                    leader_x + offset_x,
                    leader_y + offset_y,
                    leader_z + offset_z
                )

        return positions

    @staticmethod
    def circular_orbit(center: Tuple[float, float, float], radius: float,
                      num_drones: int, time_factor: float) -> List[Tuple]:
        """Circular orbit pattern around a center point"""
        positions = []
        cx, cy, cz = center

        for i in range(num_drones):
            angle = (2 * math.pi * i / num_drones) + time_factor
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            z = cz + random.uniform(-3, 3)
            positions.append((x, y, z))

        return positions

    @staticmethod
    def random_swarm(center: Tuple[float, float, float], bounds: float,
                    num_drones: int) -> List[Tuple]:
        """Random swarm within bounded area"""
        positions = []
        cx, cy, cz = center

        for _ in range(num_drones):
            x = cx + random.uniform(-bounds, bounds)
            y = cy + random.uniform(-bounds, bounds)
            z = cz + random.uniform(-bounds/2, bounds/2)
            positions.append((x, y, z))

        return positions

    @staticmethod
    def pursuit_evasion(pursuer_pos: Tuple, evader_pos: Tuple,
                       escape_vector: Optional[Tuple] = None) -> Tuple[Tuple, Tuple]:
        """Pursuit-evasion pattern between two drones"""
        px, py, pz = pursuer_pos
        ex, ey, ez = evader_pos

        # Calculate direction from evader to pursuer
        dx = px - ex
        dy = py - ey
        dz = pz - ez
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < 0.1:
            dist = 0.1

        # Pursuer moves toward evader
        pursuit_speed = 8.0
        pursuer_vel = (
            -dx / dist * pursuit_speed,
            -dy / dist * pursuit_speed,
            -dz / dist * pursuit_speed
        )

        # Evader moves away with some randomness
        if escape_vector:
            evx, evy, evz = escape_vector
        else:
            evx = dx / dist + random.uniform(-0.3, 0.3)
            evy = dy / dist + random.uniform(-0.3, 0.3)
            evz = dz / dist + random.uniform(-0.2, 0.2)

        evasion_speed = 7.0
        evader_vel = (
            evx * evasion_speed,
            evy * evasion_speed,
            evz * evasion_speed
        )

        return pursuer_vel, evader_vel

    @staticmethod
    def figure_eight(center: Tuple[float, float, float], scale: float,
                    time_factor: float) -> Tuple[float, float, float]:
        """Figure-eight pattern"""
        cx, cy, cz = center

        # Lemniscate parametric equations
        t = time_factor
        x = cx + scale * math.cos(t) / (1 + math.sin(t)**2)
        y = cy + scale * math.sin(t) * math.cos(t) / (1 + math.sin(t)**2)
        z = cz + 2 * math.sin(t * 0.5)  # Vertical oscillation

        return (x, y, z)


# ====================== Multi-Drone Controller ======================

class MultiDroneController:
    """Controller for multiple drones with collision avoidance"""

    def __init__(self, client: airsim.MultirotorClient, drone_names: List[str],
                 min_separation: float = 8.0):
        self.client = client
        self.drone_names = drone_names
        self.min_separation = min_separation
        self.drone_states = {}
        self.movement_patterns = DroneMovementPatterns()
        self.pattern_time = 0.0
        self.current_pattern = 'mixed'
        self.running = True

    def initialize_drones(self):
        """Initialize all drones"""
        for drone in self.drone_names:
            try:
                self.client.enableApiControl(True, drone)
                self.client.armDisarm(True, drone)
                print(f"Initialized drone: {drone}")
            except Exception as e:
                print(f"Error initializing {drone}: {e}")

    def takeoff_all(self, altitude: float = -10.0):
        """Takeoff all drones"""
        tasks = []
        for drone in self.drone_names:
            try:
                task = self.client.takeoffAsync(vehicle_name=drone)
                tasks.append(task)
            except Exception as e:
                print(f"Error taking off {drone}: {e}")

        # Wait for all takeoffs
        for task in tasks:
            try:
                task.join()
            except:
                pass

        # Move to initial altitude
        time.sleep(2)
        tasks = []
        for i, drone in enumerate(self.drone_names):
            try:
                # Spread initial positions
                x = i * 15 - 30
                y = (i % 2) * 10 - 5
                task = self.client.moveToPositionAsync(
                    x, y, altitude, 5, vehicle_name=drone
                )
                tasks.append(task)
            except Exception as e:
                print(f"Error positioning {drone}: {e}")

        for task in tasks:
            try:
                task.join()
            except:
                pass

    def get_drone_position(self, drone: str) -> Tuple[float, float, float]:
        """Get current position of a drone"""
        try:
            pose = self.client.simGetVehiclePose(vehicle_name=drone)
            return (pose.position.x_val, pose.position.y_val, pose.position.z_val)
        except:
            return (0, 0, 0)

    def check_collision_risk(self, drone1: str, drone2: str) -> bool:
        """Check if two drones are too close"""
        pos1 = self.get_drone_position(drone1)
        pos2 = self.get_drone_position(drone2)

        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        return dist < self.min_separation

    def apply_collision_avoidance(self, velocities: Dict[str, Tuple]) -> Dict[str, Tuple]:
        """Apply collision avoidance to velocity commands"""
        adjusted = velocities.copy()

        for i, drone1 in enumerate(self.drone_names):
            for drone2 in self.drone_names[i+1:]:
                if self.check_collision_risk(drone1, drone2):
                    # Get positions
                    pos1 = self.get_drone_position(drone1)
                    pos2 = self.get_drone_position(drone2)

                    # Calculate repulsion vector
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dz = pos1[2] - pos2[2]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                    if dist > 0.1:
                        # Apply repulsion force
                        repel_strength = 10.0 * (1.0 - dist / self.min_separation)
                        repel1 = (dx/dist * repel_strength,
                                 dy/dist * repel_strength,
                                 dz/dist * repel_strength)
                        repel2 = (-dx/dist * repel_strength,
                                 -dy/dist * repel_strength,
                                 -dz/dist * repel_strength)

                        # Adjust velocities
                        v1 = adjusted.get(drone1, (0, 0, 0))
                        v2 = adjusted.get(drone2, (0, 0, 0))

                        adjusted[drone1] = (v1[0] + repel1[0],
                                          v1[1] + repel1[1],
                                          v1[2] + repel1[2])
                        adjusted[drone2] = (v2[0] + repel2[0],
                                          v2[1] + repel2[1],
                                          v2[2] + repel2[2])

                        print(f"[COLLISION AVOIDANCE] {drone1} <-> {drone2}, dist: {dist:.1f}m")

        return adjusted

    def execute_movement_pattern(self, pattern: str, ego_pos: Tuple[float, float, float]):
        """Execute a specific movement pattern for all intruder drones"""
        velocities = {}
        intruders = [d for d in self.drone_names if d != 'Ego']

        if pattern == 'formation_v':
            positions = self.movement_patterns.formation_v(
                intruders, ego_pos, spacing=12.0
            )
            for drone, target in positions.items():
                current = self.get_drone_position(drone)
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                dz = target[2] - current[2]
                velocities[drone] = (dx * 0.5, dy * 0.5, dz * 0.3)

        elif pattern == 'circular_orbit':
            positions = self.movement_patterns.circular_orbit(
                ego_pos, radius=25.0, num_drones=len(intruders),
                time_factor=self.pattern_time * 0.5
            )
            for drone, target in zip(intruders, positions):
                current = self.get_drone_position(drone)
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                dz = target[2] - current[2]
                velocities[drone] = (dx * 0.8, dy * 0.8, dz * 0.3)

        elif pattern == 'random_swarm':
            positions = self.movement_patterns.random_swarm(
                ego_pos, bounds=30.0, num_drones=len(intruders)
            )
            for drone, target in zip(intruders, positions):
                current = self.get_drone_position(drone)
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                dz = target[2] - current[2]
                velocities[drone] = (dx * 0.4, dy * 0.4, dz * 0.3)

        elif pattern == 'pursuit_pairs':
            # Pair up drones for pursuit-evasion
            for i in range(0, len(intruders)-1, 2):
                pursuer = intruders[i]
                evader = intruders[i+1]
                p_pos = self.get_drone_position(pursuer)
                e_pos = self.get_drone_position(evader)
                p_vel, e_vel = self.movement_patterns.pursuit_evasion(p_pos, e_pos)
                velocities[pursuer] = p_vel
                velocities[evader] = e_vel

        elif pattern == 'mixed':
            # Mix different patterns for different drones
            if len(intruders) >= 2:
                # First two drones do pursuit-evasion
                p_pos = self.get_drone_position(intruders[0])
                e_pos = self.get_drone_position(intruders[1])
                p_vel, e_vel = self.movement_patterns.pursuit_evasion(p_pos, e_pos)
                velocities[intruders[0]] = p_vel
                velocities[intruders[1]] = e_vel

            if len(intruders) >= 3:
                # Third drone does figure-eight
                target = self.movement_patterns.figure_eight(
                    ego_pos, scale=20.0, time_factor=self.pattern_time
                )
                current = self.get_drone_position(intruders[2])
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                dz = target[2] - current[2]
                velocities[intruders[2]] = (dx * 0.6, dy * 0.6, dz * 0.3)

            if len(intruders) >= 4:
                # Fourth drone orbits
                positions = self.movement_patterns.circular_orbit(
                    ego_pos, radius=20.0, num_drones=1,
                    time_factor=self.pattern_time * 0.7
                )
                target = positions[0]
                current = self.get_drone_position(intruders[3])
                dx = target[0] - current[0]
                dy = target[1] - current[1]
                dz = target[2] - current[2]
                velocities[intruders[3]] = (dx * 0.7, dy * 0.7, dz * 0.3)

        return velocities

    def movement_loop(self, ego_name: str = 'Ego', ego_speed: float = 6.0):
        """Main movement control loop"""
        target_altitude = -10.0

        while self.running:
            try:
                # Get Ego state
                ego_state = self.client.getMultirotorState(vehicle_name=ego_name)
                ego_pos = (
                    ego_state.kinematics_estimated.position.x_val,
                    ego_state.kinematics_estimated.position.y_val,
                    ego_state.kinematics_estimated.position.z_val
                )

                # Calculate Ego movement (forward motion)
                ego_yaw = self.yaw_from_quat(ego_state.kinematics_estimated.orientation)
                ego_vx = ego_speed * math.cos(ego_yaw)
                ego_vy = ego_speed * math.sin(ego_yaw)
                ego_vz = (target_altitude - ego_pos[2]) * 0.2
                ego_vz = max(-1.0, min(1.0, ego_vz))

                # Command Ego movement
                self.client.moveByVelocityAsync(
                    ego_vx, ego_vy, ego_vz,
                    duration=0.5,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(False, 0),
                    vehicle_name=ego_name
                )

                # Get movement pattern velocities for intruders
                velocities = self.execute_movement_pattern(self.current_pattern, ego_pos)

                # Add Ego velocity compensation
                for drone in velocities:
                    v = velocities[drone]
                    velocities[drone] = (v[0] + ego_vx, v[1] + ego_vy, v[2])

                # Apply collision avoidance
                velocities = self.apply_collision_avoidance(velocities)

                # Command intruder movements
                for drone, (vx, vy, vz) in velocities.items():
                    # Limit velocities
                    vx = max(-10, min(10, vx))
                    vy = max(-10, min(10, vy))
                    vz = max(-3, min(3, vz))

                    self.client.moveByVelocityAsync(
                        vx, vy, vz,
                        duration=0.5,
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(False, 0),
                        vehicle_name=drone
                    )

                # Update pattern time
                self.pattern_time += 0.1

                # Occasionally change pattern
                if random.random() < 0.005:  # 0.5% chance per iteration
                    patterns = ['formation_v', 'circular_orbit', 'random_swarm',
                              'pursuit_pairs', 'mixed']
                    self.current_pattern = random.choice(patterns)
                    print(f"Switching to pattern: {self.current_pattern}")

            except Exception as e:
                print(f"Movement loop error: {e}")

            time.sleep(0.1)

    def yaw_from_quat(self, q):
        """Convert quaternion to yaw angle"""
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop(self):
        """Stop movement controller"""
        self.running = False


# ====================== Utils ======================

def fx_fy_from_fov(width, height, fov_deg):
    f = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return f, f

def pose_to_dict(pose):
    return {
        "position": {
            "x": pose.position.x_val,
            "y": pose.position.y_val,
            "z": pose.position.z_val,
        },
        "orientation_xyzw": [
            pose.orientation.x_val,
            pose.orientation.y_val,
            pose.orientation.z_val,
            pose.orientation.w_val,
        ],
    }

def lidar_to_xyz(lidar):
    if not getattr(lidar, "point_cloud", None):
        return np.zeros((0, 3), np.float32)
    return np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)

def save_image(path, img_bytes, width, height):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    if arr.size == width * height * 3:
        img = arr.reshape(height, width, 3)
    elif arr.size == width * height * 4:
        img = arr.reshape(height, width, 4)[:, :, :3]
    else:
        img = arr.reshape(height, width, -1)[:, :, :3]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


# ===================== Main ========================

def main():
    ap = argparse.ArgumentParser(description="Collect multi-drone dataset with varied scenarios")
    ap.add_argument("--ip", default="172.19.80.1")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--vehicle", default="Ego", help="Main observation drone")
    ap.add_argument("--intruders", nargs="+",
                   default=["Intruder1", "Intruder2", "Intruder3", "Intruder4"],
                   help="List of intruder drone names")
    ap.add_argument("--lidar", default="LidarFront")
    ap.add_argument("--cams", nargs="+", default=["front_center"])
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--out", default="dataset_multi")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--vary_weather", action="store_true", help="Enable weather variations")
    ap.add_argument("--vary_time", action="store_true", help="Enable time of day variations")
    ap.add_argument("--weather_interval", type=int, default=200,
                   help="Frames between weather changes")
    ap.add_argument("--time_interval", type=int, default=300,
                   help="Frames between time changes")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    # Create output directories
    out = Path(args.out)
    for sub in ["images", "seg", "lidar", "meta"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
        if sub in ["images", "seg"]:
            for cam in args.cams:
                (out / sub / cam).mkdir(parents=True, exist_ok=True)

    # Connect to AirSim
    client = airsim.MultirotorClient(ip=args.ip, port=args.port)
    client.confirmConnection()
    print("✅ Connected to AirSim")

    # List available vehicles
    try:
        vehicles = client.listVehicles()
        print(f"Available vehicles: {vehicles}")
    except:
        pass

    # Initialize scene variation
    scene_var = SceneVariation(
        client,
        enable_weather=args.vary_weather,
        enable_time_changes=args.vary_time
    )

    # Set initial conditions
    scene_var.set_weather('clear')
    scene_var.set_time_of_day('noon')

    # Initialize multi-drone controller
    all_drones = [args.vehicle] + args.intruders
    controller = MultiDroneController(client, all_drones, min_separation=10.0)

    print("Initializing drones...")
    controller.initialize_drones()

    print("Taking off all drones...")
    controller.takeoff_all(altitude=-10.0)

    # Start movement thread
    movement_thread = threading.Thread(
        target=controller.movement_loop,
        args=(args.vehicle, 6.0),
        daemon=True
    )
    movement_thread.start()

    # Save calibration
    calib = {
        "vehicle": args.vehicle,
        "intruders": args.intruders,
        "cams": {},
        "image_size": [args.width, args.height]
    }

    for cam in args.cams:
        try:
            info = client.simGetCameraInfo(cam, vehicle_name=args.vehicle)
            fx, fy = fx_fy_from_fov(args.width, args.height, info.fov)
            cx, cy = args.width / 2.0, args.height / 2.0
            calib["cams"][cam] = {
                "name": cam,
                "fov_deg": float(info.fov),
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "pose_vehicle": pose_to_dict(info.pose),
            }
        except:
            pass

    with open(out / "calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    # Prepare image requests
    def make_requests():
        reqs = []
        for cam in args.cams:
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False))
            reqs.append(airsim.ImageRequest(cam, airsim.ImageType.Segmentation, False, False))
        return reqs

    print(f"Starting data collection for {args.frames} frames...")
    print(f"Weather variations: {args.vary_weather}, Time variations: {args.vary_time}")

    # Main collection loop
    for i in tqdm(range(1, args.frames + 1), desc="Collecting"):
        # Vary weather conditions
        if args.vary_weather and i % args.weather_interval == 0:
            weather = scene_var.set_random_weather()
            print(f"  [Frame {i}] Changed weather to: {weather}")

        # Vary time of day
        if args.vary_time and i % args.time_interval == 0:
            time_name = scene_var.set_random_time()
            print(f"  [Frame {i}] Changed time to: {time_name}")

        # Capture images
        responses = client.simGetImages(make_requests(), vehicle_name=args.vehicle)
        it = iter(responses)
        for cam in args.cams:
            scene = next(it)
            seg = next(it)
            img_path = out / "images" / cam / f"{i:06d}.png"
            seg_path = out / "seg" / cam / f"{i:06d}.png"
            save_image(img_path, scene.image_data_uint8, scene.width, scene.height)
            save_image(seg_path, seg.image_data_uint8, seg.width, seg.height)

        # Capture LiDAR
        try:
            lidar = client.getLidarData(lidar_name=args.lidar, vehicle_name=args.vehicle)
            pts = lidar_to_xyz(lidar)
        except:
            pts = np.zeros((0, 3), np.float32)
        np.save(out / "lidar" / f"{i:06d}.npy", pts)

        # Save metadata
        meta = {
            "frame": i,
            "timestamp": time.time(),
            "image_size": {"w": args.width, "h": args.height},
            "cams": args.cams,
            "lidar_points": int(pts.shape[0]),
            "vehicles": {},
            "conditions": scene_var.get_current_conditions(),
            "movement_pattern": controller.current_pattern
        }

        # Save all vehicle poses
        for drone in all_drones:
            try:
                pose = client.simGetVehiclePose(vehicle_name=drone)
                meta["vehicles"][drone] = {"pose": pose_to_dict(pose)}
            except:
                pass

        with open(out / "meta" / f"{i:06d}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Log progress
        if args.log_every and i % args.log_every == 0:
            num_drones = len([v for v in meta["vehicles"] if v in all_drones])
            print(f"  [Frame {i:06d}] Active drones: {num_drones}, "
                  f"Pattern: {controller.current_pattern}, "
                  f"Weather: {meta['conditions']['weather']}, "
                  f"Time: {meta['conditions']['time']}")

        # Small delay
        time.sleep(0.01)

    # Stop controller
    controller.stop()
    movement_thread.join(timeout=1.0)

    # Reset scene
    scene_var.reset_to_default()

    print(f"✅ Dataset collection complete!")
    print(f"   Saved {args.frames} frames to: {out.resolve()}")
    print(f"   Total drones used: {len(all_drones)}")
    print(f"   Weather variations: {args.vary_weather}")
    print(f"   Time variations: {args.vary_time}")


if __name__ == "__main__":
    main()