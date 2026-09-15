#!/usr/bin/env python3
"""ARCHIVED. LiDAR-camera fusion, corrected projection.

Projects the LiDAR cloud onto the RGB image with the corrected frame convention.
The single surviving representative of a thirteen-file family of Phase-1
projection scripts; it produces the LiDAR-returns-on-image view of the chapter.

NOT the chapter's fusion: the late fusion of the architectural comparison is a
decision-level union of two detection dumps and lives in evaluation/fuse_dumps.py.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
from common.config import airsim_host, airsim_port

import numpy as np
import cv2
import cosysairsim as airsim
import time
from pathlib import Path
import matplotlib.pyplot as plt

class CorrectedLidarCameraFusion:
    def __init__(self):
        # Camera parameters (1280x720, FOV 90 x 60 deg)
        self.img_width = 1280
        self.img_height = 720
        self.fov_h = np.radians(90)
        self.fov_v = np.radians(60)

        # Compute the intrinsic parameters
        self.fx = self.img_width / (2 * np.tan(self.fov_h / 2))
        self.fy = self.img_height / (2 * np.tan(self.fov_v / 2))
        self.cx = self.img_width / 2
        self.cy = self.img_height / 2

        print(f"Camera configured:")
        print(f"   resolution: {self.img_width}x{self.img_height}")
        print(f"   FOV: {np.degrees(self.fov_h):.0f}° x {np.degrees(self.fov_v):.0f}°")
        print(f"   Focal: fx={self.fx:.1f}, fy={self.fy:.1f}")

    def transform_lidar_to_camera(self, points_lidar):
        """
        Transform points from the LiDAR frame into the camera frame

        LiDAR no AirSim (confirmado):
        - X: forward (positive = ahead)
        - Y: right (positive = to the right)
        - Z: down (positive = below the sensor, which is why only the ground is seen)

        Camera (typical convention):
        - X: right
        - Y: down
        - Z: forward

        Transform required:
        - X_lidar -> Z_camera (forward)
        - Y_lidar -> X_camera (right)
        - Z_lidar -> Y_camera (down)
        """

        # IMPORTANT: the LiDAR Z axis is positive DOWNWARD
        # which is why only the ground is visible
        # Invert Z, so objects above the sensor come out positive

        points_camera = np.column_stack([
            points_lidar[:, 1],   # Y_lidar -> X_camera (right)
            points_lidar[:, 2],   # Z_lidar -> Y_camera (down)
            points_lidar[:, 0]    # X_lidar -> Z_camera (forward)
        ])

        return points_camera

    def project_to_image(self, points_camera):
        """
        Project 3D points from the camera frame to 2D pixels
        """
        # Discard points behind the camera
        valid = points_camera[:, 2] > 0.5  # Z > 0.5 m (in front)
        points = points_camera[valid]

        if len(points) == 0:
            return np.array([]), np.array([])

        # Perspective projection
        x = points[:, 0]  # lateral
        y = points[:, 1]  # vertical
        z = points[:, 2]  # depth

        # Project onto the image plane
        u = (self.fx * x / z) + self.cx
        v = (self.fy * y / z) + self.cy

        # Keep the points inside the image
        in_image = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)

        pixels = np.column_stack([u[in_image], v[in_image]]).astype(int)
        depths = z[in_image]

        return pixels, depths

    def create_fusion_image(self, img_rgb, points_lidar):
        """
        Build the fused image with the LiDAR points projected correctly
        """
        # Transform the points into the camera frame
        points_camera = self.transform_lidar_to_camera(points_lidar)

        # Project into the image
        pixels, depths = self.project_to_image(points_camera)

        # Build the visualisation
        fusion_img = img_rgb.copy()

        if len(pixels) > 0:
            # Normalise the depths for colouring
            depth_min = max(0.5, depths.min())
            depth_max = min(50, depths.max())
            depths_norm = np.clip((depths - depth_min) / (depth_max - depth_min), 0, 1)

            # Map to colours (jet colormap)
            colors = plt.cm.jet(depths_norm)[:, :3] * 255

            # Draw the points
            for i, (u, v) in enumerate(pixels):
                # Size from the range (closer = larger)
                size = max(1, int(5 * (1 - depths_norm[i])))
                color = colors[i].astype(int).tolist()
                cv2.circle(fusion_img, (u, v), size, color, -1)

            # Add the information
            self.add_info_overlay(fusion_img, len(pixels), depth_min, depth_max)

        return fusion_img

    def add_info_overlay(self, img, num_points, depth_min, depth_max):
        """Draw the information onto the image."""

        # Information panel
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        cv2.putText(img, "LiDAR-Camera Fusion (CORRECTED)", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"Projected Points: {num_points}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Depth Range: {depth_min:.1f} - {depth_max:.1f}m", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Colour bar
        h, w = img.shape[:2]
        bar_height = 200
        bar_width = 30
        bar_x = w - 60
        bar_y = h - bar_height - 60

        for i in range(bar_height):
            depth_norm = 1 - (i / bar_height)
            color = plt.cm.jet(depth_norm)[:3]
            color_bgr = tuple([int(c * 255) for c in color])
            cv2.line(img, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i), color_bgr, 1)

        cv2.rectangle(img, (bar_x-1, bar_y-1), (bar_x+bar_width+1, bar_y+bar_height+1), (255,255,255), 2)
        cv2.putText(img, f"{depth_max:.0f}m", (bar_x-45, bar_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, f"{depth_min:.0f}m", (bar_x-45, bar_y+bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


def test_corrected_fusion():
    """Test the corrected fusion live."""

    print("\n" + "="*70)
    print("CORRECTED LIDAR-CAMERA FUSION TEST")
    print("="*70)

    # Conecta
    client = airsim.MultirotorClient(ip=airsim_host(), port=airsim_port())
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"\nConnected. Vehicles: {vehicles}")

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
            client.takeoffAsync(vehicle_name=v)
        except:
            pass
    time.sleep(5)

    # Place the drones for the test
    print("Posicionando drones...")

    # Ego: observer at mid altitude
    client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

    # The other drones at different positions and ALTITUDES
    # IMPORTANT: place the drones at DIFFERENT ALTITUDES, to see whether they show up
    if "Drone3" in vehicles:
        client.moveToPositionAsync(15, -5, -20, 5, vehicle_name="Drone3").join()  # same altitude
        print("   Drone3: 15 m ahead, 5 m left, same altitude")

    if "Drone4" in vehicles:
        client.moveToPositionAsync(20, 0, -15, 5, vehicle_name="Drone4").join()  # higher
        print("   Drone4: 20 m ahead, centred, 5 m above")

    if "Intruder1" in vehicles:
        client.moveToPositionAsync(25, 5, -25, 5, vehicle_name="Intruder1").join()  # lower
        print("   Intruder1: 25 m ahead, 5 m right, 5 m below")

    time.sleep(3)

    # Initialise the fusion
    fusion = CorrectedLidarCameraFusion()

    output_dir = Path("corrected_fusion_output")
    output_dir.mkdir(exist_ok=True)

    print("\nCapturing and fusing...")

    for frame in range(5):
        print(f"\n Frame {frame+1}/5:")

        # Rotate the ego, to vary the viewpoint
        yaw = frame * 20
        client.rotateToYawAsync(yaw, vehicle_name="Ego")

        # Captura RGB
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Captura LiDAR
            lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

            if lidar_data and len(lidar_data.point_cloud) > 3:
                points_raw = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

                # Keep the points inside the approximate camera FOV
                x = points_raw[:, 0]  # forward
                y = points_raw[:, 1]  # lateral

                # FOV horizontal
                angles = np.degrees(np.arctan2(y, x))
                in_fov = (x > 0) & (angles > -45) & (angles < 45)
                points_fov = points_raw[in_fov]

                print(f"   total points: {len(points_raw)}")
                print(f"   points inside the FOV: {len(points_fov)}")

                # Quick analysis
                if len(points_fov) > 0:
                    z_range = points_fov[:, 2]
                    print(f"   Z range (vertical): {z_range.min():.2f} a {z_range.max():.2f}")

                    # Check whether there are points above the ground
                    above_ground = np.sum(z_range < 0)  # negative Z = above, in the world frame
                    print(f"   points above the ground (Z<0): {above_ground}")

                # Build the fusion
                fusion_img = fusion.create_fusion_image(img_bgr, points_fov)

                # Save the results
                cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)
                cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion_img)

                # Side-by-side comparison
                h, w = img_bgr.shape[:2]
                comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
                comparison[:, :w] = img_bgr
                comparison[:, w+20:] = fusion_img
                cv2.putText(comparison, "Original", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(comparison, "Fusion Corrected", (w+30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comparison)
                print(f"   Saved")

        time.sleep(0.5)

    # Pousa
    print("\n Finalizando...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    print("\n" + "="*70)
    print("TESTE COMPLETO!")
    print("="*70)
    print(f"\nResults in: {output_dir.absolute()}")
    print("\n NOTA IMPORTANTE:")
    print("   The AirSim LiDAR mostly returns from the GROUND")
    print("   which is why there are many terrain points and few airborne ones")
    print("   This is a limitation of the sensor's default configuration")


if __name__ == "__main__":
    test_corrected_fusion()