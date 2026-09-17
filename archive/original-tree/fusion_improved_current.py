#!/usr/bin/env python3
"""
Fusão melhorada com a configuração atual do LiDAR
Captura múltiplas rotações para aumentar densidade de pontos
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

def capture_dense_pointcloud(client, num_rotations=4):
    """
    Captura nuvem de pontos densa rotacionando o drone
    """
    all_points = []

    for i in range(num_rotations):
        # Rotaciona
        yaw = (360 / num_rotations) * i
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        # Captura LiDAR
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

        if lidar_data and len(lidar_data.point_cloud) > 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            # Transforma pontos para coordenadas globais baseado no yaw
            yaw_rad = np.radians(yaw)
            cos_yaw = np.cos(yaw_rad)
            sin_yaw = np.sin(yaw_rad)

            # Rotação em Z (yaw)
            points_rotated = points.copy()
            points_rotated[:, 0] = points[:, 0] * cos_yaw - points[:, 1] * sin_yaw
            points_rotated[:, 1] = points[:, 0] * sin_yaw + points[:, 1] * cos_yaw

            all_points.append(points_rotated)

    # Combina todos os pontos
    if all_points:
        combined = np.vstack(all_points)
        print(f"   📡 Pontos combinados: {len(combined)} ({num_rotations} rotações)")
        return combined
    return np.array([])


def improved_fusion():
    """
    Fusão melhorada com captura densa
    """
    print("\n" + "="*70)
    print("🚀 FUSÃO MELHORADA COM CAPTURA DENSA")
    print("="*70)

    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"\n✅ Conectado! Veículos: {vehicles}")

    # Prepara drones
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except:
            pass

    # Decola
    print("\n🛫 Decolando...")
    for v in vehicles:
        try:
            client.takeoffAsync(vehicle_name=v)
        except:
            pass
    time.sleep(5)

    # Posiciona Ego em posição estratégica
    print("📍 Posicionando drones...")
    client.moveToPositionAsync(0, 0, -30, 5, vehicle_name="Ego").join()

    # Posiciona outros drones em grid
    positions = [
        (20, -10, -5),   # Drone baixo frontal esquerdo
        (25, 10, -10),   # Drone médio frontal direito
        (30, 0, -15),    # Drone alto frontal centro
        (15, -15, -20),  # Drone lateral
        (15, 15, -25),   # Drone lateral oposto
    ]

    for i, v in enumerate(vehicles[1:min(6, len(vehicles))]):
        if i < len(positions):
            x, y, z = positions[i]
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=v)
            print(f"   {v}: {x}m frente, {y}m lateral, {-z}m altura")

    time.sleep(3)

    output_dir = Path("improved_fusion_output")
    output_dir.mkdir(exist_ok=True)

    print("\n📸 CAPTURANDO COM MÚLTIPLAS ROTAÇÕES...")

    for frame in range(3):
        print(f"\n🔄 Frame {frame+1}/3:")

        # Captura imagem frontal
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Captura nuvem densa
        print("   Capturando nuvem de pontos densa...")
        points_dense = capture_dense_pointcloud(client, num_rotations=4)

        # Volta para frente
        client.rotateToYawAsync(0, vehicle_name="Ego").join()

        if len(points_dense) > 0:
            # Filtra FOV da câmera
            x = points_dense[:, 0]
            y = points_dense[:, 1]
            angles = np.degrees(np.arctan2(y, x))
            in_fov = (x > 0) & (angles > -45) & (angles < 45)
            points_fov = points_dense[in_fov]

            print(f"   Pontos no FOV: {len(points_fov)}")

            # Analisa distribuição
            if len(points_fov) > 0:
                z_vals = points_fov[:, 2]
                print(f"   Z range: {z_vals.min():.1f} a {z_vals.max():.1f}m")

                # Categoriza pontos
                ground = np.sum((z_vals > 0) & (z_vals < 5))  # Chão
                aerial = np.sum((z_vals > -30) & (z_vals < 0))  # Aéreo
                print(f"   Chão (Z>0): {ground} pts, Aéreo (Z<0): {aerial} pts")

            # Cria fusão
            if img_bgr is not None:
                fusion = create_improved_fusion(img_bgr, points_fov)
                cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion)
                cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)

                # Comparação
                h, w = img_bgr.shape[:2]
                comp = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
                comp[:, :w] = img_bgr
                comp[:, w+20:] = fusion
                cv2.putText(comp, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(comp, "Dense LiDAR Fusion", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comp)

                print(f"   ✅ Salvo!")

        # Move Ego para nova posição
        if frame == 0:
            client.moveToPositionAsync(10, 10, -25, 5, vehicle_name="Ego")
        elif frame == 1:
            client.moveToPositionAsync(-10, -10, -20, 5, vehicle_name="Ego")

        time.sleep(2)

    # Pousa
    print("\n🛬 Finalizando...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    print("\n" + "="*70)
    print("✅ FUSÃO DENSA COMPLETA!")
    print("="*70)
    print(f"\n📁 Resultados em: {output_dir.absolute()}")


def create_improved_fusion(img, points):
    """Cria fusão visual melhorada"""
    h, w = img.shape[:2]

    # Calibração da câmera
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    # Cria camadas para diferentes profundidades
    layers = {
        'near': {'mask': np.zeros((h, w), dtype=np.uint8), 'color': (0, 255, 0)},     # Verde
        'mid': {'mask': np.zeros((h, w), dtype=np.uint8), 'color': (0, 255, 255)},    # Amarelo
        'far': {'mask': np.zeros((h, w), dtype=np.uint8), 'color': (0, 0, 255)},      # Vermelho
    }

    if len(points) > 0:
        # Filtra pontos válidos
        valid = points[:, 0] > 0.5
        points_valid = points[valid]

        if len(points_valid) > 0:
            X = points_valid[:, 0]
            Y = points_valid[:, 1]
            Z = points_valid[:, 2]

            # Projeta
            u = fx * (Y / X) + cx
            v = fy * (Z / X) + cy

            # Filtra dentro da imagem
            in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[in_img].astype(int)
            v = v[in_img].astype(int)
            depths = X[in_img]

            # Categoriza por profundidade
            for i in range(len(u)):
                if depths[i] < 20:
                    cv2.circle(layers['near']['mask'], (u[i], v[i]), 3, 255, -1)
                elif depths[i] < 50:
                    cv2.circle(layers['mid']['mask'], (u[i], v[i]), 2, 255, -1)
                else:
                    cv2.circle(layers['far']['mask'], (u[i], v[i]), 1, 255, -1)

    # Aplica camadas na imagem
    fusion = img.copy()

    for layer_name, layer in layers.items():
        if layer['mask'].any():
            colored = np.zeros_like(img)
            colored[layer['mask'] > 0] = layer['color']
            fusion = cv2.addWeighted(fusion, 1.0, colored, 0.5, 0)

    # Adiciona informações
    cv2.putText(fusion, f"Dense Fusion: {len(points)} pts", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Legenda
    cv2.rectangle(fusion, (w-150, h-100), (w-10, h-10), (0, 0, 0), -1)
    cv2.putText(fusion, "Distance:", (w-140, h-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(fusion, "< 20m", (w-140, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(fusion, "20-50m", (w-140, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    cv2.putText(fusion, "> 50m", (w-140, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    return fusion


if __name__ == "__main__":
    improved_fusion()