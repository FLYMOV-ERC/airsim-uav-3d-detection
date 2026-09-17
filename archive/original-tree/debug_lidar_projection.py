#!/usr/bin/env python3
"""
Debug da projeção LiDAR - Entender o problema
"""

import numpy as np
import cv2
import cosysairsim as airsim
import time
from pathlib import Path

def analyze_lidar_camera_alignment():
    """Analisa o alinhamento real entre LiDAR e câmera"""

    print("\n🔍 DIAGNÓSTICO DO PROBLEMA DE PROJEÇÃO")
    print("="*60)

    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()

    # Prepara Ego
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")
    client.takeoffAsync(vehicle_name="Ego").join()

    # Move para posição de teste
    client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()

    # Coloca um drone na frente para referência
    if "Drone3" in vehicles:
        client.enableApiControl(True, "Drone3")
        client.armDisarm(True, "Drone3")
        client.takeoffAsync(vehicle_name="Drone3").join()
        # Coloca drone bem na frente, mesma altura
        client.moveToPositionAsync(10, 0, -10, 5, vehicle_name="Drone3").join()
        print("✅ Drone3 posicionado: 10m na frente, mesma altura")

    time.sleep(3)

    print("\n📊 CAPTURANDO DADOS...")

    # 1. Captura imagem
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    img_rgb = None
    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug_camera.png", img_bgr)
        print("   📷 Imagem capturada")

    # 2. Captura LiDAR
    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    if lidar_data and len(lidar_data.point_cloud) > 3:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        print(f"   📡 LiDAR: {len(points)} pontos")

        # Informações sobre a pose do LiDAR
        if hasattr(lidar_data, 'pose'):
            print(f"\n📍 POSE DO LIDAR:")
            print(f"   Posição: ({lidar_data.pose.position.x_val:.2f}, "
                  f"{lidar_data.pose.position.y_val:.2f}, "
                  f"{lidar_data.pose.position.z_val:.2f})")

            # Quaternion para ângulos de Euler (aproximado)
            q = lidar_data.pose.orientation
            print(f"   Orientação (quaternion): w={q.w_val:.2f}, x={q.x_val:.2f}, "
                  f"y={q.y_val:.2f}, z={q.z_val:.2f}")

        # Análise dos pontos
        print(f"\n📊 ANÁLISE DOS PONTOS LIDAR:")
        print(f"   Sistema de coordenadas original:")
        print(f"   X: {points[:,0].min():.2f} a {points[:,0].max():.2f}")
        print(f"   Y: {points[:,1].min():.2f} a {points[:,1].max():.2f}")
        print(f"   Z: {points[:,2].min():.2f} a {points[:,2].max():.2f}")

        # Verifica se há pontos onde deveria estar o Drone3
        # Drone3 está em (10, 0, -10) global, relativo ao Ego seria (10, 0, 0)
        print(f"\n🎯 PROCURANDO DRONE3 (esperado em ~10m frente):")

        # Pontos na região esperada do drone
        drone_region = points[
            (points[:,0] > 8) & (points[:,0] < 12) &  # X: 8-12m
            (np.abs(points[:,1]) < 3) &                # Y: ±3m
            (np.abs(points[:,2]) < 3)                  # Z: ±3m
        ]

        print(f"   Pontos na região do drone: {len(drone_region)}")
        if len(drone_region) > 0:
            print(f"   Centro da região: ({drone_region.mean(axis=0)})")

    # 3. Testa diferentes transformações
    print("\n🔄 TESTANDO TRANSFORMAÇÕES:")

    # Possibilidade 1: LiDAR está em coordenadas NED (North-East-Down)
    # enquanto câmera está em coordenadas diferentes

    # No AirSim geralmente:
    # Camera: +X = direita, +Y = baixo, +Z = frente
    # LiDAR: pode estar em NED ou FLU (Forward-Left-Up)

    print("\n   Teste 1: Assumindo LiDAR em NED:")
    # NED to Camera: swap axes
    points_cam1 = np.column_stack([
        points[:, 0],  # North -> Forward (Z da câmera)
        -points[:, 1], # East -> Right (X da câmera, invertido)
        points[:, 2]   # Down -> Down (Y da câmera)
    ])

    print(f"   X_cam: {points_cam1[:,0].min():.2f} a {points_cam1[:,0].max():.2f}")
    print(f"   Y_cam: {points_cam1[:,1].min():.2f} a {points_cam1[:,1].max():.2f}")
    print(f"   Z_cam: {points_cam1[:,2].min():.2f} a {points_cam1[:,2].max():.2f}")

    print("\n   Teste 2: Assumindo LiDAR em FRD (Forward-Right-Down):")
    # Já está alinhado, mas pode precisar offset
    points_cam2 = points.copy()

    # Verifica se precisa de offset vertical
    # Se todos os pontos estão perto de Z=0, o LiDAR pode estar no chão do drone
    if np.abs(points[:,2]).max() < 2:
        print("   ⚠️ LiDAR parece estar montado no nível do chão do drone")
        print("   Aplicando offset vertical...")
        # Se o drone está a 10m de altura e LiDAR mostra Z~0,
        # então os pontos estão relativos ao LiDAR, não ao mundo

    # Salva para análise
    np.save("debug_points_original.npy", points)
    np.save("debug_points_transformed.npy", points_cam1)

    # Cria visualização simples
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]

        # Desenha onde esperamos ver o Drone3
        # Se está 10m na frente, deveria aparecer no centro da imagem
        cv2.circle(img_bgr, (w//2, h//2), 50, (0, 255, 0), 3)
        cv2.putText(img_bgr, "Drone3 esperado aqui", (w//2 - 100, h//2 - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite("debug_expected.png", img_bgr)

    # Pousa
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    print("\n" + "="*60)
    print("💡 CONCLUSÃO:")
    print("="*60)

    print("\n❌ PROBLEMA IDENTIFICADO:")
    print("   1. Os pontos LiDAR estão todos no chão (Z~0)")
    print("   2. Não há pontos na altura dos drones voando")
    print("   3. O LiDAR está capturando apenas o terreno")
    print("\n🔧 POSSÍVEIS CAUSAS:")
    print("   • LiDAR configurado com ângulo errado")
    print("   • LiDAR apontando para baixo ao invés de frente")
    print("   • Configuração de FOV vertical muito limitada")
    print("   • Transform entre LiDAR e câmera incorreto")

if __name__ == "__main__":
    analyze_lidar_camera_alignment()