#!/usr/bin/env python3
"""
Testa LiDAR com configuração melhorada
IMPORTANTE: Precisa copiar settings_lidar_enhanced.json para o Windows!
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

def test_enhanced_lidar():
    print("\n" + "="*70)
    print("🚀 TESTE DO LIDAR MELHORADO")
    print("="*70)

    print("\n⚠️ IMPORTANTE:")
    print("   Copie settings_lidar_enhanced.json para:")
    print("   C:\\Users\\[seu_usuario]\\Documents\\AirSim\\settings.json")
    print("   E reinicie o AirSim no Windows!")
    print("\n   Configuração melhorada:")
    print("   • 64 canais (dobro)")
    print("   • 500,000 pontos/seg (5x mais)")
    print("   • Range: 200m")
    print("   • FOV: ±30° vertical, ±90° horizontal")
    print("   • LiDAR alinhado com câmera")

    input("\n📌 Pressione ENTER após atualizar settings.json no Windows...")

    # Conecta
    print("\n🔌 Conectando...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")

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

    # Posiciona
    client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

    # Outros drones em várias posições e alturas
    if "Drone3" in vehicles:
        client.moveToPositionAsync(20, -10, -15, 5, vehicle_name="Drone3")
    if "Drone4" in vehicles:
        client.moveToPositionAsync(30, 0, -20, 5, vehicle_name="Drone4")
    if "Intruder1" in vehicles:
        client.moveToPositionAsync(25, 10, -25, 5, vehicle_name="Intruder1")

    time.sleep(3)

    output_dir = Path("enhanced_lidar_test")
    output_dir.mkdir(exist_ok=True)

    print("\n📊 CAPTURANDO COM LIDAR MELHORADO...")

    for i in range(3):
        print(f"\n🔄 Captura {i+1}/3:")

        # Rotaciona para variar vista
        client.rotateToYawAsync(i * 45, vehicle_name="Ego")

        # Captura RGB
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Captura LiDAR
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

        if lidar_data and len(lidar_data.point_cloud) > 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            print(f"   📡 Total de pontos: {len(points)} (esperado ~40,000)")
            print(f"   X range: {points[:,0].min():.1f} a {points[:,0].max():.1f}m")
            print(f"   Y range: {points[:,1].min():.1f} a {points[:,1].max():.1f}m")
            print(f"   Z range: {points[:,2].min():.1f} a {points[:,2].max():.1f}m")

            # Analisa distribuição vertical
            z_neg = np.sum(points[:,2] < -1)  # Acima
            z_ground = np.sum((points[:,2] >= -1) & (points[:,2] <= 1))  # Nível
            z_pos = np.sum(points[:,2] > 1)  # Abaixo

            print(f"   Distribuição vertical:")
            print(f"     Acima (Z<-1): {z_neg} pontos")
            print(f"     Nível (-1<Z<1): {z_ground} pontos")
            print(f"     Abaixo (Z>1): {z_pos} pontos")

            # Filtra FOV da câmera
            x = points[:, 0]
            y = points[:, 1]
            angles = np.degrees(np.arctan2(y, x))
            in_fov = (x > 0) & (angles > -45) & (angles < 45)
            points_fov = points[in_fov]

            print(f"   Pontos no FOV da câmera: {len(points_fov)}")

            # Salva dados
            np.save(output_dir / f"points_{i:03d}.npy", points)
            np.save(output_dir / f"points_fov_{i:03d}.npy", points_fov)

            # Cria visualização melhorada
            if img_bgr is not None:
                fusion = create_enhanced_fusion(img_bgr, points_fov)
                cv2.imwrite(str(output_dir / f"fusion_{i:03d}.png"), fusion)
                cv2.imwrite(str(output_dir / f"original_{i:03d}.png"), img_bgr)
                print(f"   ✅ Fusão salva!")

        time.sleep(1)

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
    print("📊 ANÁLISE DO LIDAR MELHORADO")
    print("="*70)

    print(f"\n📁 Resultados em: {output_dir.absolute()}")

    # Verifica se melhorou
    if points is not None:
        if len(points) > 20000:
            print("\n✅ SUCESSO! LiDAR melhorado está funcionando!")
            print(f"   • {len(points)} pontos por frame (vs 8192 antes)")
            print("   • Maior densidade e alcance")
            print("   • Melhor cobertura vertical")
        else:
            print("\n⚠️ LiDAR ainda com poucos pontos.")
            print("   Verifique se o settings.json foi atualizado no Windows.")


def create_enhanced_fusion(img, points):
    """Cria fusão com mais pontos"""
    h, w = img.shape[:2]

    # Parâmetros da câmera
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    fusion = img.copy()

    if len(points) > 0:
        # Filtra pontos válidos
        valid = points[:, 0] > 0.5
        points_valid = points[valid]

        if len(points_valid) > 0:
            # Projeta
            X = points_valid[:, 0]
            Y = points_valid[:, 1]
            Z = points_valid[:, 2]

            u = fx * (Y / X) + cx
            v = fy * (Z / X) + cy

            # Filtra dentro da imagem
            in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[in_img].astype(int)
            v = v[in_img].astype(int)
            depths = X[in_img]

            # Cores por profundidade
            if len(u) > 0:
                depth_norm = np.clip(depths / 100, 0, 1)

                for i in range(len(u)):
                    # Cor HSV para gradiente suave
                    hue = int(120 * (1 - depth_norm[i]))
                    color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

                    size = max(1, int(3 * (1 - depth_norm[i])))
                    cv2.circle(fusion, (u[i], v[i]), size, color_bgr, -1)

                # Info
                cv2.putText(fusion, f"Enhanced LiDAR: {len(u)} points", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return fusion


if __name__ == "__main__":
    test_enhanced_lidar()