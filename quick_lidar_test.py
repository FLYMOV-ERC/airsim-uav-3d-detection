#!/usr/bin/env python3
"""
Teste rápido do LiDAR melhorado
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 TESTE RÁPIDO DO LIDAR MELHORADO")
print("="*70)

# Conecta
print("\n🔌 Conectando...")
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"✅ Conectado! Veículos: {vehicles}")

# Prepara Ego
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")

# Decola e posiciona
print("🛫 Decolando...")
client.takeoffAsync(vehicle_name="Ego").join()
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

time.sleep(3)

# Captura LiDAR
print("\n📡 Capturando LiDAR...")
lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    print("\n" + "="*50)
    print("📊 RESULTADOS DO LIDAR MELHORADO:")
    print("="*50)

    print(f"\n🎯 Total de pontos capturados: {len(points):,}")

    if len(points) > 20000:
        print(f"✅ SUCESSO! {len(points)/8192:.1f}x mais pontos que antes!")
    elif len(points) > 10000:
        print(f"⚠️ Melhoria parcial: {len(points)/8192:.1f}x mais pontos")
    else:
        print(f"❌ Ainda com configuração antiga: apenas {len(points)} pontos")

    print(f"\n📏 Alcances capturados:")
    print(f"  X (frente): {points[:,0].min():.1f} a {points[:,0].max():.1f}m")
    print(f"  Y (lateral): {points[:,1].min():.1f} a {points[:,1].max():.1f}m")
    print(f"  Z (vertical): {points[:,2].min():.1f} a {points[:,2].max():.1f}m")

    # Analisa distribuição vertical
    z_neg = np.sum(points[:,2] < -2)  # Acima do sensor
    z_near = np.sum((points[:,2] >= -2) & (points[:,2] <= 2))  # Nível
    z_pos = np.sum(points[:,2] > 2)  # Abaixo do sensor

    print(f"\n📊 Distribuição Vertical dos pontos:")
    print(f"  Acima (Z < -2m): {z_neg:,} pontos ({z_neg/len(points)*100:.1f}%)")
    print(f"  Nível (-2 a 2m): {z_near:,} pontos ({z_near/len(points)*100:.1f}%)")
    print(f"  Abaixo (Z > 2m): {z_pos:,} pontos ({z_pos/len(points)*100:.1f}%)")

    if z_neg > 1000:
        print("\n✅ EXCELENTE! Agora captura objetos ACIMA!")
        print("   O FOV vertical está funcionando corretamente!")
    elif z_neg > 100:
        print("\n⚠️ Captura alguns objetos acima, mas ainda limitado")
    else:
        print("\n❌ Ainda não captura objetos acima (FOV ainda olhando para baixo)")

    # Captura uma imagem para fusão
    print("\n📸 Capturando imagem...")
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Cria fusão simples
        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Projeta pontos frontais
        front = points[points[:, 0] > 0]
        if len(front) > 0:
            # Parâmetros da câmera
            fx = w / (2 * np.tan(np.radians(45)))
            fy = h / (2 * np.tan(np.radians(30)))
            cx = w / 2
            cy = h / 2

            # Projeta
            X = front[:, 0]
            Y = front[:, 1]
            Z = front[:, 2]

            u = (fx * Y / X + cx).astype(int)
            v = (fy * Z / X + cy).astype(int)

            # Filtra pontos na imagem
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[valid]
            v = v[valid]
            depths = X[valid]

            print(f"\n🎨 Projetando {len(u)} pontos na imagem...")

            # Desenha pontos
            for i in range(min(len(u), 10000)):  # Limita para não travar
                depth_norm = min(1.0, depths[i] / 100)
                color = (
                    int(255 * depth_norm),      # Vermelho = longe
                    int(255 * (1 - depth_norm)), # Verde = perto
                    0
                )
                cv2.circle(fusion, (u[i], v[i]), 2, color, -1)

            # Info
            cv2.putText(fusion, f"LiDAR: {len(points):,} pts", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(fusion, f"Projected: {len(u):,} pts", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Salva
            output_dir = Path("lidar_test_output")
            output_dir.mkdir(exist_ok=True)

            cv2.imwrite(str(output_dir / "fusion_test.png"), fusion)
            cv2.imwrite(str(output_dir / "original.png"), img_bgr)
            np.save(output_dir / "points.npy", points)

            print(f"\n💾 Resultados salvos em: {output_dir.absolute()}")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

print("\n" + "="*70)
print("✅ TESTE COMPLETO!")
print("="*70)

if 'points' in locals():
    if len(points) > 20000:
        print("\n🎉 CONFIGURAÇÃO MELHORADA FUNCIONANDO!")
        print(f"   • {len(points):,} pontos capturados")
        print("   • Cobertura vertical completa")
        print("   • Pronto para fusão de alta qualidade!")
    else:
        print("\n⚠️ Ainda usando configuração antiga")
        print("   Verifique se o settings.json foi salvo corretamente")