#!/usr/bin/env python3
"""
Fusão com foco na detecção de drones
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 FUSÃO COM DETECÇÃO DE DRONES")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n✅ Veículos: {vehicles}")

# Prepara todos
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola todos
print("🛫 Decolando todos...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v)
    except:
        pass
time.sleep(5)

# Posiciona drones em formação visível
print("\n📍 Posicionando drones em FORMAÇÃO VISÍVEL...")

# Ego como observador
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()
print("   Ego: Posição de observação (15m altura)")

# Drones em diferentes distâncias e alturas NA FRENTE
if "Drone3" in vehicles:
    client.moveToPositionAsync(10, -3, -12, 5, vehicle_name="Drone3").join()
    print("   Drone3: 10m frente, 3m esquerda, 12m altura (3m ACIMA)")

if "Drone4" in vehicles:
    client.moveToPositionAsync(15, 3, -15, 5, vehicle_name="Drone4").join()
    print("   Drone4: 15m frente, 3m direita, 15m altura (MESMO nível)")

if "Intruder1" in vehicles:
    client.moveToPositionAsync(20, 0, -18, 5, vehicle_name="Intruder1").join()
    print("   Intruder1: 20m frente, centro, 18m altura (3m ABAIXO)")

time.sleep(3)

output_dir = Path("drone_detection_fusion")
output_dir.mkdir(exist_ok=True)

print("\n📸 CAPTURANDO E ANALISANDO...")

for frame in range(3):
    print(f"\n🔄 Frame {frame+1}/3:")

    # Rotaciona Ego para variar perspectiva
    yaw = frame * 15 - 15  # -15, 0, +15 graus
    client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
    time.sleep(1)

    # Captura imagem
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

        print(f"   📡 Total pontos: {len(points):,}")

        # Análise de detecção
        z_values = points[:, 2]

        # Pontos acima (Z < -1m para ignorar ruído)
        above_mask = z_values < -1
        above_points = points[above_mask]

        # Pontos ao nível (-1 a 1m)
        level_mask = (z_values >= -1) & (z_values <= 1)
        level_points = points[level_mask]

        # Pontos abaixo (Z > 1m)
        below_mask = z_values > 1
        below_points = points[below_mask]

        print(f"   📊 Distribuição vertical:")
        print(f"      Acima: {len(above_points)} pontos")
        print(f"      Nível: {len(level_points)} pontos")
        print(f"      Abaixo: {len(below_points)} pontos")

        # Analisa clusters de pontos não-chão
        non_ground = points[(z_values < 15) & (points[:, 0] > 5) & (points[:, 0] < 30)]

        if len(non_ground) > 0:
            print(f"   🎯 Possíveis objetos aéreos: {len(non_ground)} pontos")

            # Agrupa por distância
            for dist, name in [(10, "Drone3"), (15, "Drone4"), (20, "Intruder1")]:
                cluster = non_ground[(non_ground[:,0] > dist-2) & (non_ground[:,0] < dist+2)]
                if len(cluster) > 0:
                    z_mean = cluster[:,2].mean()
                    print(f"      {name} região (~{dist}m): {len(cluster)} pontos, Z médio={z_mean:.1f}m")

        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            fusion = img_bgr.copy()

            # Parâmetros da câmera
            fx = w / (2 * np.tan(np.radians(45)))
            fy = h / (2 * np.tan(np.radians(30)))
            cx = w / 2
            cy = h / 2

            # Filtra pontos frontais no FOV
            front_mask = points[:, 0] > 0.5
            front_points = points[front_mask]

            if len(front_points) > 0:
                # Calcula ângulos horizontais
                angles = np.degrees(np.arctan2(front_points[:, 1], front_points[:, 0]))
                fov_mask = (angles > -45) & (angles < 45)
                points_fov = front_points[fov_mask]

                if len(points_fov) > 0:
                    X = points_fov[:, 0]
                    Y = points_fov[:, 1]
                    Z = points_fov[:, 2]

                    # Projeta
                    u = (fx * Y / X + cx).astype(int)
                    v = (fy * Z / X + cy).astype(int)

                    # Filtra dentro da imagem
                    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                    u = u[valid]
                    v = v[valid]
                    depths = X[valid]
                    z_proj = Z[valid]

                    print(f"   🎨 Projetando {len(u)} pontos")

                    # Desenha com cores diferentes por altura
                    for i in range(len(u)):
                        if z_proj[i] < -1:  # Acima
                            color = (255, 0, 255)  # Magenta = objeto acima
                            size = 4
                        elif z_proj[i] < 1:  # Nível
                            color = (0, 255, 255)  # Amarelo = mesmo nível
                            size = 3
                        elif z_proj[i] < 15:  # Pouco abaixo
                            color = (0, 165, 255)  # Laranja = pouco abaixo
                            size = 2
                        else:  # Chão
                            color = (0, 255, 0)  # Verde = chão
                            size = 1

                        cv2.circle(fusion, (u[i], v[i]), size, color, -1)

                    # Info
                    cv2.rectangle(fusion, (10, 10), (350, 120), (0, 0, 0), -1)
                    cv2.rectangle(fusion, (10, 10), (350, 120), (255, 255, 255), 2)

                    cv2.putText(fusion, f"LiDAR: {len(points)} pts", (20, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(fusion, f"Projetados: {len(u)} pts", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(fusion, f"Acima: {len(above_points)} pts", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                    cv2.putText(fusion, f"Frame: {frame+1}/3, Yaw: {yaw}", (20, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # Legenda
                    cv2.rectangle(fusion, (w-180, h-150), (w-10, h-10), (0, 0, 0), -1)
                    cv2.rectangle(fusion, (w-180, h-150), (w-10, h-10), (255, 255, 255), 2)
                    cv2.putText(fusion, "Altura:", (w-170, h-125),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.circle(fusion, (w-150, h-100), 4, (255, 0, 255), -1)
                    cv2.putText(fusion, "Acima", (w-130, h-95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    cv2.circle(fusion, (w-150, h-75), 3, (0, 255, 255), -1)
                    cv2.putText(fusion, "Nivel", (w-130, h-70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    cv2.circle(fusion, (w-150, h-50), 2, (0, 165, 255), -1)
                    cv2.putText(fusion, "Abaixo", (w-130, h-45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                    cv2.circle(fusion, (w-150, h-25), 1, (0, 255, 0), -1)
                    cv2.putText(fusion, "Chao", (w-130, h-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Salva
                    cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion)
                    cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)

                    # Comparação
                    comp = np.hstack([img_bgr, fusion])
                    cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comp)

                    print(f"   ✅ Salvo!")

# Volta para posição inicial
client.rotateToYawAsync(0, vehicle_name="Ego").join()

# Pousa todos
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("✅ TESTE DE DETECÇÃO COMPLETO!")
print("="*70)
print(f"\n📁 Resultados em: {output_dir.absolute()}")
print("\n💡 CORES NA FUSÃO:")
print("   • MAGENTA = Objetos ACIMA do drone")
print("   • AMARELO = Objetos no MESMO nível")
print("   • LARANJA = Objetos ABAIXO")
print("   • VERDE = Chão")
print("\n🔍 Verifique as imagens para ver se os drones aparecem!")