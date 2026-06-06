#!/usr/bin/env python3
"""
Fusão funcionando com detecção de anomalias
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 FUSÃO COM DETECÇÃO DE ANOMALIAS")
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

# Decola
print("🛫 Decolando...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

print("\n📍 Posicionando drones...")

# Posiciona para melhor detecção
client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()

# Drones próximos em diferentes posições
positions = [
    ("Drone3", 3, -1, -10, "3m frente"),
    ("Drone4", 5, 1, -10, "5m frente"),
    ("Intruder1", 7, 0, -8, "7m frente, 2m acima")
]

for name, x, y, z, desc in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"  {name}: {desc}")

time.sleep(3)

output_dir = Path("fusion_detection_working")
output_dir.mkdir(exist_ok=True)

print("\n📸 CAPTURANDO COM DETECÇÃO...")

for frame in range(3):
    print(f"\n🔄 Frame {frame+1}/3:")

    # Varia ângulo
    yaw = (frame - 1) * 20
    client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
    time.sleep(0.5)

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

    if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        print(f"  📡 Total pontos: {len(points):,}")

        # Detecta anomalias (pontos que não são chão padrão)
        z_values = points[:, 2]

        # Identifica o valor Z do chão (mais comum)
        z_rounded = np.round(z_values, 0)
        z_mode = np.bincount(z_rounded.astype(int)[z_rounded >= 0]).argmax()

        # Pontos anormais = não estão próximos do chão modal
        anomalies = points[np.abs(z_values - z_mode) > 1.0]

        print(f"  📊 Z modal (chão): {z_mode}m")
        print(f"  🎯 Pontos anômalos: {len(anomalies)}")

        # Cria fusão
        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Parâmetros câmera
        fx = w / (2 * np.tan(np.radians(45)))
        fy = h / (2 * np.tan(np.radians(30)))
        cx = w / 2
        cy = h / 2

        # Projeta pontos
        front = points[points[:, 0] > 0.5]

        if len(front) > 0:
            X = front[:, 0]
            Y = front[:, 1]
            Z = front[:, 2]

            u = (fx * Y / X + cx).astype(int)
            v = (fy * Z / X + cy).astype(int)

            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[valid]
            v = v[valid]
            z_vals = Z[valid]
            x_vals = X[valid]

            # Conta detecções por região
            detections = 0

            # Desenha pontos com destaque para anomalias
            for i in range(len(u)):
                # Verifica se é anomalia
                is_anomaly = abs(z_vals[i] - z_mode) > 1.0

                if is_anomaly:
                    # ANOMALIA DETECTADA!
                    color = (255, 0, 255)  # Magenta brilhante
                    size = 5
                    detections += 1
                    # Desenha círculo maior para destacar
                    cv2.circle(fusion, (u[i], v[i]), 8, (255, 255, 0), 2)
                elif x_vals[i] < 10 and abs(z_vals[i] - z_mode) > 0.5:
                    # Possível objeto próximo
                    color = (0, 255, 255)  # Amarelo
                    size = 3
                else:
                    # Chão normal
                    depth_norm = min(1.0, x_vals[i] / 50)
                    green = int(255 * (1 - depth_norm))
                    color = (0, green, 0)
                    size = 1

                cv2.circle(fusion, (u[i], v[i]), size, color, -1)

            print(f"  ✨ Detecções projetadas: {detections}")

            # Interface melhorada
            overlay = fusion.copy()
            cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
            fusion = cv2.addWeighted(fusion, 0.8, overlay, 0.2, 0)

            cv2.putText(fusion, "ANOMALY DETECTION LIDAR", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(fusion, f"Total Points: {len(points):,}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(fusion, f"Anomalies: {len(anomalies)}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, f"Ground Z: {z_mode:.1f}m", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(fusion, f"Detections: {detections}", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Legenda
            legend = fusion.copy()
            cv2.rectangle(legend, (w-200, h-120), (w-10, h-10), (0, 0, 0), -1)
            fusion = cv2.addWeighted(fusion, 0.85, legend, 0.15, 0)

            cv2.putText(fusion, "Detection:", (w-190, h-95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.circle(fusion, (w-170, h-70), 5, (255, 0, 255), -1)
            cv2.putText(fusion, "ANOMALY", (w-150, h-65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.circle(fusion, (w-170, h-40), 3, (0, 255, 255), -1)
            cv2.putText(fusion, "Possible", (w-150, h-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.circle(fusion, (w-170, h-15), 1, (0, 200, 0), -1)
            cv2.putText(fusion, "Ground", (w-150, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # Salva
            cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion)
            cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)

            # Comparação
            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comparison)

            print(f"  ✅ Salvo!")

            # Salva pontos e anomalias
            np.save(output_dir / f"points_{frame:03d}.npy", points)
            np.save(output_dir / f"anomalies_{frame:03d}.npy", anomalies)

# Volta ao centro
client.rotateToYawAsync(0, vehicle_name="Ego").join()

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v)
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("✅ FUSÃO COM DETECÇÃO COMPLETA!")
print("="*70)
print(f"\n📁 Resultados em: {output_dir.absolute()}")
print("\n🎯 LEGENDA:")
print("  • MAGENTA + Círculo amarelo = ANOMALIA DETECTADA (possível drone)")
print("  • AMARELO = Possível objeto")
print("  • VERDE = Chão normal")
print("\n💡 A detecção funciona identificando pontos com Z diferente do chão!")