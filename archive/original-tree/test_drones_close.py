#!/usr/bin/env python3
"""
Teste com drones bem próximos
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 TESTE COM DRONES PRÓXIMOS")
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
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

print("\n📍 Posicionando drones BEM PRÓXIMOS...")

# Ego como observador
client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()
print("   Ego: 10m altura")

# Outros drones BEM próximos e na frente
if "Drone3" in vehicles:
    client.moveToPositionAsync(5, -1, -10, 5, vehicle_name="Drone3").join()
    print("   Drone3: 5m frente (BEM PERTO)")

if "Drone4" in vehicles:
    client.moveToPositionAsync(5, 1, -10, 5, vehicle_name="Drone4").join()
    print("   Drone4: 5m frente (BEM PERTO)")

if "Intruder1" in vehicles:
    client.moveToPositionAsync(7, 0, -8, 5, vehicle_name="Intruder1").join()
    print("   Intruder1: 7m frente, 2m ACIMA")

time.sleep(3)

print("\n📸 Capturando com drones próximos...")

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

    print(f"\n📊 ANÁLISE DETALHADA:")
    print(f"  Total pontos: {len(points):,}")

    # Analisa região onde drones deveriam estar (5-7m frente)
    front_close = points[(points[:, 0] > 4) & (points[:, 0] < 8)]
    print(f"  Pontos 4-8m frente: {len(front_close)}")

    if len(front_close) > 0:
        z_values = front_close[:, 2]
        print(f"    Z range nesta região: {z_values.min():.2f} a {z_values.max():.2f}m")

        # Pontos não-chão nesta região
        non_ground = front_close[(z_values < 5) | (z_values < 0)]
        print(f"    Pontos não-chão: {len(non_ground)}")

        if len(non_ground) > 0:
            print("    ✅ POSSÍVEIS DRONES DETECTADOS!")
        else:
            print("    ❌ Apenas chão detectado")

    # Cria fusão
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Parâmetros câmera
        fx = w / (2 * np.tan(np.radians(45)))
        fy = h / (2 * np.tan(np.radians(30)))
        cx = w / 2
        cy = h / 2

        # Projeta pontos frontais
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

            # Desenha com destaque para região dos drones
            for i in range(len(u)):
                if 4 < x_vals[i] < 8:  # Região dos drones
                    if z_vals[i] < 5:  # Não é chão
                        color = (255, 0, 255)  # Magenta = possível drone
                        size = 5
                    else:
                        color = (255, 255, 0)  # Amarelo = região de interesse
                        size = 2
                else:
                    color = (0, 255, 0)  # Verde = resto
                    size = 1

                cv2.circle(fusion, (u[i], v[i]), size, color, -1)

            # Marca região esperada dos drones
            cv2.rectangle(fusion, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), (0, 255, 255), 2)
            cv2.putText(fusion, "Drones should be here", (int(w*0.3), int(h*0.3)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Info
            cv2.putText(fusion, f"Drones at 5-7m", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(fusion, f"Points in region: {len(front_close)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Salva
            output_dir = Path("drones_close_test")
            output_dir.mkdir(exist_ok=True)

            cv2.imwrite(str(output_dir / "fusion.png"), fusion)
            cv2.imwrite(str(output_dir / "original.png"), img_bgr)

            # Comparação
            comp = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "comparison.png"), comp)

            print(f"\n💾 Salvo em {output_dir}")

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
print("📌 CONCLUSÃO:")
if 'non_ground' in locals() and len(non_ground) > 0:
    print("✅ Drones detectados pelo LiDAR!")
else:
    print("❌ Drones NÃO detectados pelo LiDAR")
    print("\nPROVÁVEL CAUSA:")
    print("• Drones no Blocks environment não têm mesh de colisão para LiDAR")
    print("• O LiDAR só detecta o terreno e objetos estáticos")
    print("\nSOLUÇÃO:")
    print("• Use um environment diferente com objetos que tenham colisão")
    print("• Ou adicione objetos estáticos na cena para testar")
print("="*70)