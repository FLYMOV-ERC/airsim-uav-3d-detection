#!/usr/bin/env python3
"""
Teste de ajuste do FOV vertical do LiDAR para capturar drones mais altos
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("🔍 TESTE FOV VERTICAL DO LIDAR")
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
print("\n🛫 Decolando...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

print("\n📍 Posicionando drones em diferentes alturas...")

# Ego no meio
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

# Drones em diferentes alturas relativas para testar detecção
positions = [
    ("Drone3", 10, 0, -20, "5m abaixo"),     # Abaixo do Ego
    ("Drone4", 15, 0, -15, "mesma altura"),  # Mesma altura
    ("Intruder1", 20, 0, -10, "5m acima")    # Acima do Ego
]

for name, x, y, z, desc in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"   {name}: {x}m frente, {desc} (z={z})")

time.sleep(2)

output_dir = Path("test_lidar_fov")
output_dir.mkdir(exist_ok=True)

# Parâmetros da câmera
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360

# Testa diferentes configurações de pitch
pitch_angles = [-30, -20, -15, -10, 0]

for pitch_test in pitch_angles:
    print(f"\n🔄 Testando com pitch da câmera: {pitch_test}°")

    pitch_rad = np.radians(pitch_test)

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

        h, w = img_bgr.shape[:2]
        fusion = img_bgr.copy()

        # Filtra pontos frontais
        front_mask = points[:, 0] > 0.5
        front_points = points[front_mask]

        if len(front_points) > 0:
            X = front_points[:, 0]
            Y = front_points[:, 1]
            Z = front_points[:, 2]

            # Análise de distribuição vertical
            z_min = Z.min()
            z_max = Z.max()
            z_range = z_max - z_min

            print(f"   📊 Alcance vertical do LiDAR:")
            print(f"      Z mín: {z_min:.2f}m | Z máx: {z_max:.2f}m | Range: {z_range:.2f}m")

            # Detecta anomalias ANTES de projetar
            z_mean = Z.mean()
            z_std = Z.std()
            anomalies_mask = np.abs(Z - z_mean) > 2

            # Aplica correção de pitch
            X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
            Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

            # Projeta
            u = (fx * Y / X_rot + cx).astype(int)
            v = (fy * Z_rot / X_rot + cy).astype(int)

            # Filtra válidos
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            u_valid = u[valid]
            v_valid = v[valid]
            x_valid = X[valid]
            z_valid = Z[valid]
            anomalies_valid = anomalies_mask[valid]

            # Estatísticas de projeção
            if len(v_valid) > 0:
                v_min = v_valid.min()
                v_max = v_valid.max()
                print(f"   📐 Projeção na imagem:")
                print(f"      Pixels Y: {v_min} a {v_max} (altura img: {h})")
                print(f"      Cobertura: {((v_max-v_min)/h)*100:.1f}% da altura")

            detections_low = 0    # Parte inferior da imagem (y > h/2)
            detections_high = 0   # Parte superior da imagem (y < h/2)

            # Desenha pontos
            for i in range(len(u_valid)):
                if anomalies_valid[i]:
                    # Drone detectado
                    if v_valid[i] < h/2:
                        # Parte superior
                        color = (255, 0, 255)  # Magenta
                        detections_high += 1
                        label = "HIGH"
                    else:
                        # Parte inferior
                        color = (255, 255, 0)  # Ciano
                        detections_low += 1
                        label = "LOW"

                    cv2.circle(fusion, (u_valid[i], v_valid[i]), 8, color, 2)
                    cv2.putText(fusion, label, (u_valid[i]+10, v_valid[i]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Normal
                    depth = min(1.0, x_valid[i] / 50)
                    color = (0, int(255*(1-depth)), 0)
                    cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, color, -1)

            print(f"   🎯 Detecções:")
            print(f"      Parte superior: {detections_high}")
            print(f"      Parte inferior: {detections_low}")
            print(f"      Total: {detections_high + detections_low}")

            # Linha do horizonte
            cv2.line(fusion, (0, h//2), (w, h//2), (0, 255, 255), 1)
            cv2.putText(fusion, "HORIZON", (10, h//2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Info
            cv2.putText(fusion, f"Pitch: {pitch_test} deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(fusion, f"High: {detections_high} | Low: {detections_low}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, f"Z range: {z_min:.1f} to {z_max:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Salva
            cv2.imwrite(str(output_dir / f"fusion_pitch_{pitch_test}.png"), fusion)
            cv2.imwrite(str(output_dir / f"original_pitch_{pitch_test}.png"), img_bgr)

            # Comparação
            comp = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / f"comparison_pitch_{pitch_test}.png"), comp)

print("\n" + "="*50)
print("📡 TESTANDO AJUSTE DO ÂNGULO VERTICAL DO LIDAR...")
print("="*50)

# Agora vamos ajustar o ângulo de varredura do LiDAR
print("\n🔧 Ajustando parâmetros do LiDAR para maior FOV vertical...")

# Rotaciona o Ego para olhar mais para cima
client.rotateByYawPitchRollAsync(0, -0.3, 0, vehicle_name="Ego").join()  # Pitch negativo = olhar para cima
time.sleep(1)

print("📸 Capturando com LiDAR ajustado...")

# Captura com ajuste
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")

img_bgr = None
if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    h, w = img_bgr.shape[:2]
    fusion = img_bgr.copy()

    # Usa pitch ajustado para compensar inclinação do drone
    adjusted_pitch = -15 - 17  # Pitch original + compensação
    pitch_rad = np.radians(adjusted_pitch)

    front_points = points[points[:, 0] > 0.5]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        print(f"\n📊 Com LiDAR ajustado:")
        print(f"   Z range: {Z.min():.2f} a {Z.max():.2f}")

        # Detecta anomalias
        z_mean = Z.mean()
        anomalies_mask = np.abs(Z - z_mean) > 2

        # Projeta com pitch ajustado
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u_valid = u[valid]
        v_valid = v[valid]
        anomalies_valid = anomalies_mask[valid]

        high_detections = 0
        low_detections = 0

        for i in range(len(u_valid)):
            if anomalies_valid[i]:
                if v_valid[i] < h/2:
                    color = (255, 0, 255)
                    high_detections += 1
                else:
                    color = (255, 255, 0)
                    low_detections += 1
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 10, color, 2)
            else:
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, (0, 200, 0), -1)

        cv2.line(fusion, (0, h//2), (w, h//2), (0, 255, 255), 1)
        cv2.putText(fusion, "LIDAR ADJUSTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(fusion, f"High: {high_detections} | Low: {low_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imwrite(str(output_dir / "fusion_adjusted.png"), fusion)
        cv2.imwrite(str(output_dir / "comparison_adjusted.png"), np.hstack([img_bgr, fusion]))

        print(f"   Detecções alta: {high_detections}")
        print(f"   Detecções baixa: {low_detections}")

# Volta orientação normal
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
print("✅ TESTE COMPLETO!")
print("="*70)
print(f"\n📁 Imagens salvas em: {output_dir.absolute()}")
print("\n🔍 ANÁLISE:")
print("   • Compare as imagens com diferentes valores de pitch")
print("   • Verifique quantos drones são detectados acima/abaixo da linha do horizonte")
print("   • A imagem 'fusion_adjusted' mostra o resultado com LiDAR otimizado")
print("="*70)