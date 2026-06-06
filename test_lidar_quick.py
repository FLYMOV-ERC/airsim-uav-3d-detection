#!/usr/bin/env python3
"""
Teste rápido do LiDAR melhorado - Verifica detecção em toda a imagem
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("⚡ TESTE RÁPIDO - LIDAR COM FOV EXPANDIDO")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n✅ Veículos: {vehicles}")

# Prepara e decola
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

print("\n🛫 Decolando...")
for v in vehicles:
    client.takeoffAsync(vehicle_name=v)
time.sleep(5)

# Posiciona drones estrategicamente
print("\n📍 Posicionando drones (alto, médio, baixo)...")

client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

positions = [
    ("Drone3", 12, 0, -5, "10m ACIMA (deve aparecer no topo)"),
    ("Drone4", 15, 3, -15, "mesma altura (meio da imagem)"),
    ("Intruder1", 18, -3, -25, "10m ABAIXO (parte inferior)")
]

for name, x, y, z, desc in positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"   {name}: {desc}")

time.sleep(2)

output_dir = Path("test_lidar_quick")
output_dir.mkdir(exist_ok=True)

# Parâmetros da câmera
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360

print("\n📊 TESTE 1: LiDAR padrão (single scan)")
print("-"*40)

# Captura normal (sem extensão de FOV)
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
    fusion_normal = img_bgr.copy()

    # Processa pontos
    front_points = points[points[:, 0] > 0.5]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        print(f"   Pontos capturados: {len(front_points)}")
        print(f"   Z range: {Z.min():.2f} a {Z.max():.2f}m")

        # Detecta e projeta
        z_mean = Z.mean()
        anomalies = np.abs(Z - z_mean) > 2

        pitch_rad = np.radians(-15)
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        det_high = 0
        det_low = 0

        for i in np.where(valid)[0]:
            if anomalies[i]:
                if v[i] < h//2:
                    det_high += 1
                    cv2.circle(fusion_normal, (u[i], v[i]), 8, (255, 0, 255), 2)
                else:
                    det_low += 1
                    cv2.circle(fusion_normal, (u[i], v[i]), 8, (0, 255, 255), 2)

        print(f"   Detecções: {det_high} alta, {det_low} baixa")

        cv2.line(fusion_normal, (0, h//2), (w, h//2), (255, 255, 255), 1)
        cv2.putText(fusion_normal, "NORMAL FOV", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(fusion_normal, f"High:{det_high} Low:{det_low}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imwrite(str(output_dir / "fusion_normal.png"), fusion_normal)

print("\n📊 TESTE 2: LiDAR expandido (múltiplas varreduras)")
print("-"*40)

# Captura com FOV expandido
all_points = []
PITCH_OFFSETS = [-0.3, -0.15, 0, 0.15, 0.3]

for idx, pitch_offset in enumerate(PITCH_OFFSETS):
    client.rotateByYawPitchRollAsync(0, pitch_offset, 0, vehicle_name="Ego").join()
    time.sleep(0.2)

    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    if lidar_data and len(lidar_data.point_cloud) > 3:
        pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Compensa rotação
        cos_p = np.cos(-pitch_offset)
        sin_p = np.sin(-pitch_offset)
        X_comp = pts[:, 0] * cos_p - pts[:, 2] * sin_p
        Z_comp = pts[:, 0] * sin_p + pts[:, 2] * cos_p

        pts_comp = np.column_stack([X_comp, pts[:, 1], Z_comp])
        all_points.append(pts_comp)
        print(f"   Varredura {idx+1}: {len(pts)} pontos")

# Volta orientação normal
client.rotateByYawPitchRollAsync(0, 0, 0, vehicle_name="Ego").join()

if all_points and img_bgr is not None:
    combined = np.vstack(all_points)
    # Remove duplicatas
    rounded = np.round(combined, decimals=1)
    unique_points = np.unique(rounded, axis=0)

    print(f"   Total combinado: {len(unique_points)} pontos únicos")

    h, w = img_bgr.shape[:2]
    fusion_extended = img_bgr.copy()

    front_points = unique_points[unique_points[:, 0] > 0.5]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        print(f"   Z range expandido: {Z.min():.2f} a {Z.max():.2f}m")

        # Detecta e projeta
        z_mean = Z.mean()
        anomalies = np.abs(Z - z_mean) > 2

        pitch_rad = np.radians(-15)
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        det_high = 0
        det_low = 0

        for i in np.where(valid)[0]:
            if anomalies[i]:
                if v[i] < h//2:
                    det_high += 1
                    cv2.circle(fusion_extended, (u[i], v[i]), 8, (255, 0, 255), 2)
                    cv2.putText(fusion_extended, "HIGH", (u[i]+10, v[i]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                else:
                    det_low += 1
                    cv2.circle(fusion_extended, (u[i], v[i]), 8, (0, 255, 255), 2)
                    cv2.putText(fusion_extended, "LOW", (u[i]+10, v[i]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                cv2.circle(fusion_extended, (u[i], v[i]), 1, (0, 200, 0), -1)

        print(f"   Detecções: {det_high} alta, {det_low} baixa")

        cv2.line(fusion_extended, (0, h//2), (w, h//2), (255, 255, 255), 1)
        cv2.putText(fusion_extended, "EXTENDED FOV", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(fusion_extended, f"High:{det_high} Low:{det_low}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imwrite(str(output_dir / "fusion_extended.png"), fusion_extended)
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)

        # Comparação lado a lado
        comparison = np.hstack([fusion_normal, fusion_extended])
        cv2.putText(comparison, "NORMAL vs EXTENDED", (w-200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imwrite(str(output_dir / "comparison.png"), comparison)

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
print(f"\n📁 Resultados em: {output_dir.absolute()}")
print("\n🔍 ANÁLISE:")
print("   • fusion_normal.png: LiDAR padrão (FOV limitado)")
print("   • fusion_extended.png: LiDAR expandido (múltiplas varreduras)")
print("   • comparison.png: Comparação lado a lado")
print("\n💡 ESPERADO:")
print("   • FOV expandido deve detectar mais drones")
print("   • Especialmente drones ACIMA do horizonte")
print("   • Maior cobertura vertical da imagem")
print("="*70)