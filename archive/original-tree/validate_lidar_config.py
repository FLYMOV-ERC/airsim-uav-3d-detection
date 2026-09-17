#!/usr/bin/env python3
"""
Validação da nova configuração do LiDAR - FOV expandido
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("\n" + "="*70)
print("✅ VALIDAÇÃO DO LIDAR COM NOVO SETTINGS.JSON")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"\n📡 Veículos conectados: {vehicles}")

# Prepara e decola todos
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

print("\n🛫 Decolando todos os drones...")
takeoff_tasks = []
for v in vehicles:
    task = client.takeoffAsync(vehicle_name=v)
    takeoff_tasks.append(task)

for task in takeoff_tasks:
    task.join()

print("✅ Todos no ar! Aguardando estabilização...")
time.sleep(3)

# Posiciona drones em diferentes alturas para testar FOV
print("\n📍 Posicionando drones em alturas estratégicas:")
print("   Objetivo: Testar detecção em toda altura da imagem")

# Ego na altura média
client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()
print("   Ego: 15m altura (referência)")

# Coloca drones em diferentes posições verticais
test_positions = [
    ("Drone3", 12, 0, -5, "ALTO: 10m ACIMA do Ego"),
    ("Drone4", 15, -3, -15, "MEIO: mesma altura"),
    ("Intruder1", 18, 3, -25, "BAIXO: 10m ABAIXO")
]

for name, x, y, z, desc in test_positions:
    if name in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
        print(f"   {name}: {desc} (x={x}m, z={z})")

time.sleep(2)

# Diretório de saída
output_dir = Path("validate_lidar")
output_dir.mkdir(exist_ok=True)

# Parâmetros da câmera
fx = 1280 / (2 * np.tan(np.radians(45)))
fy = 720 / (2 * np.tan(np.radians(30)))
cx = 640
cy = 360
camera_pitch = -15
pitch_rad = np.radians(camera_pitch)

print("\n📸 Capturando dados com o novo FOV do LiDAR...")

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

    print(f"\n📊 ESTATÍSTICAS DO LIDAR:")
    print(f"   Total de pontos capturados: {len(points)}")

    h, w = img_bgr.shape[:2]
    fusion = img_bgr.copy()

    # Filtra pontos frontais
    front_mask = points[:, 0] > 0.5
    front_points = points[front_mask]

    if len(front_points) > 0:
        X = front_points[:, 0]
        Y = front_points[:, 1]
        Z = front_points[:, 2]

        # Análise do alcance vertical
        z_min = Z.min()
        z_max = Z.max()
        z_range = z_max - z_min
        z_mean = Z.mean()
        z_std = Z.std()

        print(f"   Pontos frontais: {len(front_points)}")
        print(f"   Z mínimo: {z_min:.2f}m")
        print(f"   Z máximo: {z_max:.2f}m")
        print(f"   Alcance vertical total: {z_range:.2f}m")
        print(f"   Z médio: {z_mean:.2f}m ± {z_std:.2f}")

        # Detecta anomalias (drones)
        anomalies_mask = np.abs(Z - z_mean) > 2 * z_std

        print(f"\n🎯 DETECÇÕES:")
        print(f"   Anomalias detectadas (possíveis drones): {anomalies_mask.sum()}")

        # Aplica correção de pitch e projeta
        X_rot = X * np.cos(pitch_rad) - Z * np.sin(pitch_rad)
        Z_rot = X * np.sin(pitch_rad) + Z * np.cos(pitch_rad)

        u = (fx * Y / X_rot + cx).astype(int)
        v = (fy * Z_rot / X_rot + cy).astype(int)

        # Filtra pontos válidos na imagem
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        u_valid = u[valid]
        v_valid = v[valid]
        x_valid = X[valid]
        z_valid = Z[valid]
        anomalies_valid = anomalies_mask[valid]

        # Análise da distribuição vertical na imagem
        if len(v_valid) > 0:
            v_min = v_valid.min()
            v_max = v_valid.max()
            v_coverage = (v_max - v_min) / h * 100

            print(f"\n📐 PROJEÇÃO NA IMAGEM:")
            print(f"   Pixel Y mínimo: {v_min} (topo=0)")
            print(f"   Pixel Y máximo: {v_max} (base={h})")
            print(f"   Cobertura vertical: {v_coverage:.1f}% da imagem")

        # Conta detecções por região
        horizon_line = h // 2
        upper_third = h // 3
        lower_third = 2 * h // 3

        detections_top = 0     # Terço superior
        detections_mid = 0     # Terço médio
        detections_bot = 0     # Terço inferior

        # Desenha pontos e conta detecções
        for i in range(len(u_valid)):
            if anomalies_valid[i]:
                # Drone detectado!
                if v_valid[i] < upper_third:
                    # Terço superior
                    color = (255, 0, 255)  # Magenta
                    detections_top += 1
                    label = "TOP"
                elif v_valid[i] < lower_third:
                    # Terço médio
                    color = (255, 255, 0)  # Ciano
                    detections_mid += 1
                    label = "MID"
                else:
                    # Terço inferior
                    color = (0, 255, 255)  # Amarelo
                    detections_bot += 1
                    label = "BOT"

                # Desenha círculo grande para destacar
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 12, color, 2)
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 6, color, -1)
                cv2.putText(fusion, label, (u_valid[i]+15, v_valid[i]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Pontos normais (chão/ambiente)
                depth = min(1.0, x_valid[i] / 50)
                color = (0, int(255*(1-depth)), 0)
                cv2.circle(fusion, (u_valid[i], v_valid[i]), 1, color, -1)

        print(f"\n🎯 DETECÇÕES POR REGIÃO DA IMAGEM:")
        print(f"   Terço SUPERIOR: {detections_top} drones")
        print(f"   Terço MÉDIO: {detections_mid} drones")
        print(f"   Terço INFERIOR: {detections_bot} drones")
        print(f"   TOTAL: {detections_top + detections_mid + detections_bot} drones")

        # Desenha linhas de referência
        cv2.line(fusion, (0, upper_third), (w, upper_third), (100, 100, 100), 1)
        cv2.line(fusion, (0, horizon_line), (w, horizon_line), (255, 255, 255), 2)
        cv2.line(fusion, (0, lower_third), (w, lower_third), (100, 100, 100), 1)

        # Adiciona informações na imagem
        cv2.putText(fusion, "VALIDACAO FOV EXPANDIDO", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        info_text = f"Detectados: TOP:{detections_top} MID:{detections_mid} BOT:{detections_bot}"
        cv2.putText(fusion, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(fusion, f"Z range: {z_range:.1f}m | Coverage: {v_coverage:.0f}%", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Labels das regiões
        cv2.putText(fusion, "TOP", (w-50, upper_third-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(fusion, "MID", (w-50, horizon_line-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(fusion, "BOT", (w-50, lower_third-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Salva imagens
        cv2.imwrite(str(output_dir / "fusion_validated.png"), fusion)
        cv2.imwrite(str(output_dir / "original.png"), img_bgr)

        # Comparação lado a lado
        comparison = np.hstack([img_bgr, fusion])
        cv2.imwrite(str(output_dir / "comparison.png"), comparison)

        print(f"\n💾 Imagens salvas em: {output_dir.absolute()}")

        # Análise de sucesso
        print("\n" + "="*50)
        if detections_top > 0:
            print("✅ SUCESSO! LiDAR detectando drones no TOPO da imagem!")
            print("   O FOV expandido (70° para cima) está funcionando!")
        else:
            print("⚠️  Nenhuma detecção no topo - verificar configuração")

        if v_coverage > 70:
            print("✅ Excelente cobertura vertical (>70% da imagem)")
        elif v_coverage > 50:
            print("✅ Boa cobertura vertical (>50% da imagem)")
        else:
            print("⚠️  Cobertura vertical limitada (<50% da imagem)")

# Pousa todos os drones
print("\n🛬 Pousando todos os drones...")
for v in vehicles:
    try:
        client.moveToPositionAsync(0, 0, -3, 3, vehicle_name=v).join()
    except:
        pass

for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("🏁 VALIDAÇÃO COMPLETA!")
print("="*70)
print("\n📋 RESUMO:")
print("   Se detectou drones no TOPO da imagem → FOV melhorado funcionando!")
print("   Se cobertura > 70% → Excelente alcance vertical")
print("   Verifique as imagens em: validate_lidar/")
print("="*70)