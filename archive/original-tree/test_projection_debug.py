#!/usr/bin/env python3
"""
Debug da projeção 3D para 2D e tamanho das bounding boxes
"""

import cosysairsim as airsim
import numpy as np
import cv2
import sys

print("\n" + "="*70)
print("DEBUG: TESTE DE PROJEÇÃO E BOUNDING BOXES")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Prepara e decola apenas Ego e Drone3
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")
client.enableApiControl(True, "Drone3")
client.armDisarm(True, "Drone3")

print("\n🛫 Decolando...")
client.takeoffAsync(vehicle_name="Ego").join()
client.takeoffAsync(vehicle_name="Drone3").join()
import time
time.sleep(3)

# Posiciona Drone3 bem na frente
print("\n📍 Posicionando drones para teste...")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone3").join()  # 10m na frente
time.sleep(2)

# Parâmetros da câmera
FOV_H = 90
FOV_V = 60
image_width = 1280
image_height = 720

fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = image_height / (2 * np.tan(np.radians(FOV_V/2)))
cx = image_width / 2
cy = image_height / 2

print(f"\n📷 Parâmetros da câmera:")
print(f"   FOV: {FOV_H}° x {FOV_V}°")
print(f"   Resolução: {image_width}x{image_height}")
print(f"   fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

# Pega posições
ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
drone3_pose = client.simGetVehiclePose(vehicle_name="Drone3")

ego_pos = np.array([ego_pose.position.x_val, ego_pose.position.y_val, ego_pose.position.z_val])
drone3_pos = np.array([drone3_pose.position.x_val, drone3_pose.position.y_val, drone3_pose.position.z_val])

# Posição relativa
relative_pos = drone3_pos - ego_pos

print(f"\n📍 Posições:")
print(f"   Ego: {ego_pos}")
print(f"   Drone3: {drone3_pos}")
print(f"   Relativa: {relative_pos}")

# Captura imagem e depth
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
], vehicle_name="Ego")

# Processa imagem
img_bgr = None
if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Teste 1: Projeta um único ponto (centro do drone)
print("\n🔴 TESTE 1: Projeção do centro do drone")
print(f"   Centro 3D relativo: {relative_pos}")

# Projeta centro
if relative_pos[0] > 0:  # X positivo = frente
    u = fx * relative_pos[1] / relative_pos[0] + cx
    v = fy * (-relative_pos[2]) / relative_pos[0] + cy  # -Z porque Y da tela é invertido

    print(f"   Projeção 2D: u={u:.1f}, v={v:.1f}")

    if 0 <= u < image_width and 0 <= v < image_height:
        print("   ✅ Dentro da imagem!")
    else:
        print("   ❌ FORA da imagem!")

# Teste 2: Cria bounding box 3D pequena
print("\n🔴 TESTE 2: Bounding box 3D")
drone_size = {"width": 1.0, "height": 0.4, "depth": 1.0}  # metros

# 8 cantos da caixa
corners_3d = []
for dx in [-drone_size['width']/2, drone_size['width']/2]:
    for dy in [-drone_size['height']/2, drone_size['height']/2]:
        for dz in [-drone_size['depth']/2, drone_size['depth']/2]:
            corner = relative_pos + np.array([dz, dx, dy])  # Nota: pode precisar ajustar ordem
            corners_3d.append(corner)

print(f"   Tamanho do drone: {drone_size}")
print(f"   Cantos 3D (primeiro): {corners_3d[0]}")

# Projeta todos os cantos
points_2d = []
for i, corner in enumerate(corners_3d):
    if corner[0] > 0:  # X positivo
        u = fx * corner[1] / corner[0] + cx
        v = fy * (-corner[2]) / corner[0] + cy
        points_2d.append([u, v])
        print(f"   Canto {i}: 3D={corner} → 2D=({u:.1f}, {v:.1f})")

if len(points_2d) >= 4:
    points_2d = np.array(points_2d)
    x_min = max(0, int(points_2d[:, 0].min()))
    x_max = min(image_width-1, int(points_2d[:, 0].max()))
    y_min = max(0, int(points_2d[:, 1].min()))
    y_max = min(image_height-1, int(points_2d[:, 1].max()))

    print(f"\n   Bounding Box 2D:")
    print(f"   x_min={x_min}, x_max={x_max}")
    print(f"   y_min={y_min}, y_max={y_max}")
    print(f"   Largura: {x_max-x_min}px")
    print(f"   Altura: {y_max-y_min}px")

    if x_max - x_min > image_width * 0.8:
        print("   ⚠️ PROBLEMA: BBox muito larga!")
    if y_max - y_min > image_height * 0.8:
        print("   ⚠️ PROBLEMA: BBox muito alta!")

# Visualiza
if img_bgr is not None:
    vis = img_bgr.copy()

    # Desenha centro projetado
    if relative_pos[0] > 0:
        u = int(fx * relative_pos[1] / relative_pos[0] + cx)
        v = int(fy * (-relative_pos[2]) / relative_pos[0] + cy)
        cv2.circle(vis, (u, v), 10, (0, 255, 0), -1)
        cv2.putText(vis, "Centro", (u+15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Desenha bbox
    if len(points_2d) >= 4:
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(vis, f"BBox: {x_max-x_min}x{y_max-y_min}",
                   (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Desenha todos os pontos projetados
    for p2d in points_2d:
        cv2.circle(vis, (int(p2d[0]), int(p2d[1])), 3, (0, 0, 255), -1)

    cv2.imshow("Debug Projection", vis)
    cv2.imwrite("debug_projection.png", vis)
    print("\n✅ Visualização salva em debug_projection.png")
    print("Pressione qualquer tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Pousa
client.landAsync(vehicle_name="Ego").join()
client.landAsync(vehicle_name="Drone3").join()
client.armDisarm(False, "Ego")
client.armDisarm(False, "Drone3")
client.enableApiControl(False, "Ego")
client.enableApiControl(False, "Drone3")

print("\n✅ Debug concluído!")