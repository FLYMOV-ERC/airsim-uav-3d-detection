#!/usr/bin/env python3
"""
Debug - Analisa distribuição dos pontos LiDAR
"""

import numpy as np
import cv2
from pathlib import Path
import json

# Carrega um frame
frame_idx = 0
dataset_dir = Path("dataset_lidar_sync")

# Carrega dados
points = np.load(dataset_dir / "lidar_points" / f"frame_{frame_idx:04d}.npy")
img_rgb = cv2.imread(str(dataset_dir / "images_rgb" / f"frame_{frame_idx:04d}.png"))

with open(dataset_dir / "metadata" / f"frame_{frame_idx:04d}.json", 'r') as f:
    metadata = json.load(f)

print("="*60)
print(f"ANÁLISE DO FRAME {frame_idx:04d}")
print("="*60)

print(f"\nTotal de pontos: {len(points)}")
print(f"Shape: {points.shape}")

# Analisa distribuição X, Y, Z
X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

print(f"\nDistribuição dos pontos:")
print(f"X: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
print(f"Y: min={Y.min():.2f}, max={Y.max():.2f}, mean={Y.mean():.2f}")
print(f"Z: min={Z.min():.2f}, max={Z.max():.2f}, mean={Z.mean():.2f}")

# Filtra pontos frontais
front_mask = X > 0.5
front_points = points[front_mask]
print(f"\nPontos frontais (X > 0.5): {len(front_points)}")

if len(front_points) > 0:
    X_front = front_points[:, 0]
    Y_front = front_points[:, 1]
    Z_front = front_points[:, 2]

    print(f"\nDistribuição frontal:")
    print(f"X: min={X_front.min():.2f}, max={X_front.max():.2f}")
    print(f"Y: min={Y_front.min():.2f}, max={Y_front.max():.2f}")
    print(f"Z: min={Z_front.min():.2f}, max={Z_front.max():.2f}")

    # Analisa pontos por altura
    print(f"\nAnálise por altura (Z):")
    print(f"Pontos com Z < -5 (abaixo): {np.sum(Z_front < -5)}")
    print(f"Pontos com -5 <= Z <= 5 (nível): {np.sum((Z_front >= -5) & (Z_front <= 5))}")
    print(f"Pontos com Z > 5 (acima): {np.sum(Z_front > 5)}")

    # Analisa distâncias
    distances = np.sqrt(X_front**2 + Y_front**2 + Z_front**2)
    print(f"\nDistâncias:")
    print(f"Min: {distances.min():.2f}m")
    print(f"Max: {distances.max():.2f}m")
    print(f"Média: {distances.mean():.2f}m")

    # Verifica pontos suspeitos (muito altos ou no ar)
    suspicious = Z_front > 10  # Pontos 10m acima
    if np.any(suspicious):
        susp_points = front_points[suspicious]
        print(f"\n⚠️ PONTOS SUSPEITOS (Z > 10):")
        print(f"Quantidade: {len(susp_points)}")
        for i, pt in enumerate(susp_points[:5]):  # Mostra até 5
            dist = np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)
            print(f"  Ponto {i}: X={pt[0]:.2f}, Y={pt[1]:.2f}, Z={pt[2]:.2f}, dist={dist:.2f}m")

# Visualização 3D simples dos pontos
fig = np.zeros((720, 1280, 3), dtype=np.uint8)

# Vista lateral (X-Z)
lateral = np.zeros((360, 640, 3), dtype=np.uint8)
if len(front_points) > 0:
    # Normaliza para pixels
    x_norm = ((X_front / 100) * 320 + 320).astype(int)  # -100 to 100 -> 0 to 640
    z_norm = ((-Z_front / 50 + 1) * 180).astype(int)   # -50 to 50 -> 0 to 360

    # Clipa valores
    x_norm = np.clip(x_norm, 0, 639)
    z_norm = np.clip(z_norm, 0, 359)

    for i in range(len(x_norm)):
        if Z_front[i] > 10:  # Suspeitos em vermelho
            color = (0, 0, 255)
        elif Z_front[i] < -5:  # Abaixo em verde
            color = (0, 255, 0)
        else:  # Normal em branco
            color = (255, 255, 255)

        cv2.circle(lateral, (x_norm[i], z_norm[i]), 1, color, -1)

    # Linha do horizonte
    cv2.line(lateral, (0, 180), (640, 180), (0, 255, 255), 1)
    cv2.putText(lateral, "Vista Lateral (X-Z)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(lateral, "Vermelho=Suspeito, Verde=Chao", (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Vista superior (X-Y)
superior = np.zeros((360, 640, 3), dtype=np.uint8)
if len(front_points) > 0:
    x_norm = ((X_front / 100) * 320 + 320).astype(int)
    y_norm = ((Y_front / 100 + 1) * 180).astype(int)

    x_norm = np.clip(x_norm, 0, 639)
    y_norm = np.clip(y_norm, 0, 359)

    for i in range(len(x_norm)):
        if Z_front[i] > 10:  # Suspeitos
            color = (0, 0, 255)
        else:
            dist = distances[i] if len(distances) > i else 0
            intensity = int(255 * (1 - min(1, dist/100)))
            color = (intensity, intensity, intensity)

        cv2.circle(superior, (x_norm[i], y_norm[i]), 1, color, -1)

    # Centro (posição do drone)
    cv2.circle(superior, (320, 180), 5, (0, 255, 0), -1)
    cv2.putText(superior, "Vista Superior (X-Y)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Monta visualização
fig[0:360, 0:640] = lateral
fig[0:360, 640:1280] = superior

# Info geral
info_area = np.zeros((360, 1280, 3), dtype=np.uint8)
y_offset = 30
cv2.putText(info_area, f"Total points: {len(points)}", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
y_offset += 30
cv2.putText(info_area, f"Front points: {len(front_points)}", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
y_offset += 30
cv2.putText(info_area, f"Suspicious (Z>10): {np.sum(Z_front > 10) if len(front_points) > 0 else 0}", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

fig[360:720, :] = info_area

cv2.imshow("LiDAR Analysis", fig)
cv2.imwrite("lidar_analysis.png", fig)
print("\n✅ Análise salva em lidar_analysis.png")
print("Pressione qualquer tecla para continuar...")
cv2.waitKey(0)
cv2.destroyAllWindows()