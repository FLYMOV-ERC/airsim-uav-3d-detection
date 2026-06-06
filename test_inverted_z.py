#!/usr/bin/env python3
"""
Teste se Z está invertido
"""

import numpy as np
import cv2
from pathlib import Path

# Carrega dados existentes
dataset_dir = Path("dataset_FINAL")
points = np.load(dataset_dir / "lidar_points" / "frame_0000.npy")
img = cv2.imread(str(dataset_dir / "images_rgb" / "frame_0000.png"))

print("TESTE DE INVERSÃO DE Z")
print("="*60)

X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

print(f"Z original: {Z.min():.2f} to {Z.max():.2f}")

# Testa 3 hipóteses
hypotheses = [
    ("Original", Z),
    ("Invertido (-Z)", -Z),
    ("Offset (-Z-12)", -Z - 12)
]

# Parâmetros de projeção
FOV_H = 90
FOV_V = 60
fx = 640 / np.tan(np.radians(FOV_H/2))
fy = 360 / np.tan(np.radians(FOV_V/2))
cx = 640
cy = 360

h, w = img.shape[:2]

for name, Z_test in hypotheses:
    fusion = img.copy()

    # Filtra pontos frontais
    front = (X > 0.5) & (X < 50)
    X_f = X[front]
    Y_f = Y[front]
    Z_f = Z_test[front]

    # Projeta
    u = (fx * Y_f / X_f + cx).astype(int)
    v = (fy * Z_f / X_f + cy).astype(int)

    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Desenha
    for i in range(len(u)):
        if valid[i]:
            # Cor baseada em Z
            if Z_f[i] < -8:  # Deveria ser chão
                color = (0, 255, 0)  # Verde
            elif -2 < Z_f[i] < 2:  # Nível do sensor
                color = (255, 0, 255)  # Magenta
            else:
                color = (100, 100, 100)  # Cinza

            cv2.circle(fusion, (u[i], v[i]), 2, color, -1)

    # Adiciona título
    cv2.putText(fusion, f"Hipotese: {name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(fusion, f"Z range: {Z_f.min():.1f} to {Z_f.max():.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Salva
    cv2.imwrite(f"test_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", fusion)
    print(f"\n{name}:")
    print(f"  Z range após transformação: {Z_f.min():.2f} to {Z_f.max():.2f}")
    print(f"  Imagem salva: test_{name.replace(' ', '_')}.png")

print("\n✅ Teste concluído! Verifique as 3 imagens geradas.")