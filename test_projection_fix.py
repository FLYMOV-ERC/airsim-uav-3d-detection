#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path

dataset_dir = Path("dataset_lidar_sync")
points = np.load(dataset_dir / "lidar_points" / "frame_0000.npy")
img_rgb = cv2.imread(str(dataset_dir / "images_rgb" / "frame_0000.png"))

print("ANÁLISE DO PROBLEMA")
print("="*50)

# Analisa pontos
X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

print(f"Total pontos: {len(points)}")
print(f"Z range: {Z.min():.2f} to {Z.max():.2f}")
print(f"Pontos com Z > 10 (céu?): {np.sum(Z > 10)}")
print(f"Pontos com Z entre -5 e 5: {np.sum((Z > -5) & (Z < 5))}")
print(f"Pontos com Z < -5 (chão?): {np.sum(Z < -5)}")

# O problema: pontos das MONTANHAS distantes estão sendo interpretados como céu
distances = np.sqrt(X**2 + Y**2 + Z**2)
print(f"\nDistâncias: min={distances.min():.1f}m, max={distances.max():.1f}m")

# Pontos distantes (montanhas)
far_mask = distances > 50
print(f"Pontos > 50m (montanhas): {np.sum(far_mask)}")
print(f"Z médio dos pontos distantes: {Z[far_mask].mean():.2f}" if np.any(far_mask) else "N/A")

# Solução: filtrar por distância
MAX_RANGE = 40
near_mask = distances < MAX_RANGE
near_points = points[near_mask]
print(f"\nPontos < {MAX_RANGE}m: {len(near_points)}")

if len(near_points) > 0:
    Z_near = near_points[:, 2]
    print(f"Z range (próximos): {Z_near.min():.2f} to {Z_near.max():.2f}")
    print(f"Pontos próximos no chão (Z<-2): {np.sum(Z_near < -2)}")
