#!/usr/bin/env python3
"""
Dataset otimizado para ambientes pesados (Mountains/City)
Com delays e tratamento de erros
"""

import airsim
import time
import json
import math
from pathlib import Path

print("🏙️ DATASET PARA MOUNTAINS/CITY")
print("="*60)
print("⚠️  Usando delays maiores para ambientes pesados")
print("="*60)

# Configuração
output_dir = Path("dataset_cidade")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)

# Conecta com timeout maior
print("\n📡 Conectando...")
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451, timeout_value=30)
client.confirmConnection()

# Lista veículos
vehicles = client.listVehicles()
print(f"✅ Conectado! Veículos: {vehicles}")

# IMPORTANTE: Delays maiores para ambientes pesados
DELAY_AFTER_COMMAND = 2  # segundos
DELAY_AFTER_MOVEMENT = 3  # segundos

# Identifica observador e alvos
if "Ego" in vehicles:
    OBSERVER = "Ego"
    TARGETS = [v for v in vehicles if v != "Ego"]
elif len(vehicles) > 0:
    OBSERVER = vehicles[0]
    TARGETS = vehicles[1:]
else:
    print("❌ Nenhum veículo encontrado!")
    exit(1)

print(f"\n📷 Observador: {OBSERVER}")
print(f"🎯 Alvos: {TARGETS}")

# Preparação com delays
print("\n🚁 Preparando veículos (pode demorar)...")
for vehicle in vehicles:
    try:
        print(f"   Habilitando {vehicle}...")
        client.enableApiControl(True, vehicle)
        time.sleep(1)  # Delay entre comandos

        client.armDisarm(True, vehicle)
        time.sleep(1)
    except Exception as e:
        print(f"   ⚠️ Erro em {vehicle}: {e}")

# Decolagem lenta
print("\n🛫 Decolando (um por vez)...")
for vehicle in vehicles:
    try:
        print(f"   Decolando {vehicle}...")
        client.takeoffAsync(vehicle_name=vehicle)
        time.sleep(DELAY_AFTER_COMMAND)  # Aguarda cada um
    except:
        pass

print("   Aguardando estabilização...")
time.sleep(5)

# Posicionamento inicial
print("\n📍 Posicionando veículos...")

# Observador em posição estratégica
if OBSERVER:
    try:
        client.moveToPositionAsync(
            -30, 0, -25, 3,  # Velocidade menor (3 m/s)
            vehicle_name=OBSERVER
        )
        time.sleep(DELAY_AFTER_MOVEMENT)
    except:
        pass

# Alvos em formação inicial
initial_positions = [
    (10, 0, -20),    # Frente
    (10, 10, -22),   # Frente-direita
    (10, -10, -22),  # Frente-esquerda
    (20, 0, -25),    # Mais longe
]

for i, target in enumerate(TARGETS[:4]):
    try:
        if i < len(initial_positions):
            x, y, z = initial_positions[i]
            client.moveToPositionAsync(x, y, z, 3, vehicle_name=target)
            time.sleep(1)  # Pequeno delay entre comandos
    except:
        pass

time.sleep(DELAY_AFTER_MOVEMENT)

# Geração do dataset
TOTAL_FRAMES = 50  # Menos frames para teste
frame_count = 0

print(f"\n🎬 Capturando {TOTAL_FRAMES} frames")
print("   (Com delays para evitar travamento)\n")

for frame in range(TOTAL_FRAMES):
    t = frame / TOTAL_FRAMES * 2 * math.pi

    # Movimento suave dos alvos
    for i, target in enumerate(TARGETS[:4]):
        try:
            # Padrão circular suave
            angle = (2 * math.pi * i / min(4, len(TARGETS))) + t

            x = 15 + 10 * math.cos(angle)
            y = 10 * math.sin(angle)
            z = -22 + 5 * math.sin(t)

            # Movimento com velocidade baixa
            client.moveToPositionAsync(x, y, z, 2, vehicle_name=target)
        except:
            pass

    # Aguarda movimento
    time.sleep(1)

    # Captura imagem
    try:
        # Tenta múltiplas câmeras
        cameras = ["front_center", "0", ""]

        captured = False
        for cam in cameras:
            if not captured:
                try:
                    png = client.simGetImage(
                        cam,
                        airsim.ImageType.Scene,
                        vehicle_name=OBSERVER
                    )

                    if png and len(png) > 1000:
                        filename = f"frame_{frame_count:04d}.png"
                        filepath = output_dir / "images" / filename

                        with open(filepath, 'wb') as f:
                            f.write(png)

                        frame_count += 1
                        captured = True

                        # Salva metadata
                        metadata = {
                            "frame": frame_count,
                            "camera": cam,
                            "timestamp": time.time()
                        }

                        meta_file = output_dir / f"frame_{frame_count:04d}.json"
                        with open(meta_file, 'w') as f:
                            json.dump(metadata, f)

                        break
                except:
                    continue

        if captured:
            print(f"📸 Frame {frame_count}/{TOTAL_FRAMES}", end='\r')
        else:
            print(f"⚠️ Frame {frame} - não capturou", end='\r')

    except Exception as e:
        print(f"❌ Erro no frame {frame}: {e}")

    # Delay entre frames
    time.sleep(0.5)

# Pouso lento
print(f"\n\n🛬 Pousando {len(vehicles)} veículos...")
for vehicle in vehicles:
    try:
        client.landAsync(vehicle_name=vehicle)
        time.sleep(1)
    except:
        pass

time.sleep(5)

# Desarma
for vehicle in vehicles:
    try:
        client.armDisarm(False, vehicle)
        client.enableApiControl(False, vehicle)
    except:
        pass

# Resumo
num_images = len(list((output_dir / "images").glob("*.png")))

print("\n" + "="*60)
print("📊 RESUMO DA COLETA")
print("="*60)
print(f"✅ Imagens capturadas: {num_images}")
print(f"📁 Salvas em: {output_dir.absolute()}")

if num_images == 0:
    print("\n⚠️ NENHUMA IMAGEM FOI CAPTURADA!")
    print("Possíveis problemas:")
    print("1. Ambiente muito pesado - tente reduzir qualidade gráfica")
    print("2. Câmeras não configuradas - verifique settings.json")
    print("3. Ambiente travado - reinicie o simulador")
else:
    print(f"\n✅ Dataset gerado com sucesso!")

print("="*60)