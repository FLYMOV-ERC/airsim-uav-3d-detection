#!/usr/bin/env python3
"""
Gerador automático de dataset - Mountains
"""

import airsim
import time
import json
from pathlib import Path
from datetime import datetime

print("🚀 GERANDO DATASET NO MOUNTAINS")
print("="*60)

# Setup
output_dir = Path("dataset_mountains")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()
print(f"✅ Conectado! {len(drones)} drones: {drones}\n")

# Prepara drones
print("🚁 Preparando drones...")
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)
print("✅ Drones no ar!\n")

# Configurações de cenários
scenarios = [
    {"name": "Dia Claro", "fog": 0, "hour": 12},
    {"name": "Neblina", "fog": 0.8, "hour": 8},
    {"name": "Entardecer", "fog": 0.2, "hour": 18},
    {"name": "Noite", "fog": 0, "hour": 22},
]

FRAMES_PER_SCENARIO = 25  # 25 frames x 4 cenários = 100 frames total
frame_count = 0
start_time = time.time()

print(f"📊 Coletando {FRAMES_PER_SCENARIO * len(scenarios)} frames total\n")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Aplica clima e horário
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, scenario['fog'])
    client.simSetTimeOfDay(True, f"2024-01-01 {scenario['hour']:02d}:00:00")
    time.sleep(1)

    for frame in range(FRAMES_PER_SCENARIO):
        # Move drones em círculo
        import math
        t = frame / FRAMES_PER_SCENARIO * 2 * 3.14159

        for i, drone in enumerate(drones):
            angle = (2 * 3.14159 * i / len(drones)) + t
            x = 20 * math.cos(angle)
            y = 20 * math.sin(angle)
            z = -15 - i * 3
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone)

        time.sleep(0.5)

        # Captura imagens
        for drone in drones:
            try:
                # Captura RGB
                png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=drone)

                if png and len(png) > 1000:
                    # Salva imagem
                    img_file = output_dir / "images" / f"frame_{frame_count:04d}_{drone}.png"
                    with open(img_file, 'wb') as f:
                        f.write(png)

                    # Salva metadata
                    state = client.getMultirotorState(vehicle_name=drone)
                    pos = state.kinematics_estimated.position

                    metadata = {
                        "frame": frame_count,
                        "drone": drone,
                        "scenario": scenario['name'],
                        "position": {
                            "x": pos.x_val,
                            "y": pos.y_val,
                            "z": pos.z_val
                        }
                    }

                    meta_file = output_dir / "metadata" / f"frame_{frame_count:04d}_{drone}.json"
                    with open(meta_file, 'w') as f:
                        json.dump(metadata, f)

            except Exception as e:
                print(f"   ⚠️ Erro em {drone}: {e}")

        frame_count += 1
        print(f"   📸 Frame {frame_count}/{FRAMES_PER_SCENARIO * len(scenarios)}", end='\r')

    print()  # Nova linha após cenário

# Pousa drones
print("\n🛬 Pousando drones...")
for drone in drones:
    client.landAsync(vehicle_name=drone)
time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

# Estatísticas
elapsed = time.time() - start_time
num_images = len(list((output_dir / "images").glob("*.png")))

print("\n" + "="*60)
print("✅ DATASET GERADO COM SUCESSO!")
print("="*60)
print(f"📊 Frames capturados: {frame_count}")
print(f"🖼️ Imagens salvas: {num_images}")
print(f"⏱️ Tempo total: {elapsed:.1f} segundos")
print(f"📁 Local: {output_dir.absolute()}")
print("="*60)