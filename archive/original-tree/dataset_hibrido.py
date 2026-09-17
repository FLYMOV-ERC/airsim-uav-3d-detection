#!/usr/bin/env python3
"""
Dataset Híbrido: Captura cenários bonitos + Blocks com drones
Depois você pode combinar as imagens
"""

import airsim
import time
from pathlib import Path

print("🎨 DATASET HÍBRIDO")
print("="*60)
print("Estratégia:")
print("1. Captura paisagens do Mountains/Abandoned")
print("2. Captura drones do Blocks")
print("3. Combine depois com edição")
print("="*60)

# Cria estrutura
output = Path("dataset_hibrido")
(output / "cenarios").mkdir(parents=True, exist_ok=True)
(output / "drones").mkdir(parents=True, exist_ok=True)

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

ambiente_atual = input("\nQual ambiente está rodando? (mountains/abandoned/blocks): ").lower()

if ambiente_atual in ["mountains", "abandoned"]:
    print(f"\n📸 Capturando CENÁRIOS do {ambiente_atual}")

    # Captura várias vistas do cenário
    for i in range(20):
        # Varia horário e clima
        hora = 6 + i  # De 6h às 2h da manhã
        fog = (i % 5) * 0.2  # Varia neblina

        client.simSetTimeOfDay(True, f"2024-01-01 {hora:02d}:00:00")
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)

        time.sleep(0.5)

        img = client.simGetImage("0", 0, "Ego")
        if img:
            filename = output / "cenarios" / f"{ambiente_atual}_{i:03d}.png"
            with open(filename, 'wb') as f:
                f.write(img)
            print(f"   Cenário {i+1}/20", end='\r')

    print(f"\n✅ 20 cenários salvos em {output}/cenarios/")
    print("\n🔄 Agora FECHE este ambiente e ABRA o Blocks")
    print("   Depois execute este script novamente")

elif ambiente_atual == "blocks":
    print("\n🚁 Capturando DRONES do Blocks")

    # Prepara drones
    for v in client.listVehicles():
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        client.takeoffAsync(vehicle_name=v)

    time.sleep(5)

    # Captura drones em várias posições
    for i in range(20):
        # Move drones
        t = i / 20 * 3.14159 * 2
        for j, v in enumerate(["Drone3", "Drone4", "Intruder1"]):
            if v in client.listVehicles():
                x = 10 + 5 * math.cos(t + j)
                y = 5 * math.sin(t + j)
                z = -20 + 3 * math.sin(t)
                client.moveToPositionAsync(x, y, z, 3, vehicle_name=v)

        time.sleep(0.5)

        img = client.simGetImage("0", 0, "Ego")
        if img:
            filename = output / "drones" / f"drones_{i:03d}.png"
            with open(filename, 'wb') as f:
                f.write(img)
            print(f"   Drones {i+1}/20", end='\r')

    # Pousa
    for v in client.listVehicles():
        client.landAsync(vehicle_name=v)

    print(f"\n✅ 20 imagens com drones salvas em {output}/drones/")

print("\n" + "="*60)
print("📋 PRÓXIMOS PASSOS:")
print("1. Use software de edição (Photoshop, GIMP, etc)")
print("2. Combine cenários + drones")
print("3. Ou use IA para gerar drones nos cenários")
print("="*60)