#!/usr/bin/env python3
"""
Script simplificado para testar coleta multi-drone
Compatível com a versão atual do AirSim
"""

import airsim
import time
import json
from pathlib import Path

print("🚁 TESTE DE COLETA MULTI-DRONE\n")

# Conecta ao AirSim
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

# Lista todos os drones
drones = client.listVehicles()
print(f"✅ Drones encontrados: {drones}\n")

# Cria diretório de saída
output_dir = Path("teste_coleta")
output_dir.mkdir(exist_ok=True)

# Habilita controle de todos os drones
print("🎮 Preparando drones...")
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola todos
print("🛫 Decolando...")
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)

# ====== CENÁRIO 1: FORMAÇÃO EM LINHA ======
print("\n📍 CENÁRIO 1: Formação em linha")
for i, drone in enumerate(drones):
    x = i * 10  # Espaçamento de 10m entre drones
    y = 0
    z = -10
    print(f"   Movendo {drone} para posição ({x}, {y}, {z})")
    client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone)

time.sleep(5)

# Coleta dados de posição
frame_data = {"cenario": "linha", "drones": {}}
for drone in drones:
    state = client.getMultirotorState(vehicle_name=drone)
    pos = state.kinematics_estimated.position
    frame_data["drones"][drone] = {
        "x": pos.x_val,
        "y": pos.y_val,
        "z": pos.z_val
    }

# Salva dados
with open(output_dir / "cenario1_linha.json", 'w') as f:
    json.dump(frame_data, f, indent=2)

# ====== CENÁRIO 2: CÍRCULO ======
print("\n📍 CENÁRIO 2: Formação circular")
import math

num_drones = len(drones)
radius = 15

for i, drone in enumerate(drones):
    angle = (2 * math.pi * i) / num_drones
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = -12
    print(f"   Movendo {drone} para posição ({x:.1f}, {y:.1f}, {z})")
    client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone)

time.sleep(5)

# Coleta dados
frame_data = {"cenario": "circulo", "drones": {}}
for drone in drones:
    state = client.getMultirotorState(vehicle_name=drone)
    pos = state.kinematics_estimated.position
    frame_data["drones"][drone] = {
        "x": pos.x_val,
        "y": pos.y_val,
        "z": pos.z_val
    }

with open(output_dir / "cenario2_circulo.json", 'w') as f:
    json.dump(frame_data, f, indent=2)

# ====== MUDANÇA DE CLIMA ======
print("\n🌤️ Testando mudança de clima...")

# Limpo
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
print("   Clima: LIMPO")
time.sleep(2)

# Neblina
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.8)
print("   Clima: NEBLINA")
time.sleep(2)

# Volta ao limpo
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
print("   Clima: LIMPO")

# ====== POUSA TODOS ======
print("\n🛬 Pousando todos os drones...")
for drone in drones:
    client.landAsync(vehicle_name=drone)

time.sleep(5)

# Desarma
for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

print(f"\n✅ TESTE COMPLETO!")
print(f"📁 Dados salvos em: {output_dir.absolute()}")
print(f"🚁 Drones testados: {', '.join(drones)}")

# Mostra como adicionar mais drones
print("\n" + "="*60)
print("💡 PARA ADICIONAR MAIS DRONES:")
print("="*60)
print("1. No Windows, edite: C:\\Users\\[seu_usuario]\\Documents\\AirSim\\settings.json")
print("2. Adicione novos drones assim:\n")
print('''   "Drone3": {
      "VehicleType": "SimpleFlight",
      "X": 20, "Y": 0, "Z": -2
   },
   "Drone4": {
      "VehicleType": "SimpleFlight",
      "X": -20, "Y": 0, "Z": -2
   }''')
print("\n3. Salve o arquivo")
print("4. Reinicie o AirSim")
print("5. Execute este script novamente")
print("="*60)