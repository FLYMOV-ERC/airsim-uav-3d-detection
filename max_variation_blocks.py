#!/usr/bin/env python3
"""
Maximiza variações visuais no ambiente Blocks
Cria aparência de múltiplos cenários usando apenas clima/horário/altitude
"""

import airsim
import time
import json
from pathlib import Path

print("🎨 CRIANDO MÚLTIPLOS 'CENÁRIOS' NO BLOCKS\n")

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

drones = client.listVehicles()
print(f"Drones: {drones}\n")

# Prepara drones
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)

# Decola
for drone in drones:
    client.takeoffAsync(vehicle_name=drone)
time.sleep(5)

# ========== CENÁRIO 1: MANHÃ CLARA ==========
print("🌅 CENÁRIO 1: Manhã Clara (parece campo aberto)")
client.simSetTimeOfDay(True, "2024-01-01 08:00:00")
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)

# Voa alto - minimiza blocos, parece campo
for i, drone in enumerate(drones):
    client.moveToPositionAsync(i*20, 0, -50, 5, vehicle_name=drone)
time.sleep(5)

# ========== CENÁRIO 2: NEBLINA DENSA ==========
print("🌫️ CENÁRIO 2: Neblina Densa (parece floresta misteriosa)")
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.95)

# Voa baixo - blocos parecem árvores na neblina
for i, drone in enumerate(drones):
    client.moveToPositionAsync(i*10, i*5, -5, 5, vehicle_name=drone)
time.sleep(5)

# ========== CENÁRIO 3: TEMPESTADE ==========
print("⛈️ CENÁRIO 3: Tempestade (parece mar revolto)")
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.3)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 0.2)

# Movimento ondulado
for i, drone in enumerate(drones):
    client.moveToPositionAsync(i*15, i*10, -20, 3, vehicle_name=drone)
time.sleep(5)

# ========== CENÁRIO 4: NOITE URBANA ==========
print("🌃 CENÁRIO 4: Noite (parece cidade à noite)")
client.simSetTimeOfDay(True, "2024-01-01 22:00:00")
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.1)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)

# Padrão urbano - ruas e quarteirões
for i, drone in enumerate(drones):
    client.moveToPositionAsync(i*30, 0, -15, 5, vehicle_name=drone)
time.sleep(5)

# ========== CENÁRIO 5: DESERTO ==========
print("🏜️ CENÁRIO 5: Deserto (meio-dia, sem sombras)")
client.simSetTimeOfDay(True, "2024-01-01 12:00:00")
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0.3)

# Voa muito alto - blocos somem, parece deserto
for i, drone in enumerate(drones):
    client.moveToPositionAsync(i*40, i*20, -100, 5, vehicle_name=drone)
time.sleep(5)

# ========== CENÁRIO 6: NEVE ==========
print("❄️ CENÁRIO 6: Ártico (neve e vento)")
client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 1.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.4)
client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0)

for i, drone in enumerate(drones):
    client.moveToPositionAsync(i*10, i*10, -30, 5, vehicle_name=drone)
time.sleep(5)

# Pousa
print("\n🛬 Pousando...")
for drone in drones:
    client.landAsync(vehicle_name=drone)
time.sleep(5)

for drone in drones:
    client.armDisarm(False, drone)
    client.enableApiControl(False, drone)

print("\n✅ DEMONSTRAÇÃO COMPLETA!")
print("📝 Você viu 6 'cenários' diferentes usando apenas:")
print("   - Mudanças de clima")
print("   - Mudanças de horário")
print("   - Diferentes altitudes")
print("   - Diferentes formações")
print("\n💡 Dica: Combine essas variações durante a coleta de dados!")
print("   Cada combinação cria um visual completamente diferente.")