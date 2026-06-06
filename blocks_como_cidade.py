#!/usr/bin/env python3
"""
Faz o Blocks parecer uma cidade usando efeitos visuais
"""

import airsim
import time

print("🏙️ TRANSFORMANDO BLOCKS EM CIDADE\n")

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

print("Aplicando efeitos visuais...")

# Neblina urbana
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.3)
client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0.2)

# Horário de entardecer (golden hour)
client.simSetTimeOfDay(True, "2024-01-01 18:30:00")

print("✅ Visual urbano aplicado!")
print("\nO ambiente agora parece uma cidade ao entardecer")
print("com poluição e neblina urbana!")

# Você pode variar:
# - Noite: 22:00 (com fog 0.1) = cidade noturna
# - Manhã: 07:00 (com fog 0.5) = cidade com neblina matinal
# - Chuva: Rain 0.8 = cidade chuvosa