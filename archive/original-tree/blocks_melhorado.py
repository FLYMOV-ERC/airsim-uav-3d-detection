#!/usr/bin/env python3
"""
Usa Blocks mas com visual melhorado para parecer diferente
"""

import airsim
import time

print("🎨 MELHORANDO VISUAL DO BLOCKS\n")

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

# Transforma Blocks em diferentes ambientes visuais
cenarios = [
    {
        "nome": "Cidade Nebulosa",
        "fog": 0.6,
        "rain": 0,
        "hora": "07:00",
        "desc": "Manhã com neblina urbana"
    },
    {
        "nome": "Tempestade",
        "fog": 0.3,
        "rain": 0.9,
        "hora": "18:00",
        "desc": "Chuva forte ao entardecer"
    },
    {
        "nome": "Noite Urbana",
        "fog": 0.1,
        "rain": 0,
        "hora": "22:00",
        "desc": "Cidade à noite"
    },
    {
        "nome": "Deserto",
        "fog": 0,
        "dust": 0.4,
        "hora": "12:00",
        "desc": "Meio-dia com poeira"
    }
]

for cenario in cenarios:
    print(f"\n🎬 {cenario['nome']}: {cenario['desc']}")

    # Aplica clima
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, cenario.get('fog', 0))
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, cenario.get('rain', 0))
    client.simSetWeatherParameter(airsim.WeatherParameter.Dust, cenario.get('dust', 0))

    # Aplica horário
    client.simSetTimeOfDay(True, f"2024-01-01 {cenario['hora']}:00")

    time.sleep(3)

    # Captura
    img = client.simGetImage("0", 0, "Ego")
    if img:
        filename = f"blocks_{cenario['nome'].lower().replace(' ', '_')}.png"
        with open(filename, 'wb') as f:
            f.write(img)
        print(f"   📸 Salvou: {filename}")

print("\n✅ Blocks agora tem 4 visuais diferentes!")
print("   Use esses efeitos durante a coleta do dataset")