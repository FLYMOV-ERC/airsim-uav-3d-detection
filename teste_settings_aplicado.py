#!/usr/bin/env python3
"""
Verifica se o settings.json está sendo aplicado
"""

import airsim

print("🔍 VERIFICANDO SETTINGS.JSON\n")

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451, timeout_value=5)
client.confirmConnection()

# Verifica veículos
vehicles = client.listVehicles()
print(f"Veículos detectados: {vehicles}")

# O que deveria ter (baseado no settings.json)
expected = ["Ego", "Intruder1", "Intruder2", "Intruder3", "Intruder4"]

print("\n📋 Análise:")
print("-"*40)

if set(vehicles) == set(expected):
    print("✅ Settings.json está sendo aplicado corretamente!")
    print("   Todos os 5 veículos foram criados")
else:
    print("⚠️ Settings.json pode não estar sendo aplicado!")
    print(f"   Esperado: {expected}")
    print(f"   Encontrado: {vehicles}")
    print("\n   Possíveis problemas:")
    print("   1. Settings.json no local errado")
    print("   2. Ambiente sobrescrevendo configurações")
    print("   3. Erro de sintaxe no JSON")

# Testa câmeras
print("\n📷 Testando câmeras:")
print("-"*40)

cameras = ["front_center", "back_center", "bottom_center", "0", ""]

for cam in cameras:
    try:
        img = client.simGetImage(cam, airsim.ImageType.Scene, "Ego")
        if img and len(img) > 1000:
            print(f"✅ Câmera '{cam}': Funcionando ({len(img)} bytes)")
        else:
            print(f"❌ Câmera '{cam}': Sem dados")
    except Exception as e:
        print(f"❌ Câmera '{cam}': Erro - {str(e)[:30]}")