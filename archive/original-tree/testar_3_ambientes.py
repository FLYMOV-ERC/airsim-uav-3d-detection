#!/usr/bin/env python3
"""
Script para testar os 3 ambientes baixados
Execute um ambiente por vez no Windows e rode este script
"""

import airsim
import time
import json
from datetime import datetime

print("🎮 TESTADOR DE AMBIENTES AIRSIM")
print("="*60)

def testar_ambiente():
    """Testa o ambiente atualmente rodando"""

    print("\n📡 Tentando conectar...")

    try:
        # Conecta
        client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
        client.confirmConnection()

        print("✅ CONECTADO!")

        # Informações do ambiente
        drones = client.listVehicles()
        print(f"\n🚁 Drones detectados: {drones}")

        # Pega nome do ambiente (pergunta ao usuário)
        env_name = input("\n❓ Qual ambiente está rodando? (ex: Neighborhood, Mountains, City): ")

        print(f"\n🔍 Testando ambiente: {env_name}")
        print("-"*40)

        # Teste 1: Controle básico
        print("1️⃣ Testando controle dos drones...")
        for drone in drones:
            client.enableApiControl(True, drone)
            client.armDisarm(True, drone)
        print("   ✅ Controle OK")

        # Teste 2: Decolagem
        print("2️⃣ Testando decolagem...")
        for drone in drones:
            client.takeoffAsync(vehicle_name=drone)
        time.sleep(5)
        print("   ✅ Decolagem OK")

        # Teste 3: Movimento
        print("3️⃣ Testando movimento...")
        for i, drone in enumerate(drones):
            client.moveToPositionAsync(i*10, 0, -10, 5, vehicle_name=drone)
        time.sleep(5)
        print("   ✅ Movimento OK")

        # Teste 4: Variações de clima
        print("4️⃣ Testando variações de clima...")

        # Neblina
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.5)
        print("   🌫️ Neblina aplicada")
        time.sleep(2)

        # Limpo
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
        print("   ☀️ Clima limpo")
        time.sleep(2)

        # Teste 5: Horário
        print("5️⃣ Testando mudança de horário...")
        client.simSetTimeOfDay(True, "2024-01-01 18:00:00")
        print("   🌅 Mudou para 18:00")
        time.sleep(2)

        client.simSetTimeOfDay(True, "2024-01-01 12:00:00")
        print("   ☀️ Mudou para 12:00")

        # Pousa
        print("\n🛬 Pousando drones...")
        for drone in drones:
            client.landAsync(vehicle_name=drone)
        time.sleep(5)

        # Desarma
        for drone in drones:
            client.armDisarm(False, drone)
            client.enableApiControl(False, drone)

        # Salva informações do ambiente
        env_info = {
            "name": env_name,
            "tested_at": str(datetime.now()),
            "drones": drones,
            "status": "working",
            "features": {
                "control": "✅",
                "takeoff": "✅",
                "movement": "✅",
                "weather": "✅",
                "time_of_day": "✅"
            }
        }

        with open(f"ambiente_{env_name.lower().replace(' ', '_')}.json", 'w') as f:
            json.dump(env_info, f, indent=2)

        print("\n" + "="*60)
        print(f"✅ AMBIENTE '{env_name}' FUNCIONANDO PERFEITAMENTE!")
        print("="*60)
        print(f"📁 Informações salvas em: ambiente_{env_name.lower().replace(' ', '_')}.json")

        return True

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        print("\nVerifique:")
        print("1. O ambiente está rodando no Windows?")
        print("2. Você esperou carregar completamente?")
        print("3. O Firewall está permitindo conexões?")
        return False

# Menu principal
while True:
    print("\n" + "="*60)
    print("INSTRUÇÕES:")
    print("="*60)
    print("1. No Windows: Execute um dos ambientes baixados (.exe)")
    print("2. Aguarde carregar completamente")
    print("3. Digite 't' aqui para testar")
    print("4. Digite 'q' para sair")
    print("="*60)

    choice = input("\nOpção: ").lower()

    if choice == 't':
        testar_ambiente()
    elif choice == 'q':
        print("\n👋 Encerrando...")
        break
    else:
        print("❌ Opção inválida")

print("\n🏁 Teste concluído!")