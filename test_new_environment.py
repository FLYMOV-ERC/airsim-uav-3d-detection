#!/usr/bin/env python3
"""
Testa conexão com o novo ambiente AirSim
"""

import airsim
import time

print("🔄 TESTANDO NOVO AMBIENTE AIRSIM\n")

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao novo ambiente!")

    # Informações do ambiente
    drones = client.listVehicles()
    print(f"🚁 Drones disponíveis: {drones}")

    # Pega informação de cenário (se disponível)
    for drone in drones:
        state = client.getMultirotorState(vehicle_name=drone)
        pos = state.kinematics_estimated.position
        print(f"📍 {drone} está em: X={pos.x_val:.1f}, Y={pos.y_val:.1f}, Z={pos.z_val:.1f}")

    # Testa controle básico
    print("\n🎮 Testando controle...")
    client.enableApiControl(True)
    client.armDisarm(True)

    print("🛫 Decolando...")
    client.takeoffAsync().join()

    time.sleep(3)

    print("🛬 Pousando...")
    client.landAsync().join()

    client.armDisarm(False)
    client.enableApiControl(False)

    print("\n✅ NOVO AMBIENTE FUNCIONANDO PERFEITAMENTE!")
    print("🎉 Você pode usar todos os scripts de coleta neste ambiente!")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\nVerifique:")
    print("1. O novo ambiente está rodando no Windows?")
    print("2. Você esperou ele carregar completamente?")
    print("3. O settings.json está configurado corretamente?")