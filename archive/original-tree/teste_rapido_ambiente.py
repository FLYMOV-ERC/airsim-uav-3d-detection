#!/usr/bin/env python3
"""
Teste rápido do ambiente AirSim atual
"""

import airsim
import time

print("🔍 TESTANDO AMBIENTE AIRSIM ATUAL\n")

try:
    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    print("✅ CONECTADO AO AMBIENTE!")

    # Descobre drones
    drones = client.listVehicles()
    print(f"🚁 Drones: {drones}")

    # Pega estado
    for drone in drones:
        state = client.getMultirotorState(vehicle_name=drone)
        pos = state.kinematics_estimated.position
        print(f"📍 {drone}: X={pos.x_val:.1f}, Y={pos.y_val:.1f}, Z={pos.z_val:.1f}")

    # Controle rápido
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

    print("\n✅ AMBIENTE FUNCIONANDO PERFEITAMENTE!")
    print("🎉 Você pode usar qualquer script de coleta neste ambiente!")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n⚠️  INSTRUÇÕES:")
    print("1. Execute um dos ambientes .exe no Windows")
    print("2. Aguarde carregar completamente")
    print("3. Execute este script novamente")