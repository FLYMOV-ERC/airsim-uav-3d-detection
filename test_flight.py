#!/usr/bin/env python3
"""
Teste de voo completo com AirSim
"""
import airsim
import time

print("🚁 TESTE DE VOO AIRSIM\n")

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
print("✅ Conectado ao AirSim")

# Lista veículos
vehicles = client.listVehicles()
print(f"🚁 Veículos disponíveis: {vehicles}\n")

# Reseta o drone
print("🔄 Resetando simulação...")
client.reset()
time.sleep(1)

# Habilita controle API
print("🎮 Habilitando controle via API...")
client.enableApiControl(True)
client.armDisarm(True)

# Decola
print("🛫 Decolando...")
client.takeoffAsync().join()
print("✅ No ar!\n")

# Paira por 2 segundos
time.sleep(2)

# Move para frente
print("➡️ Movendo 5m para frente...")
client.moveToPositionAsync(5, 0, -3, 2).join()

# Move para direita
print("➡️ Movendo 5m para direita...")
client.moveToPositionAsync(5, 5, -3, 2).join()

# Volta para origem
print("🏠 Voltando para origem...")
client.moveToPositionAsync(0, 0, -3, 2).join()

# Pousa
print("🛬 Pousando...")
client.landAsync().join()

# Desarma
client.armDisarm(False)
client.enableApiControl(False)

print("\n✅ TESTE COMPLETO COM SUCESSO!")
print("O drone decolou, voou em um padrão e pousou!")

# Pega informações finais
state = client.getMultirotorState()
print(f"\n📍 Posição final: X={state.kinematics_estimated.position.x_val:.2f}, Y={state.kinematics_estimated.position.y_val:.2f}, Z={state.kinematics_estimated.position.z_val:.2f}")