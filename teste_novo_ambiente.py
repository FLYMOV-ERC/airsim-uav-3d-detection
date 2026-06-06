#!/usr/bin/env python3
import airsim
import time

print("🚁 TESTANDO NOVO AMBIENTE\n")

try:
    # Conecta com timeout menor
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451, timeout_value=5)
    client.confirmConnection()

    print("✅ CONECTADO!")

    # Info básica
    vehicles = client.listVehicles()
    print(f"Veículos: {vehicles}")

    # Pega posição
    for v in vehicles:
        state = client.getMultirotorState(vehicle_name=v)
        pos = state.kinematics_estimated.position
        print(f"{v}: X={pos.x_val:.1f}, Y={pos.y_val:.1f}, Z={pos.z_val:.1f}")

    print("\n✅ NOVO AMBIENTE FUNCIONANDO!")
    print("🎉 Qual ambiente é esse? (Mountains, Africa, City, etc?)")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\nPossíveis problemas:")
    print("1. O ambiente ainda está carregando")
    print("2. Precisa configurar o settings.json")
    print("3. Versão incompatível")