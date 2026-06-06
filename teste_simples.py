#!/usr/bin/env python3
"""
Teste simplificado - Apenas conecta, decola e verifica alturas
"""

import cosysairsim as airsim
import time

print("TESTE SIMPLES - VERIFICAÇÃO RÁPIDA")
print("="*50)

try:
    # Conecta
    print("1. Conectando ao simulador...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("   ✅ Conectado!")

    # Lista veículos
    vehicles = client.listVehicles()
    print(f"2. Veículos encontrados: {vehicles}")

    # Prepara apenas o Ego
    ego = "Ego"
    if ego in vehicles:
        print(f"3. Preparando {ego}...")
        client.enableApiControl(True, ego)
        client.armDisarm(True, ego)

        # Verifica posição inicial
        pose = client.simGetVehiclePose(vehicle_name=ego)
        z_inicial = pose.position.z_val
        print(f"   Posição inicial: z={z_inicial:.2f}")

        # Decola
        print("4. Decolando...")
        client.takeoffAsync(vehicle_name=ego).join()
        time.sleep(2)

        # Verifica após decolagem
        pose = client.simGetVehiclePose(vehicle_name=ego)
        z_decolagem = pose.position.z_val
        print(f"   Após decolagem: z={z_decolagem:.2f} (altura={-z_decolagem:.2f}m)")

        # Move para altura específica
        print("5. Movendo para -15m...")
        client.moveToPositionAsync(0, 0, -15, 5, vehicle_name=ego).join()
        time.sleep(1)

        # Verifica posição final
        pose = client.simGetVehiclePose(vehicle_name=ego)
        x, y, z = pose.position.x_val, pose.position.y_val, pose.position.z_val
        print(f"   Posição final: x={x:.1f}, y={y:.1f}, z={z:.1f}")
        print(f"   Altura real: {-z:.1f} metros")

        # Análise
        if z < -10:
            print("   ✅ DRONE ESTÁ VOANDO! (altura > 10m)")
        elif z < -5:
            print("   ⚠️  Drone baixo (5-10m)")
        else:
            print("   ❌ PROBLEMA: Drone muito baixo ou no chão!")

        # Pousa
        print("6. Pousando...")
        client.landAsync(vehicle_name=ego).join()
        client.armDisarm(False, ego)
        client.enableApiControl(False, ego)

        print("✅ TESTE CONCLUÍDO!")

    else:
        print("❌ Ego não encontrado!")

except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

print("="*50)