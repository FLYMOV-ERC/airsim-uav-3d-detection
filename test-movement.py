#!/usr/bin/env python3
"""
Test simples de movimento para debug do AirSim
"""
import time
try:
    import airsim
except:
    import cosysairsim as airsim

def main():
    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao AirSim")

    # Habilita controle
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")

    client.enableApiControl(True, "Intruder1")
    client.armDisarm(True, "Intruder1")
    print("✅ Drones habilitados")

    # Decola
    print("Decolando...")
    client.takeoffAsync("Ego").join()
    client.takeoffAsync("Intruder1").join()
    time.sleep(3)

    print("\n🚁 TESTE DE MOVIMENTO CONTÍNUO")
    print("Ego deve ir para frente, Intruder deve fazer círculo")
    print("Pressione Ctrl+C para parar\n")

    try:
        count = 0
        while True:
            count += 1

            # EGO - sempre para frente (X positivo)
            print(f"[{count:3d}] Ego: movendo para frente...")
            client.moveByVelocityAsync(
                5, 0, 0,  # 5 m/s em X
                duration=3,
                vehicle_name="Ego"
            )

            # INTRUDER - movimento circular
            t = count * 0.2
            vx = 5 * math.cos(t)
            vy = 5 * math.sin(t)
            print(f"[{count:3d}] Intruder: vx={vx:.1f}, vy={vy:.1f}")
            client.moveByVelocityAsync(
                vx, vy, 0,
                duration=3,
                vehicle_name="Intruder1"
            )

            time.sleep(0.5)  # Envia comandos 2x por segundo

    except KeyboardInterrupt:
        print("\n⏹ Parando...")
        client.enableApiControl(False, "Ego")
        client.enableApiControl(False, "Intruder1")
        print("✅ Controle devolvido")

if __name__ == "__main__":
    import math
    main()