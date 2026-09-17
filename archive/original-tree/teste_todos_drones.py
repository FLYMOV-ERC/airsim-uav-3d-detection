#!/usr/bin/env python3
"""
Teste completo - Verifica altura de todos os drones
"""

import cosysairsim as airsim
import time

print("TESTE COMPLETO - TODOS OS DRONES")
print("="*50)

try:
    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao simulador")

    # Lista veículos
    vehicles = client.listVehicles()
    print(f"Veículos: {vehicles}")

    # Posições iniciais
    print("\n📍 POSIÇÕES INICIAIS:")
    posicoes_iniciais = {}
    for v in vehicles:
        try:
            pose = client.simGetVehiclePose(vehicle_name=v)
            z = pose.position.z_val
            posicoes_iniciais[v] = z
            print(f"   {v}: z={z:.2f} (altura={-z:.2f}m)")
        except Exception as e:
            print(f"   {v}: ERRO - {e}")

    # Prepara todos
    print("\n🔧 PREPARANDO DRONES...")
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
            print(f"   {v}: preparado")
        except Exception as e:
            print(f"   {v}: ERRO - {e}")

    # Decola todos
    print("\n🛫 DECOLANDO...")
    for v in vehicles:
        try:
            client.takeoffAsync(vehicle_name=v).join()
            print(f"   {v}: decolado")
        except Exception as e:
            print(f"   {v}: ERRO - {e}")

    time.sleep(3)

    # Verifica após decolagem
    print("\n📍 APÓS DECOLAGEM:")
    for v in vehicles:
        try:
            pose = client.simGetVehiclePose(vehicle_name=v)
            z = pose.position.z_val
            altura = -z
            status = "✅ VOANDO" if altura > 10 else "⚠️ BAIXO" if altura > 5 else "❌ NO CHÃO"
            print(f"   {v}: z={z:.2f} (altura={altura:.2f}m) {status}")
        except Exception as e:
            print(f"   {v}: ERRO - {e}")

    # Move para posições específicas
    print("\n🎯 MOVENDO PARA POSIÇÕES DE TESTE...")
    movimentos = [
        ("Ego", 0, 0, -20),
        ("Drone3", 10, -5, -18),
        ("Drone4", 15, 5, -16),
        ("Intruder1", 20, 0, -22)
    ]

    for nome, x, y, z in movimentos:
        if nome in vehicles:
            try:
                print(f"   {nome}: movendo para ({x}, {y}, {z})")
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=nome).join()
            except Exception as e:
                print(f"   {nome}: ERRO - {e}")

    time.sleep(2)

    # Posições finais
    print("\n📍 POSIÇÕES FINAIS:")
    relatorio = []
    for v in vehicles:
        try:
            pose = client.simGetVehiclePose(vehicle_name=v)
            x = pose.position.x_val
            y = pose.position.y_val
            z = pose.position.z_val
            altura = -z

            if altura > 15:
                status = "✅ EXCELENTE"
            elif altura > 10:
                status = "✅ BOM"
            elif altura > 5:
                status = "⚠️ BAIXO"
            else:
                status = "❌ PROBLEMA"

            print(f"   {v}: pos=({x:.1f}, {y:.1f}, {z:.1f}) altura={altura:.1f}m {status}")
            relatorio.append((v, x, y, z, altura, status))
        except Exception as e:
            print(f"   {v}: ERRO - {e}")
            relatorio.append((v, 0, 0, 0, 0, "❌ ERRO"))

    # Pousa todos
    print("\n🛬 POUSANDO TODOS...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v).join()
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    # Resumo final
    print("\n" + "="*50)
    print("📊 RELATÓRIO FINAL")
    print("="*50)

    voando_ok = 0
    total = len(relatorio)

    for v, x, y, z, altura, status in relatorio:
        print(f"{v:12}: {altura:5.1f}m {status}")
        if "✅" in status:
            voando_ok += 1

    print(f"\nRESULTADO: {voando_ok}/{total} drones voando corretamente")

    if voando_ok == total:
        print("🎉 TODOS OS DRONES ESTÃO VOANDO!")
    elif voando_ok > total/2:
        print("⚠️ MAIORIA DOS DRONES OK, alguns problemas")
    else:
        print("❌ PROBLEMA SÉRIO - Poucos drones voando")

except Exception as e:
    print(f"❌ ERRO GERAL: {e}")
    import traceback
    traceback.print_exc()

print("="*50)