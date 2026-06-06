#!/usr/bin/env python3
"""
Verificação rápida - Apenas lê posições atuais dos drones
"""

import cosysairsim as airsim

print("VERIFICAÇÃO RÁPIDA DE POSIÇÕES")
print("="*40)

try:
    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado")

    # Lista veículos
    vehicles = client.listVehicles()
    print(f"Veículos: {vehicles}")

    print("\n📍 POSIÇÕES ATUAIS:")
    resultados = []

    for v in vehicles:
        try:
            pose = client.simGetVehiclePose(vehicle_name=v)
            x = pose.position.x_val
            y = pose.position.y_val
            z = pose.position.z_val
            altura = -z

            if altura > 15:
                status = "✅ ALTO"
            elif altura > 10:
                status = "✅ VOANDO"
            elif altura > 5:
                status = "⚠️ BAIXO"
            elif altura > 1:
                status = "❌ MUITO BAIXO"
            else:
                status = "❌ NO CHÃO"

            print(f"   {v:12}: x={x:6.1f} y={y:6.1f} z={z:6.1f} | altura={altura:5.1f}m | {status}")
            resultados.append((v, altura, status))

        except Exception as e:
            print(f"   {v:12}: ERRO - {e}")
            resultados.append((v, 0, "❌ ERRO"))

    # Análise
    print(f"\n📊 ANÁLISE:")
    voando = sum(1 for _, altura, status in resultados if altura > 10)
    total = len(resultados)

    print(f"   Total drones: {total}")
    print(f"   Voando (>10m): {voando}")
    print(f"   Taxa sucesso: {voando/total*100:.0f}%")

    if voando == total:
        print("   🎉 TODOS VOANDO!")
    elif voando > 0:
        print("   ⚠️ ALGUNS VOANDO")
    else:
        print("   ❌ NENHUM VOANDO")

except Exception as e:
    print(f"❌ ERRO: {e}")

print("="*40)