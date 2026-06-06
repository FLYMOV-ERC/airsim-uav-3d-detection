#!/usr/bin/env python3
"""
Script simples para testar conexão com AirSim
"""

import sys
import time

try:
    import airsim
except ImportError:
    print("❌ Erro: biblioteca airsim não está instalada")
    print("Execute: pip install airsim")
    sys.exit(1)

print("=" * 60)
print("🚁 TESTE DE CONEXÃO COM AIRSIM")
print("=" * 60)

# IP do host Windows (ajuste se necessário)
WINDOWS_HOST = "172.19.80.1"
PORT = 41451

print(f"\n📡 Tentando conectar em {WINDOWS_HOST}:{PORT}...")
print("   (Aguarde até 10 segundos)")

try:
    # Cria cliente
    client = airsim.MultirotorClient(ip=WINDOWS_HOST, port=PORT)

    # Tenta confirmar conexão
    client.confirmConnection()

    print("\n✅ SUCESSO! Conectado ao AirSim!")

    # Tenta obter informações adicionais
    try:
        vehicles = client.listVehicles()
        print(f"\n🚁 Veículos disponíveis: {vehicles}")
    except:
        pass

    try:
        # Tenta pegar o estado de um veículo
        state = client.getMultirotorState()
        print(f"\n📍 Posição do drone:")
        print(f"   X: {state.kinematics_estimated.position.x_val:.2f}")
        print(f"   Y: {state.kinematics_estimated.position.y_val:.2f}")
        print(f"   Z: {state.kinematics_estimated.position.z_val:.2f}")
    except:
        pass

    print("\n✅ AirSim está funcionando corretamente!")

except Exception as e:
    print(f"\n❌ ERRO: Não foi possível conectar ao AirSim")
    print(f"   Detalhes: {e}")
    print("\n⚠️  INSTRUÇÕES:")
    print("   1. Certifique-se de que o AirSim está rodando no Windows")
    print("   2. Abra o Blocks.exe, AirSimNH.exe ou outro ambiente")
    print("   3. Aguarde o ambiente carregar completamente")
    print("   4. Verifique se o arquivo settings.json está em:")
    print("      C:\\Users\\[seu_usuario]\\Documents\\AirSim\\settings.json")
    print(f"   5. Se o IP {WINDOWS_HOST} estiver errado, edite este script")
    print("\n💡 DICA: No Windows, você pode verificar o IP com:")
    print("   ipconfig | findstr WSL")
    sys.exit(1)

print("=" * 60)