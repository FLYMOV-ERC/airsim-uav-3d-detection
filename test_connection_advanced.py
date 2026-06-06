#!/usr/bin/env python3
"""
Script avançado para conectar ao AirSim do Windows via WSL
"""
import airsim
import sys
import time

print("=" * 60)
print("🚁 TESTE AVANÇADO DE CONEXÃO AIRSIM WSL→WINDOWS")
print("=" * 60)

# Configurações
WINDOWS_HOST = "172.19.80.1"  # IP do host Windows
PORT = 41451

print(f"\n📡 Conectando em {WINDOWS_HOST}:{PORT}...")
print("⏳ Timeout configurado para 10 segundos...")

try:
    # Cria cliente com timeout maior
    client = airsim.MultirotorClient(
        ip=WINDOWS_HOST,
        port=PORT,
        timeout_value=10
    )

    # Confirma conexão
    print("🔄 Confirmando conexão...")
    client.confirmConnection()

    print("\n✅ CONECTADO COM SUCESSO!")

    # Informações do ambiente
    print("\n📊 INFORMAÇÕES DO AMBIENTE:")
    print("-" * 40)

    # Lista veículos
    vehicles = client.listVehicles()
    print(f"🚁 Veículos disponíveis: {vehicles}")

    # Para cada veículo, pega informações
    for vehicle_name in vehicles:
        print(f"\n📍 Veículo: {vehicle_name}")

        # Pega estado
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        pos = state.kinematics_estimated.position

        print(f"   Posição: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
        print(f"   Armado: {state.landed_state == airsim.LandedState.Flying}")
        print(f"   GPS: {state.gps_location if hasattr(state, 'gps_location') else 'N/A'}")

    # Testa controle básico
    print("\n🎮 TESTANDO CONTROLES:")
    print("-" * 40)

    # Habilita API control
    print("Habilitando controle via API...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✅ Controle API habilitado e drone armado")

    # Testa takeoff
    print("\n🛫 Iniciando decolagem...")
    client.takeoffAsync().join()
    print("✅ Decolagem completa!")

    # Pausa
    time.sleep(2)

    # Pousa
    print("🛬 Pousando...")
    client.landAsync().join()
    print("✅ Pouso completo!")

    # Desarma
    client.armDisarm(False)
    client.enableApiControl(False)
    print("\n✅ Drone desarmado e controle devolvido")

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("=" * 60)

except airsim.msgpackrpc.error.ConnectionRefusedError:
    print("\n❌ ERRO: Conexão recusada!")
    print("\n📋 INSTRUÇÕES PARA CORRIGIR:")
    print("-" * 40)
    print("1. No Windows, localize: C:\\Users\\[seu_usuario]\\Documents\\AirSim\\settings.json")
    print("2. Adicione estas linhas no início do JSON:")
    print('   "LocalHostIp": "0.0.0.0",')
    print('   "ApiServerPort": 41451,')
    print("3. Salve o arquivo")
    print("4. Reinicie o simulador AirSim")
    print("5. Execute este script novamente")

except airsim.msgpackrpc.error.TimeoutError:
    print("\n❌ ERRO: Timeout na conexão!")
    print("\n📋 POSSÍVEIS CAUSAS:")
    print("-" * 40)
    print("1. AirSim não está rodando no Windows")
    print("2. Firewall do Windows está bloqueando")
    print("3. IP incorreto (atual: {})".format(WINDOWS_HOST))
    print("\n💡 Para descobrir o IP correto, execute no WSL:")
    print("   ip route | grep default | awk '{print $3}'")

except Exception as e:
    print(f"\n❌ ERRO INESPERADO: {e}")
    print("\n📋 INFORMAÇÕES DE DEBUG:")
    print("-" * 40)
    print(f"Tipo do erro: {type(e).__name__}")
    print(f"Mensagem: {str(e)}")

print("\n📚 Consulte setup_windows_airsim.md para instruções detalhadas")