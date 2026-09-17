#!/usr/bin/env python3
"""
test_airsim_connection.py
Testa conexão real com o AirSim
"""

import sys
import time

try:
    import airsim
except:
    print("❌ Módulo airsim não encontrado. Instalando...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "airsim"])
    import airsim

def test_connection():
    """Testa conexão com AirSim"""

    print("="*60)
    print("🔌 TESTANDO CONEXÃO COM AIRSIM")
    print("="*60)

    # Tentar diferentes IPs comuns
    ips_to_try = [
        ("localhost", 41451),
        ("127.0.0.1", 41451),
        ("172.19.80.1", 41451),
        ("192.168.1.1", 41451)
    ]

    for ip, port in ips_to_try:
        print(f"\n📡 Tentando conectar em {ip}:{port}...")
        try:
            client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=3)
            client.confirmConnection()
            print(f"✅ CONECTADO com sucesso em {ip}:{port}!")

            # Obter informações
            print("\n📊 Informações do AirSim:")

            # Listar veículos
            try:
                vehicles = client.listVehicles()
                print(f"  Veículos disponíveis: {vehicles}")
            except:
                print("  Não foi possível listar veículos")

            # Verificar estado
            try:
                state = client.getMultirotorState()
                print(f"  Estado do drone principal: Ready")
                print(f"  Posição: x={state.kinematics_estimated.position.x_val:.2f}, "
                      f"y={state.kinematics_estimated.position.y_val:.2f}, "
                      f"z={state.kinematics_estimated.position.z_val:.2f}")
            except Exception as e:
                print(f"  Erro ao obter estado: {e}")

            return client

        except Exception as e:
            print(f"  ❌ Falha: {e}")
            continue

    print("\n❌ NÃO FOI POSSÍVEL CONECTAR AO AIRSIM!")
    print("\n⚠️  Certifique-se de que:")
    print("  1. O AirSim está rodando (Blocks.exe, City.exe, etc.)")
    print("  2. O ambiente carregou completamente")
    print("  3. As configurações de rede estão corretas")
    print("  4. O arquivo settings.json está em ~/Documents/AirSim/")

    return None

if __name__ == "__main__":
    client = test_connection()

    if client:
        print("\n" + "="*60)
        print("✅ AIRSIM ESTÁ FUNCIONANDO!")
        print("="*60)
        print("\nPróximos passos:")
        print("1. Copiar settings.json para ~/Documents/AirSim/")
        print("2. Reiniciar o AirSim para carregar os 5 drones")
        print("3. Executar: python collect-dataset-multi.py")
    else:
        print("\n" + "="*60)
        print("❌ AIRSIM NÃO ESTÁ ACESSÍVEL")
        print("="*60)
        print("\nPor favor:")
        print("1. Inicie o AirSim (ex: Blocks.exe)")
        print("2. Aguarde o ambiente carregar")
        print("3. Execute este script novamente")