#!/usr/bin/env python3
"""
Teste básico de conexão com AirSim usando diferentes configurações
"""

import sys
import time

try:
    import airsim
    print("✅ Módulo airsim importado")
except ImportError:
    print("❌ Módulo airsim não encontrado")
    sys.exit(1)

# Testa diferentes configurações de conexão
configs = [
    ("localhost", 41451),
    ("127.0.0.1", 41451),
    ("172.19.80.1", 41451),
    ("host.docker.internal", 41451),
    ("172.17.0.1", 41451),
    ("192.168.1.1", 41451)
]

print("\n🔍 Testando conexões AirSim...\n")

for ip, port in configs:
    print(f"Tentando {ip}:{port}... ", end="", flush=True)

    try:
        # Tenta conectar com timeout curto
        client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=2)
        client.confirmConnection()

        print(f"✅ CONECTADO!")
        print(f"\n📡 Informações da conexão:")
        print(f"   IP: {ip}")
        print(f"   Porta: {port}")

        # Tenta listar veículos
        try:
            vehicles = client.listVehicles()
            print(f"   Veículos: {vehicles}")
        except:
            print("   Veículos: Não foi possível listar")

        # Salva configuração que funcionou
        with open("conexao_funcionando.txt", "w") as f:
            f.write(f"IP={ip}\nPORT={port}\n")
            f.write(f"VEHICLES={vehicles if 'vehicles' in locals() else 'unknown'}\n")

        print("\n✅ Configuração salva em 'conexao_funcionando.txt'")
        break

    except Exception as e:
        print(f"❌ Falhou")
        continue
else:
    print("\n❌ Nenhuma configuração funcionou!")
    print("\nVerifique se:")
    print("  1. O AirSim está rodando no Windows")
    print("  2. O ambiente 'Blocks' está carregado")
    print("  3. Firewall não está bloqueando")
    print("  4. WSL está configurado corretamente")