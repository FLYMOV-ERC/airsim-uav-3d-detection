#!/usr/bin/env python3
"""
Teste de conexão com IPs adicionais do Windows
"""

import sys
import time

try:
    import airsim
    print("✅ Módulo airsim importado")
except ImportError:
    print("❌ Módulo airsim não encontrado")
    sys.exit(1)

# IPs para testar
configs = [
    ("10.255.255.254", 41451),  # IP do resolv.conf
    ("172.19.80.1", 41451),      # Gateway padrão
    ("172.28.240.1", 41451),     # Outro IP comum WSL
    ("172.31.240.1", 41451),     # Outro IP comum WSL
    ("192.168.0.1", 41451),      # Roteador local
    ("192.168.1.1", 41451),      # Roteador local alternativo
]

print("\n🔍 Testando conexões AirSim com novos IPs...\n")

for ip, port in configs:
    print(f"Tentando {ip}:{port}... ", end="", flush=True)

    try:
        # Tenta conectar com timeout curto
        client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=3)
        client.confirmConnection()

        print(f"✅ CONECTADO!")
        print(f"\n📡 Conexão estabelecida com sucesso!")
        print(f"   IP: {ip}")
        print(f"   Porta: {port}")

        # Lista veículos
        vehicles = client.listVehicles()
        print(f"   Veículos detectados: {vehicles}")

        # Testa API
        print("\n🎮 Testando API...")
        client.enableApiControl(True, vehicles[0] if vehicles else "")
        print("   ✅ API habilitada")

        # Salva configuração
        with open("CONEXAO_FUNCIONANDO.txt", "w") as f:
            f.write(f"IP_FUNCIONANDO={ip}\n")
            f.write(f"PORTA={port}\n")
            f.write(f"VEICULOS={vehicles}\n")

        print(f"\n✅ IP que funciona: {ip}:{port}")
        print("   Configuração salva em CONEXAO_FUNCIONANDO.txt")
        break

    except Exception as e:
        print(f"❌ Falhou")
        continue
else:
    print("\n❌ Ainda não consegui conectar!")
    print("\nTente:")
    print("  1. No Windows, abra o cmd e digite: ipconfig")
    print("  2. Procure por 'Ethernet adapter vEthernet (WSL)'")
    print("  3. Use o IPv4 Address listado")
    print("  4. Certifique-se que o AirSim está rodando no Blocks")