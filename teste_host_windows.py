#!/usr/bin/env python3
"""
Teste direto com host.docker.internal e IPs locais do Windows
"""

import sys
import socket
import airsim

print("\n🔍 Testando conexão com AirSim no Windows\n")

# Tenta resolver o hostname do Windows
try:
    windows_ip = socket.gethostbyname("host.docker.internal")
    print(f"host.docker.internal resolvido para: {windows_ip}")
except:
    windows_ip = None
    print("Não foi possível resolver host.docker.internal")

# Lista de IPs para testar (incluindo IPs locais do Windows)
test_ips = [
    "172.19.80.1",      # IP WSL padrão
    "192.168.1.100",    # IP local comum
    "192.168.0.100",    # IP local alternativo
    "10.0.0.100",       # IP local alternativo
    "127.0.0.1",        # localhost
]

if windows_ip:
    test_ips.insert(0, windows_ip)

# Adiciona o IP do host Windows baseado no gateway
try:
    with open("/proc/net/route") as f:
        for line in f:
            fields = line.strip().split()
            if fields[1] == "00000000":  # Default gateway
                gateway = socket.inet_ntoa(bytes.fromhex(fields[2])[::-1])
                if gateway not in test_ips:
                    test_ips.insert(0, gateway)
                    print(f"Gateway detectado: {gateway}")
except:
    pass

print(f"\nTestando {len(test_ips)} endereços IP...\n")

for ip in test_ips:
    print(f"🔌 Testando {ip}:41451... ", end="", flush=True)

    try:
        client = airsim.MultirotorClient(ip=ip, port=41451, timeout_value=2)
        client.confirmConnection()

        print("✅ CONECTADO!")

        vehicles = client.listVehicles()
        print(f"   Veículos: {vehicles}")

        # Salva o IP que funcionou
        with open("IP_AIRSIM.txt", "w") as f:
            f.write(f"{ip}\n")

        print(f"\n✅ SUCESSO! Use o IP: {ip}")
        print("   Salvo em IP_AIRSIM.txt")

        # Testa captura rápida
        print("\n📸 Testando captura de imagem...")
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if responses and responses[0].image_data_uint8:
            print("   ✅ Captura funcionando!")
        else:
            print("   ⚠️ Captura não retornou dados")

        break

    except Exception as e:
        print("❌")
        continue
else:
    print("\n❌ Nenhum IP funcionou!")
    print("\n📝 Para descobrir o IP correto:")
    print("1. No Windows, abra PowerShell como admin")
    print("2. Digite: Get-NetIPAddress | Where-Object {$_.InterfaceAlias -like '*WSL*'}")
    print("3. Use o IPv4Address mostrado")
    print("\nOu no cmd: ipconfig e procure 'vEthernet (WSL)'")