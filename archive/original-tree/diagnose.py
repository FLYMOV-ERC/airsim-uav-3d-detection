#!/usr/bin/env python3
import socket
import subprocess
import airsim

print("🔍 DIAGNÓSTICO DE CONEXÃO AIRSIM\n")

# 1. Verifica IP do host Windows
result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'default' in line:
        host_ip = line.split()[2]
        print(f"✅ IP do Windows detectado: {host_ip}")
        break
else:
    host_ip = "172.19.80.1"
    print(f"⚠️  Usando IP padrão: {host_ip}")

# 2. Testa várias portas
print(f"\n📡 Testando portas em {host_ip}:")
ports_to_test = [41451, 41452, 9000, 9001, 14560, 14570]

for port in ports_to_test:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host_ip, port))
    sock.close()

    if result == 0:
        print(f"  ✅ Porta {port}: ABERTA")

        # Tenta conectar via AirSim
        try:
            client = airsim.MultirotorClient(ip=host_ip, port=port, timeout_value=3)
            client.confirmConnection()
            print(f"     🎯 AIRSIM CONECTADO na porta {port}!")

            vehicles = client.listVehicles()
            print(f"     Veículos: {vehicles}")
            break
        except:
            print(f"     ⚠️  Porta aberta mas não é AirSim")
    else:
        print(f"  ❌ Porta {port}: FECHADA")

print("\n📋 CHECKLIST:")
print("1. ✓ AirSim está rodando no Windows? (Blocks.exe ou AirSimNH.exe)")
print("2. ✓ Você reiniciou o AirSim após editar settings.json?")
print("3. ✓ O settings.json tem 'LocalHostIp': '0.0.0.0'?")
print("4. ✓ O settings.json tem 'ApiServerPort': 41451?")
print("5. ✓ A regra do firewall foi criada?")
print("\n💡 No Windows, verifique se a porta está escutando:")
print("   netstat -an | findstr 41451")