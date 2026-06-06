#!/usr/bin/env python3
import airsim
import socket
import time

# Possíveis IPs e portas
ips = [
    "172.19.80.1",  # IP padrão do Windows host no WSL
    "localhost",
    "127.0.0.1",
    "host.docker.internal",  # Caso seja similar ao Docker
]

ports = [41451, 41452, 9000, 9001]  # Portas comuns do AirSim

print("Testando conexões possíveis...\n")

for ip in ips:
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, port))
            sock.close()

            if result == 0:
                print(f"✅ Porta aberta: {ip}:{port}")

                # Tenta conectar ao AirSim
                try:
                    client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=2)
                    client.confirmConnection()
                    print(f"   🎯 CONECTADO AO AIRSIM em {ip}:{port}!")

                    state = client.getMultirotorState()
                    print(f"   Posição: X={state.kinematics_estimated.position.x_val:.2f}")
                    vehicles = client.listVehicles()
                    print(f"   Veículos: {vehicles}")
                    print("\n   USE ESTE IP E PORTA PARA CONECTAR!\n")
                    break
                except:
                    print(f"   ⚠️  Porta aberta mas não é AirSim")

        except:
            pass

print("\nTestando firewall do Windows...")
print("Se não conseguiu conectar:")
print("1. Verifique se o AirSim está rodando (Blocks.exe ou AirSimNH.exe)")
print("2. No Windows, verifique o Windows Defender Firewall")
print("3. Permita conexões do WSL ao AirSim")