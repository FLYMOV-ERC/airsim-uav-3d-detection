#!/usr/bin/env python3
import airsim
import socket
import time

# Testa conectividade básica primeiro
WINDOWS_HOST = "172.19.80.1"
PORT = 41451

print(f"Testando porta {PORT} em {WINDOWS_HOST}...")

# Testa se a porta está aberta
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
result = sock.connect_ex((WINDOWS_HOST, PORT))
sock.close()

if result == 0:
    print(f"✅ Porta {PORT} está aberta!")

    # Tenta conectar ao AirSim
    print("Conectando ao AirSim...")
    try:
        client = airsim.MultirotorClient(ip=WINDOWS_HOST, port=PORT, timeout_value=5)
        client.confirmConnection()
        print("✅ Conectado com sucesso!")

        # Pega estado
        state = client.getMultirotorState()
        print(f"Posição: X={state.kinematics_estimated.position.x_val:.2f}, Y={state.kinematics_estimated.position.y_val:.2f}, Z={state.kinematics_estimated.position.z_val:.2f}")

        # Lista veículos
        vehicles = client.listVehicles()
        print(f"Veículos: {vehicles}")

    except Exception as e:
        print(f"❌ Erro ao conectar ao AirSim: {e}")
else:
    print(f"❌ Porta {PORT} está fechada ou inacessível")
    print("Certifique-se que o AirSim está rodando no Windows")