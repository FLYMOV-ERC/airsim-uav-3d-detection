#!/usr/bin/env python3
"""
Script de teste rápido para geração de dataset
"""

import airsim
import time
import numpy as np
import cv2
from pathlib import Path

print("🚀 Teste Rápido de Dataset")
print("="*50)

# Cria diretório
output_dir = Path("dataset_teste_rapido")
output_dir.mkdir(exist_ok=True)

try:
    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao AirSim")

    # Lista drones
    drones = client.listVehicles()
    print(f"📡 Drones disponíveis: {drones}")

    # Habilita Ego
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")
    print("🎮 Ego preparado")

    # Decola
    print("🛫 Decolando...")
    client.takeoffAsync(vehicle_name="Ego").join()
    time.sleep(3)

    # Move para posição
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()
    print("📍 Posicionado")

    # Captura 5 frames
    print("\n📸 Capturando imagens...")
    for i in range(5):
        # Captura RGB
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        if responses and responses[0].image_data_uint8:
            # Converte imagem
            img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Salva
            filename = output_dir / f"frame_{i:03d}.png"
            cv2.imwrite(str(filename), img_bgr)
            print(f"   Frame {i+1}/5 salvo")

        time.sleep(0.5)

    # Pousa
    print("\n🛬 Pousando...")
    client.landAsync(vehicle_name="Ego").join()
    client.armDisarm(False, "Ego")
    client.enableApiControl(False, "Ego")

    print("\n✅ Teste completo!")
    print(f"📁 Imagens em: {output_dir.absolute()}")

except Exception as e:
    print(f"\n❌ Erro: {e}")
    print("\nVerifique se:")
    print("  1. O AirSim está rodando")
    print("  2. O IP/porta estão corretos")
    print("  3. O ambiente está carregado")