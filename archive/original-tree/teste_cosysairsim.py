#!/usr/bin/env python3
"""
Teste com Cosys AirSim (Colosseum)
"""

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time

print("🔌 Conectando com Cosys AirSim...")
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

print("✅ Conectado!")

# Cria diretório
output = Path("dataset_cosys")
output.mkdir(exist_ok=True)

# Lista veículos
vehicles = client.listVehicles()
print(f"Veículos: {vehicles}")

# Prepara Ego
print("Preparando Ego...")
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")

# Decola
print("Decolando...")
client.takeoffAsync(vehicle_name="Ego").join()
time.sleep(3)

print("\n📸 Tentando capturar com Cosys AirSim...")

# Teste 1: simGetImages
try:
    print("Tentando simGetImages...")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses and responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        filename = output / "cosys_teste.png"
        cv2.imwrite(str(filename), img_bgr)
        print(f"✅ SUCESSO COM COSYSAIRSIM!")
        print(f"   Imagem salva: {filename}")
        print(f"   Tamanho: {responses[0].width}x{responses[0].height}")
    else:
        print("❌ Sem dados de imagem")

except Exception as e:
    print(f"❌ Erro: {e}")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

# Verifica resultados
images = list(output.glob("*.png"))
if images:
    print(f"\n✅ SUCESSO! Imagem capturada com Cosys AirSim!")
else:
    print(f"\n❌ Falhou com Cosys AirSim também")