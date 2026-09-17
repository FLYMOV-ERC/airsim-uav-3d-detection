#!/usr/bin/env python3
"""
Script mínimo para testar captura - compatível com diferentes versões
"""

import airsim
import numpy as np
import cv2
from pathlib import Path
import time

print("🔌 Conectando...")
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

print("✅ Conectado!")

# Cria diretório
output = Path("dataset_minimo")
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

print("\n📸 Tentando capturar imagem...")

# Teste 1: simGetImage sem parâmetros extras
try:
    print("Método 1: simGetImage básico...")
    img_bytes = client.simGetImage("0", airsim.ImageType.Scene)

    if img_bytes:
        # Decodifica imagem comprimida
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            filename = output / "teste1.png"
            cv2.imwrite(str(filename), img)
            print(f"✅ SUCESSO! Imagem salva: {filename}")
            print(f"   Tamanho: {img.shape}")
        else:
            print("❌ Falhou ao decodificar")
    else:
        print("❌ Sem dados de imagem")

except Exception as e:
    print(f"❌ Erro método 1: {e}")

# Teste 2: simGetImages
try:
    print("\nMétodo 2: simGetImages...")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    if responses and responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        filename = output / "teste2.png"
        cv2.imwrite(str(filename), img_bgr)
        print(f"✅ SUCESSO! Imagem salva: {filename}")
        print(f"   Tamanho: {responses[0].width}x{responses[0].height}")
    else:
        print("❌ Sem dados de imagem")

except Exception as e:
    print(f"❌ Erro método 2: {e}")

# Teste 3: simGetImage com vehicle_name
try:
    print("\nMétodo 3: simGetImage com vehicle_name...")
    img_bytes = client.simGetImage("0", airsim.ImageType.Scene, "Ego")

    if img_bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            filename = output / "teste3.png"
            cv2.imwrite(str(filename), img)
            print(f"✅ SUCESSO! Imagem salva: {filename}")
            print(f"   Tamanho: {img.shape}")
        else:
            print("❌ Falhou ao decodificar")
    else:
        print("❌ Sem dados de imagem")

except Exception as e:
    print(f"❌ Erro método 3: {e}")

# Teste 4: simGetImages com vehicle_name
try:
    print("\nMétodo 4: simGetImages com vehicle_name...")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], "Ego")

    if responses and responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        filename = output / "teste4.png"
        cv2.imwrite(str(filename), img_bgr)
        print(f"✅ SUCESSO! Imagem salva: {filename}")
        print(f"   Tamanho: {responses[0].width}x{responses[0].height}")
    else:
        print("❌ Sem dados de imagem")

except Exception as e:
    print(f"❌ Erro método 4: {e}")

# Teste 5: Usar nome de câmera diferente
try:
    print("\nMétodo 5: Nome de câmera 'front_center'...")
    img_bytes = client.simGetImage("front_center", airsim.ImageType.Scene, "Ego")

    if img_bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            filename = output / "teste5.png"
            cv2.imwrite(str(filename), img)
            print(f"✅ SUCESSO! Imagem salva: {filename}")
            print(f"   Tamanho: {img.shape}")
        else:
            print("❌ Falhou ao decodificar")
    else:
        print("❌ Sem dados de imagem")

except Exception as e:
    print(f"❌ Erro método 5: {e}")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

# Verifica resultados
print("\n" + "="*50)
print("RESUMO:")
print("="*50)

images = list(output.glob("*.png"))
if images:
    print(f"✅ SUCESSO! {len(images)} imagens capturadas:")
    for img in images:
        print(f"   - {img.name}")
else:
    print("❌ Nenhuma imagem foi capturada")

print(f"\n📁 Diretório: {output.absolute()}")