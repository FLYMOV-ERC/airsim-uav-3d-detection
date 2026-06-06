#!/usr/bin/env python3
"""
Testa coleta de imagens no ambiente Mountains
"""

import airsim
import numpy as np
import cv2
import time
from pathlib import Path

print("🏔️ TESTE DE COLETA DE IMAGENS - MOUNTAINS\n")

# Cria pasta de teste
output_dir = Path("teste_mountains")
output_dir.mkdir(exist_ok=True)

try:
    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao Mountains!")

    # Lista veículos
    vehicles = client.listVehicles()
    print(f"🚁 Drones disponíveis: {vehicles}\n")

    # Prepara o drone Ego (principal)
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")

    print("🛫 Decolando Ego...")
    client.takeoffAsync(vehicle_name="Ego").join()
    time.sleep(3)

    # Move para posição melhor
    print("📍 Movendo para posição de captura...")
    client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

    print("\n📸 CAPTURANDO IMAGENS:")
    print("-" * 40)

    # Tenta diferentes métodos de captura
    image_types = [
        (airsim.ImageType.Scene, "RGB", 0),
        (airsim.ImageType.Segmentation, "Segmentação", 5),
        (airsim.ImageType.DepthVis, "Profundidade", 2)
    ]

    for img_type, nome, tipo_num in image_types:
        try:
            # Método 1: simGetImage
            print(f"\n🔍 Tentando capturar {nome} (método simGetImage)...")

            png_image = client.simGetImage("0", img_type, vehicle_name="Ego")

            if png_image:
                # Salva imagem PNG
                filename = f"mountains_{nome.lower()}_metodo1.png"
                filepath = output_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(png_image)
                print(f"   ✅ Salvo: {filename} ({len(png_image)} bytes)")
            else:
                print(f"   ❌ Sem dados")

        except Exception as e:
            print(f"   ⚠️ Erro método 1: {e}")

        try:
            # Método 2: simGetImages
            print(f"🔍 Tentando capturar {nome} (método simGetImages)...")

            responses = client.simGetImages([
                airsim.ImageRequest("0", img_type, False, False)
            ], vehicle_name="Ego")

            if responses and len(responses) > 0:
                response = responses[0]

                if response.image_data_uint8:
                    # Converte para array numpy
                    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img = img.reshape(response.height, response.width, 3)

                    # Salva com OpenCV
                    filename = f"mountains_{nome.lower()}_metodo2.png"
                    filepath = output_dir / filename
                    cv2.imwrite(str(filepath), img)
                    print(f"   ✅ Salvo: {filename} ({response.width}x{response.height})")
                else:
                    print(f"   ❌ Sem dados de imagem")

        except Exception as e:
            print(f"   ⚠️ Erro método 2: {e}")

    # Testa câmeras se configuradas
    print("\n📷 VERIFICANDO CÂMERAS CONFIGURADAS:")
    print("-" * 40)

    # Tenta diferentes nomes de câmera
    camera_names = ["0", "front_center", "front", "camera_1", ""]

    for cam_name in camera_names:
        try:
            print(f"Testando câmera '{cam_name}'...")
            img = client.simGetImage(cam_name, airsim.ImageType.Scene, vehicle_name="Ego")
            if img and len(img) > 100:
                print(f"  ✅ Câmera '{cam_name}' funcionando!")
                break
        except:
            pass

    # Verifica arquivos salvos
    print("\n📁 ARQUIVOS SALVOS:")
    print("-" * 40)

    files = list(output_dir.glob("*.png"))
    if files:
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"✅ {f.name} - {size:.1f} KB")
    else:
        print("❌ Nenhuma imagem foi salva")

    # Pousa
    print("\n🛬 Pousando...")
    client.landAsync(vehicle_name="Ego").join()
    client.armDisarm(False, "Ego")
    client.enableApiControl(False, "Ego")

    print("\n" + "="*60)
    if files:
        print(f"✅ SUCESSO! {len(files)} imagens salvas em: {output_dir.absolute()}")
        print("🎉 A coleta de imagens está funcionando no Mountains!")
    else:
        print("⚠️ PROBLEMA: Nenhuma imagem foi capturada")
        print("Possíveis soluções:")
        print("1. Verificar se o settings.json tem câmeras configuradas")
        print("2. Adicionar configuração de câmeras ao settings.json")

except Exception as e:
    print(f"\n❌ Erro geral: {e}")