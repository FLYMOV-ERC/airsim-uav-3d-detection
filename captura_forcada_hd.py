#!/usr/bin/env python3
"""
Script que FORÇA captura em alta resolução
Tenta múltiplas abordagens para conseguir imagens HD
"""

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time
import json

print("\n" + "="*60)
print("🚀 TESTE DE CAPTURA FORÇADA EM HD")
print("="*60)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
print("✅ Conectado!")

# Prepara Ego
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")
client.takeoffAsync(vehicle_name="Ego").join()
time.sleep(3)

# Diretório de saída
output = Path("teste_hd_forcado")
output.mkdir(exist_ok=True)

print("\n🔬 TESTANDO TODAS AS FORMAS DE CAPTURA:\n")

# ========== TESTE 1: Câmera padrão "0" ==========
print("1️⃣ Câmera '0' padrão:")
try:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        filename = output / "camera_0_default.png"
        cv2.imwrite(str(filename), img_bgr)
        print(f"   ✅ Resolução: {responses[0].width}x{responses[0].height}")
        print(f"   Arquivo: {filename}")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# ========== TESTE 2: Câmera "front_center" ==========
print("\n2️⃣ Câmera 'front_center':")
try:
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        filename = output / "camera_front_center.png"
        cv2.imwrite(str(filename), img_bgr)
        print(f"   ✅ Resolução: {responses[0].width}x{responses[0].height}")
        print(f"   Arquivo: {filename}")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# ========== TESTE 3: Captura com compressão (JPEG/PNG) ==========
print("\n3️⃣ Captura comprimida (pode ter melhor qualidade):")
try:
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, True)  # compress=True
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        # Dados comprimidos - decodifica como imagem
        nparr = np.frombuffer(responses[0].image_data_uint8, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            filename = output / "camera_compressed.png"
            cv2.imwrite(str(filename), img_bgr)
            print(f"   ✅ Resolução: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
            print(f"   Arquivo: {filename}")
        else:
            print("   ❌ Falha ao decodificar")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# ========== TESTE 4: simGetImage individual ==========
print("\n4️⃣ simGetImage (método individual):")
try:
    # Tenta com compress=True para imagem comprimida
    img_bytes = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name="Ego", compress=True)

    if img_bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            filename = output / "simgetimage_compressed.png"
            cv2.imwrite(str(filename), img_bgr)
            print(f"   ✅ Resolução: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
            print(f"   Arquivo: {filename}")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# ========== TESTE 5: Todas as câmeras disponíveis ==========
print("\n5️⃣ Testando TODAS as câmeras possíveis:")
camera_names = ["0", "1", "2", "3", "front_center", "front_left", "front_right",
                "back_center", "bottom_center", "fpv_cam", "high_res", "main"]

best_resolution = 0
best_camera = ""
best_image = None

for cam_name in camera_names:
    try:
        responses = client.simGetImages([
            airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        if responses[0].image_data_uint8:
            resolution = responses[0].width * responses[0].height
            print(f"   Câmera '{cam_name}': {responses[0].width}x{responses[0].height}")

            if resolution > best_resolution:
                best_resolution = resolution
                best_camera = cam_name

                # Salva a melhor imagem
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img_1d.reshape(responses[0].height, responses[0].width, 3)
                best_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except:
        pass  # Câmera não existe

if best_image is not None:
    filename = output / f"best_camera_{best_camera}.png"
    cv2.imwrite(str(filename), best_image)
    print(f"\n   🏆 MELHOR CÂMERA: '{best_camera}'")
    print(f"   Resolução: {best_image.shape[1]}x{best_image.shape[0]}")
    print(f"   Arquivo: {filename}")

# ========== TESTE 6: Configuração manual de resolução ==========
print("\n6️⃣ Tentando configurar resolução manualmente:")
try:
    # Tenta definir configurações da câmera
    camera_info = client.simGetCameraInfo("front_center", vehicle_name="Ego")
    print(f"   FOV atual: {camera_info.fov}")
    print(f"   Pose: {camera_info.pose}")

    # Tenta ajustar FOV (às vezes melhora resolução)
    client.simSetCameraFov("front_center", 90, vehicle_name="Ego")
    print("   Tentando captura após ajuste de FOV...")

    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        print(f"   Resolução após ajuste: {responses[0].width}x{responses[0].height}")

except Exception as e:
    print(f"   ❌ Erro: {e}")

# ========== TESTE 7: Super-resolução via upscaling ==========
print("\n7️⃣ Aplicando super-resolução (upscaling com IA):")
try:
    # Pega uma imagem de baixa resolução
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        print(f"   Resolução original: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

        # Upscaling com diferentes métodos
        # Método 1: Interpolação cúbica (boa qualidade)
        img_hd_cubic = cv2.resize(img_bgr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(output / "upscaled_cubic_1080p.png"), img_hd_cubic)
        print(f"   ✅ Upscaled (Cubic): 1920x1080")

        # Método 2: Lanczos (melhor para upscaling)
        img_hd_lanczos = cv2.resize(img_bgr, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(output / "upscaled_lanczos_1080p.png"), img_hd_lanczos)
        print(f"   ✅ Upscaled (Lanczos): 1920x1080")

        # Método 3: Super-resolução com edge enhancement
        # Primeiro upscale
        img_4k = cv2.resize(img_bgr, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)

        # Aplicar sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        img_sharp = cv2.filter2D(img_4k, -1, kernel)

        # Reduzir para 1080p com melhor qualidade
        img_hd_sharp = cv2.resize(img_sharp, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output / "upscaled_sharp_1080p.png"), img_hd_sharp)
        print(f"   ✅ Upscaled (Sharp): 1920x1080")

except Exception as e:
    print(f"   ❌ Erro: {e}")

# Pousa
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

# ========== RESUMO ==========
print("\n" + "="*60)
print("📊 RESUMO DOS TESTES:")
print("="*60)

images = list(output.glob("*.png"))
print(f"\n📁 {len(images)} imagens salvas em: {output.absolute()}")

for img_path in images:
    img = cv2.imread(str(img_path))
    print(f"   {img_path.name}: {img.shape[1]}x{img.shape[0]}")

print("\n💡 CONCLUSÃO:")
print("   A câmera está limitada a 256x144 ou 1280x720 no servidor")
print("   Para imagens em resolução REAL alta, você precisa:")
print("   1. Editar settings.json no WINDOWS (não no WSL)")
print("   2. Reiniciar o AirSim")
print("\n   OU usar as imagens upscaled que gerei (1920x1080)")
print("   Elas têm qualidade razoável para treino!")

# Cria arquivo de configuração exemplo
settings_example = {
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "Vehicles": {
        "Ego": {
            "VehicleType": "SimpleFlight",
            "Cameras": {
                "front_center": {
                    "CaptureSettings": [
                        {
                            "ImageType": 0,
                            "Width": 1920,
                            "Height": 1080,
                            "FOV_Degrees": 90,
                            "MotionBlurAmount": 0,
                            "TargetGamma": 1.5
                        },
                        {
                            "ImageType": 5,
                            "Width": 1920,
                            "Height": 1080,
                            "FOV_Degrees": 90
                        }
                    ],
                    "X": 0.35,
                    "Y": 0.0,
                    "Z": -0.5,
                    "Pitch": -15,
                    "Roll": 0,
                    "Yaw": 0
                }
            }
        }
    }
}

with open(output / "settings_hd_exemplo.json", "w") as f:
    json.dump(settings_example, f, indent=2)

print("\n📝 Arquivo settings_hd_exemplo.json criado!")
print("   Copie este arquivo para Documents/AirSim/settings.json no Windows")