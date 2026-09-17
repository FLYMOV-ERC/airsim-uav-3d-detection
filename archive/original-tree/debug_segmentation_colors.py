#!/usr/bin/env python3
"""
Debug para identificar as cores dos drones na segmentação
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys

print("\n" + "="*70)
print("🔍 DEBUG: IDENTIFICANDO CORES DOS DRONES")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\n🛫 Decolando...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Posiciona drones bem próximos e separados para identificação clara
print("\n📍 Posicionando drones para identificação...")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()

# Coloca cada drone em posição diferente para identificar suas cores
if "Drone3" in vehicles:
    client.moveToPositionAsync(5, -3, -10, 3, vehicle_name="Drone3").join()
    print("   Drone3: 5m frente, 3m esquerda")

if "Drone4" in vehicles:
    client.moveToPositionAsync(5, 0, -10, 3, vehicle_name="Drone4").join()
    print("   Drone4: 5m frente, centro")

if "Intruder1" in vehicles:
    client.moveToPositionAsync(5, 3, -10, 3, vehicle_name="Intruder1").join()
    print("   Intruder1: 5m frente, 3m direita")

time.sleep(2)

# Captura imagens
print("\n📸 Capturando imagens...")
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
], vehicle_name="Ego")

# Processa RGB
img_bgr = None
if responses[0].image_data_uint8:
    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Processa Segmentação
seg_image = None
if responses[1].image_data_uint8:
    seg_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
    seg_image = seg_1d.reshape(responses[1].height, responses[1].width, 3)

if img_bgr is not None and seg_image is not None:
    print("\n🎨 Analisando cores na segmentação...")

    # Encontra todas as cores únicas
    unique_colors = np.unique(seg_image.reshape(-1, 3), axis=0)

    print(f"\nTotal de cores únicas: {len(unique_colors)}")
    print("\nCores encontradas (RGB):")

    # Analisa cada cor
    color_info = []
    for i, color in enumerate(unique_colors):
        # Conta pixels desta cor
        mask = cv2.inRange(seg_image, color, color)
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (seg_image.shape[0] * seg_image.shape[1])) * 100

        color_info.append({
            'color': color,
            'count': pixel_count,
            'percentage': percentage
        })

        print(f"\n   Cor {i}: RGB{tuple(color)}")
        print(f"      Pixels: {pixel_count:,} ({percentage:.2f}%)")

        # Identifica o que provavelmente é cada cor
        if np.array_equal(color, [0, 0, 0]):
            print("      → Provável: Fundo/Vazio")
        elif percentage > 30:
            print("      → Provável: Céu ou Terreno (muito grande)")
        elif percentage < 0.01:
            print("      → Provável: Ruído (muito pequeno)")
        elif 0.05 < percentage < 5:
            print("      → Provável: DRONE (tamanho adequado)")

    # Cria visualização
    h, w = seg_image.shape[:2]
    vis = np.zeros((h*2, w*2, 3), dtype=np.uint8)

    # Coloca imagem RGB no canto superior esquerdo
    vis[:h, :w] = img_bgr

    # Coloca segmentação no canto superior direito
    vis[:h, w:] = seg_image

    # Cria visualização das cores dos drones (canto inferior)
    drone_colors = []
    y_offset = h + 20

    for info in sorted(color_info, key=lambda x: x['percentage']):
        color = info['color']
        pct = info['percentage']

        # Filtra cores candidatas a drone
        if 0.05 < pct < 5 and not np.array_equal(color, [0, 0, 0]):
            drone_colors.append(color)

            # Cria máscara para esta cor
            mask = cv2.inRange(seg_image, color, color)

            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Pega o maior contorno
                largest = max(contours, key=cv2.contourArea)
                x, y, w_bbox, h_bbox = cv2.boundingRect(largest)

                # Desenha na visualização
                color_bgr = tuple(int(c) for c in color[::-1])  # RGB to BGR
                cv2.rectangle(vis[:h, w:], (x, y), (x+w_bbox, y+h_bbox), color_bgr, 2)

                # Adiciona texto
                cv2.putText(vis, f"RGB{tuple(color)}", (w + x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

                # Mostra patch da cor
                cv2.rectangle(vis, (10, y_offset), (100, y_offset + 30), color_bgr, -1)
                cv2.putText(vis, f"Drone? {pct:.2f}%", (110, y_offset + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 40

    print(f"\n✅ Cores prováveis de drones: {len(drone_colors)}")
    for color in drone_colors:
        print(f"   RGB{tuple(color)}")

    # Salva visualização
    cv2.imwrite("debug_segmentation.png", vis)
    print("\n📸 Visualização salva em: debug_segmentation.png")

    # Mostra
    cv2.imshow("Debug Segmentation", vis)
    print("\nPressione qualquer tecla para continuar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Pousa
print("\n🛬 Pousando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n✅ Debug concluído!")