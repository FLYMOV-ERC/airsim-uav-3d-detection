#!/usr/bin/env python3
"""
Teste completo da API de segmentação do AirSim
Vamos entender como identificar drones na segmentação
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys

print("\n" + "="*70)
print("🔬 INVESTIGAÇÃO PROFUNDA: SEGMENTAÇÃO NO AIRSIM")
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

# Teste 1: Verificar se existe API para mesh IDs ou object IDs
print("\n📊 TESTE 1: Explorando APIs de segmentação...")
try:
    # Tenta pegar informações sobre os objetos na cena
    # O AirSim usa IDs de mesh para segmentação
    print("   Tentando obter informações de objetos...")

    # Prepara e decola apenas alguns drones para teste
    for v in ["Ego", "Drone3"]:
        if v in vehicles:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
            client.takeoffAsync(vehicle_name=v).join()

    time.sleep(3)

    # Posiciona drones
    client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
    client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone3").join()

    time.sleep(2)

except Exception as e:
    print(f"   Erro: {e}")

# Teste 2: Captura diferentes tipos de imagem
print("\n📊 TESTE 2: Tipos de imagem disponíveis...")
image_types = [
    (airsim.ImageType.Scene, "Scene (RGB)"),
    (airsim.ImageType.DepthPlanar, "DepthPlanar"),
    (airsim.ImageType.DepthPerspective, "DepthPerspective"),
    (airsim.ImageType.DepthVis, "DepthVis"),
    (airsim.ImageType.DisparityNormalized, "DisparityNormalized"),
    (airsim.ImageType.Segmentation, "Segmentation"),
    (airsim.ImageType.SurfaceNormals, "SurfaceNormals"),
    (airsim.ImageType.Infrared, "Infrared")
]

for img_type, name in image_types:
    try:
        response = client.simGetImages([
            airsim.ImageRequest("front_center", img_type, False, False)
        ], vehicle_name="Ego")[0]

        if response.image_data_uint8:
            print(f"   ✅ {name}: Disponível")
        else:
            print(f"   ❌ {name}: Sem dados")
    except Exception as e:
        print(f"   ❌ {name}: Erro - {e}")

# Teste 3: Analisar segmentação em detalhe
print("\n📊 TESTE 3: Análise detalhada da segmentação...")

# Captura segmentação
response = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
], vehicle_name="Ego")[0]

if response.image_data_uint8:
    img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    seg_img = img_1d.reshape(response.height, response.width, 3)

    # Analisa cores únicas e seus IDs
    unique_colors = np.unique(seg_img.reshape(-1, 3), axis=0)

    print(f"\n   Total de cores/IDs únicos: {len(unique_colors)}")

    # O AirSim mapeia object IDs para cores usando uma fórmula específica
    # ID = R + G*256 + B*256*256
    print("\n   Mapeamento de cores para IDs de objeto:")

    for color in unique_colors:
        # Calcula o ID do objeto baseado na cor
        object_id = int(color[0]) + int(color[1])*256 + int(color[2])*256*256

        # Conta pixels
        mask = cv2.inRange(seg_img, color, color)
        pixel_count = np.sum(mask > 0)
        percentage = (pixel_count / (seg_img.shape[0] * seg_img.shape[1])) * 100

        print(f"\n   Cor RGB{tuple(color)}:")
        print(f"      Object ID: {object_id}")
        print(f"      Pixels: {pixel_count:,} ({percentage:.3f}%)")

        # Tenta identificar o que é baseado em características
        if object_id == 0:
            print("      → Tipo: Background/Sky")
        elif percentage > 20:
            print("      → Tipo: Terreno/Grande objeto")
        elif 0.01 < percentage < 2.0:
            print("      → Tipo: Possível drone/objeto pequeno")

            # Verifica se é um objeto escuro (drones geralmente são escuros)
            if np.sum(color) < 100:
                print("      → CANDIDATO A DRONE (escuro e pequeno)")

# Teste 4: Tenta métodos alternativos
print("\n📊 TESTE 4: Métodos alternativos de detecção...")

# Método 1: Usar pose dos veículos para validar detecções
print("\n   Método 1: Validação por posição conhecida dos drones")
for v in vehicles:
    if v != "Ego":
        try:
            pose = client.simGetVehiclePose(vehicle_name=v)
            print(f"      {v}: Posição ({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {pose.position.z_val:.1f})")
        except:
            pass

# Método 2: Verificar se há API para object annotations
print("\n   Método 2: Procurando por APIs de anotação...")
try:
    # Tenta acessar diferentes APIs que podem existir
    if hasattr(client, 'simGetSegmentationObjectID'):
        print("      ✅ simGetSegmentationObjectID disponível!")
    else:
        print("      ❌ simGetSegmentationObjectID não encontrado")

    if hasattr(client, 'simSetSegmentationObjectID'):
        print("      ✅ simSetSegmentationObjectID disponível!")
    else:
        print("      ❌ simSetSegmentationObjectID não encontrado")

    if hasattr(client, 'simGetObjectPose'):
        print("      ✅ simGetObjectPose disponível!")
    else:
        print("      ❌ simGetObjectPose não encontrado")

except Exception as e:
    print(f"      Erro: {e}")

# Teste 5: Verificar configuração de mesh names
print("\n📊 TESTE 5: Configuração de segmentação por mesh...")
print("   IMPORTANTE: No Unreal Engine, você precisa:")
print("   1. Adicionar Stencil ID aos meshes dos drones")
print("   2. Configurar Custom Depth-Stencil nas propriedades do mesh")
print("   3. Definir um ID único para cada tipo de objeto")
print("   4. No AirSim, esses IDs são convertidos em cores na segmentação")

# Salva visualização para análise
print("\n💾 Salvando visualizações para análise...")

# RGB
rgb_response = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
], vehicle_name="Ego")[0]

if rgb_response.image_data_uint8:
    img_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    rgb_img = img_1d.reshape(rgb_response.height, rgb_response.width, 3)
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Cria visualização comparativa
    h, w = seg_img.shape[:2]
    vis = np.zeros((h, w*2, 3), dtype=np.uint8)
    vis[:, :w] = rgb_bgr
    vis[:, w:] = seg_img

    cv2.putText(vis, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Segmentation", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite("airsim_segmentation_analysis.png", vis)
    print("   ✅ Salvo: airsim_segmentation_analysis.png")

# Pousa
print("\n🛬 Finalizando...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("📋 CONCLUSÕES:")
print("="*70)
print("""
1. SEGMENTAÇÃO NO AIRSIM:
   - Usa Object IDs mapeados para cores RGB
   - ID = R + G*256 + B*256*256
   - Cada objeto único tem uma cor única

2. PROBLEMA ATUAL:
   - Drones podem não ter IDs de segmentação configurados
   - Ou estão usando o mesmo ID que o terreno

3. SOLUÇÕES POSSÍVEIS:

   A) NO UNREAL ENGINE:
      - Atribuir Stencil IDs únicos aos drones
      - Configurar Custom Depth-Stencil
      - Garantir que cada drone tenha ID diferente

   B) VIA CÓDIGO:
      - Usar posições conhecidas dos drones (ground truth)
      - Combinar segmentação com depth para validação
      - Usar detecção por características (cor escura + tamanho pequeno)

   C) ALTERNATIVA MAIS SIMPLES:
      - Usar bounding boxes 3D dos drones (se disponível na API)
      - Projetar para 2D usando parâmetros da câmera

4. RECOMENDAÇÃO:
   - Configure os Stencil IDs no Unreal para solução definitiva
   - Ou use depth + posição conhecida como workaround
""")