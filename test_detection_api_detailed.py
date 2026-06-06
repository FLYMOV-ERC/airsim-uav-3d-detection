#!/usr/bin/env python3
"""
Teste detalhado da API simGetDetections
Vamos descobrir como configurar corretamente para detectar drones
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys

print("\n" + "="*70)
print("🔬 TESTE DETALHADO: API simGetDetections")
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

# Teste 1: Verificar se as funções existem
print("\n📊 TESTE 1: Verificando funções disponíveis...")
detection_functions = [
    'simGetDetections',
    'simSetDetectionFilterRadius',
    'simAddDetectionFilterMeshName',
    'simClearDetectionMeshNames',
    'simGetMeshPositionVertexBuffers'
]

for func in detection_functions:
    if hasattr(client, func):
        print(f"   ✅ {func} disponível")
    else:
        print(f"   ❌ {func} NÃO disponível")

# Prepara drones
print("\n🚁 Preparando drones...")
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("🛫 Decolando...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Posiciona drones para teste
print("\n📍 Posicionando drones...")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
if "Drone3" in vehicles:
    client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone3").join()
    print("   Drone3 posicionado 10m à frente")
if "Drone4" in vehicles:
    client.moveToPositionAsync(15, 3, -10, 3, vehicle_name="Drone4").join()
    print("   Drone4 posicionado 15m à frente, 3m à direita")
time.sleep(2)

# Teste 2: Diferentes configurações de detecção
print("\n📊 TESTE 2: Testando diferentes configurações...")

# Lista de possíveis nomes de mesh para drones
mesh_patterns = [
    "*",  # Todos os objetos
    "Drone*",  # Padrão Drone
    "drone*",  # lowercase
    "SimpleFlight*",  # Tipo do veículo
    "Multirotor*",  # Tipo genérico
    "Vehicle*",  # Veículo genérico
    "Quadrotor*",  # Quadrotor
    "BP_FlyingPawn*",  # Blueprint padrão do AirSim
    ".*",  # Regex para tudo
    "Drone3",  # Nome específico
    "Intruder*",  # Intruder
]

# Configurações de câmera
camera_configs = [
    ("front_center", airsim.ImageType.Scene),
    ("0", airsim.ImageType.Scene),
    ("front_center", airsim.ImageType.DepthPlanar),
]

best_config = None
max_detections = 0

for camera_name, image_type in camera_configs:
    print(f"\n   Câmera: {camera_name}, Tipo: {image_type}")

    for pattern in mesh_patterns:
        try:
            # Limpa filtros anteriores
            client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")

            # Define raio de detecção (em cm)
            radius_cm = 10000  # 100 metros
            client.simSetDetectionFilterRadius(camera_name, image_type, radius_cm, vehicle_name="Ego")

            # Adiciona padrão de mesh
            client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

            # Tenta detectar
            detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

            if detections and len(detections) > 0:
                print(f"      ✅ Padrão '{pattern}': {len(detections)} detecções!")

                for det in detections[:3]:  # Mostra até 3 detecções
                    print(f"         - Nome: {det.name if hasattr(det, 'name') else 'N/A'}")
                    if hasattr(det, 'box2D'):
                        print(f"           Box2D: min({det.box2D.min.x:.0f}, {det.box2D.min.y:.0f}), max({det.box2D.max.x:.0f}, {det.box2D.max.y:.0f})")
                    if hasattr(det, 'relative_pose'):
                        print(f"           Posição relativa: ({det.relative_pose.position.x_val:.1f}, {det.relative_pose.position.y_val:.1f}, {det.relative_pose.position.z_val:.1f})")

                if len(detections) > max_detections:
                    max_detections = len(detections)
                    best_config = (camera_name, image_type, pattern)
            else:
                print(f"      ❌ Padrão '{pattern}': 0 detecções")

        except Exception as e:
            print(f"      ⚠️ Erro com padrão '{pattern}': {e}")

# Teste 3: Usar a melhor configuração encontrada
if best_config:
    print(f"\n📊 TESTE 3: Usando melhor configuração encontrada...")
    camera_name, image_type, pattern = best_config
    print(f"   Melhor config: câmera='{camera_name}', tipo={image_type}, padrão='{pattern}'")

    # Configura com a melhor configuração
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    # Captura imagem e detecções
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name if camera_name != "0" else "front_center",
                           airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")

    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Pega detecções
        detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

        # Desenha bounding boxes
        for det in detections:
            if hasattr(det, 'box2D'):
                x_min = int(det.box2D.min.x)
                y_min = int(det.box2D.min.y)
                x_max = int(det.box2D.max.x)
                y_max = int(det.box2D.max.y)

                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if hasattr(det, 'name'):
                    cv2.putText(img_bgr, det.name, (x_min, y_min-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite("test_detection_result.png", img_bgr)
        print(f"   ✅ Imagem salva: test_detection_result.png")
        print(f"   Total de detecções: {len(detections)}")

else:
    print("\n❌ Nenhuma configuração funcionou para detecção!")

# Teste 4: Tentar detecção sem filtros (detectar TUDO)
print("\n📊 TESTE 4: Detecção sem filtros específicos...")
try:
    # Limpa todos os filtros
    client.simClearDetectionMeshNames("front_center", airsim.ImageType.Scene, vehicle_name="Ego")

    # Define raio grande
    client.simSetDetectionFilterRadius("front_center", airsim.ImageType.Scene, 50000, vehicle_name="Ego")

    # Não adiciona nenhum filtro específico - tenta detectar tudo
    detections = client.simGetDetections("front_center", airsim.ImageType.Scene, vehicle_name="Ego")

    print(f"   Detecções sem filtros: {len(detections) if detections else 0}")

except Exception as e:
    print(f"   ⚠️ Erro: {e}")

# Teste 5: Verificar se há alguma API alternativa
print("\n📊 TESTE 5: Procurando APIs alternativas...")

# Lista de possíveis métodos relacionados
possible_methods = dir(client)
detection_related = [m for m in possible_methods if 'detect' in m.lower() or 'object' in m.lower() or 'mesh' in m.lower()]

if detection_related:
    print("   Métodos relacionados encontrados:")
    for method in detection_related:
        print(f"      - {method}")
else:
    print("   Nenhum método alternativo encontrado")

# Pousa
print("\n🛬 Pousando...")
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

if max_detections > 0:
    print(f"✅ DETECÇÃO FUNCIONOU!")
    print(f"   Melhor configuração: {best_config}")
    print(f"   Máximo de detecções: {max_detections}")
else:
    print("❌ DETECÇÃO NÃO FUNCIONOU")
    print("""
   Possíveis razões:
   1. A versão do AirSim não tem a API completa
   2. Os drones não têm mesh names detectáveis
   3. Precisa configuração adicional no Unreal Engine
   4. A API pode estar desabilitada no build

   Soluções:
   1. Compilar AirSim do código fonte mais recente
   2. Usar método alternativo (posições conhecidas)
   3. Configurar mesh names no Unreal Engine
   """)