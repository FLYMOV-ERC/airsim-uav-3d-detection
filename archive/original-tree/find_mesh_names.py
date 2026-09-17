#!/usr/bin/env python3
"""
Descobre os mesh names de todos os objetos na cena
Isso vai nos ajudar a entender como detectar os drones
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys
import json

print("\n" + "="*70)
print("🔍 DESCOBRINDO MESH NAMES DOS OBJETOS")
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

# Teste 1: Listar objetos da cena
print("\n📊 TESTE 1: Listando objetos da cena...")
try:
    # Tenta listar objetos da cena
    if hasattr(client, 'simListSceneObjects'):
        scene_objects = client.simListSceneObjects()
        print(f"   Total de objetos na cena: {len(scene_objects) if scene_objects else 0}")
        if scene_objects:
            print("\n   Primeiros 20 objetos:")
            for obj in scene_objects[:20]:
                print(f"      - {obj}")
            if len(scene_objects) > 20:
                print(f"      ... e mais {len(scene_objects) - 20} objetos")

            # Procura por objetos que podem ser drones
            drone_candidates = []
            keywords = ['drone', 'Drone', 'vehicle', 'Vehicle', 'multi', 'Multi',
                       'quad', 'Quad', 'rotor', 'Rotor', 'intruder', 'Intruder',
                       'flying', 'Flying', 'pawn', 'Pawn']

            for obj in scene_objects:
                for keyword in keywords:
                    if keyword in obj:
                        drone_candidates.append(obj)
                        break

            if drone_candidates:
                print(f"\n   🎯 Possíveis drones encontrados ({len(drone_candidates)}):")
                for candidate in drone_candidates:
                    print(f"      - {candidate}")
    else:
        print("   ❌ simListSceneObjects não disponível")

except Exception as e:
    print(f"   ⚠️ Erro: {e}")

# Teste 2: Testar tags dos objetos
print("\n📊 TESTE 2: Listando tags dos objetos...")
try:
    if hasattr(client, 'simListSceneObjectsTags'):
        tags = client.simListSceneObjectsTags()
        if tags:
            print(f"   Tags encontradas: {tags}")
    else:
        print("   ❌ simListSceneObjectsTags não disponível")
except Exception as e:
    print(f"   ⚠️ Erro: {e}")

# Teste 3: Tentar usar simGetMeshPositionVertexBuffers
print("\n📊 TESTE 3: Testando simGetMeshPositionVertexBuffers...")
test_names = vehicles + ["Cylinder", "Ground", "Cube", "SimpleFlight", "Multirotor", "Quadrotor"]
for name in test_names:
    try:
        result = client.simGetMeshPositionVertexBuffers(name)
        if result:
            print(f"   ✅ Mesh '{name}' encontrado!")
            # Se encontrou, vamos testar se consegue detectar
            break
    except Exception as e:
        # Erro esperado para nomes não existentes
        pass

# Teste 4: Preparar drones e testar detecção com wildcard
print("\n📊 TESTE 4: Preparando drones para teste de detecção...")

# Prepara e posiciona drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

print("   Decolando...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Posiciona drones bem visíveis
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
if "Drone3" in vehicles:
    client.moveToPositionAsync(5, 0, -10, 3, vehicle_name="Drone3").join()
if "Drone4" in vehicles:
    client.moveToPositionAsync(7, 2, -10, 3, vehicle_name="Drone4").join()
if "Intruder1" in vehicles:
    client.moveToPositionAsync(10, -2, -10, 3, vehicle_name="Intruder1").join()

time.sleep(2)

# Teste 5: Detecção com configurações específicas
print("\n📊 TESTE 5: Testando detecção com diferentes configurações...")

camera_name = "front_center"
image_type = airsim.ImageType.Scene

# Lista de padrões para testar (mais específicos)
test_patterns = [
    "",  # Vazio - detecta tudo
    ".*",  # Regex para tudo
    "*",  # Wildcard para tudo
    "BP_FlyingPawn",  # Blueprint padrão do AirSim (sem wildcard)
    "BP_FlyingPawn*",  # Blueprint com wildcard
    "BP_FlyingPawn.*",  # Blueprint com regex
    "SimpleFlight",  # Tipo de veículo
    "SimpleFlight*",
    "Pawn",
    "Pawn*",
    "Vehicle",
    "Vehicle*"
]

results = {}

for pattern in test_patterns:
    try:
        # Limpa filtros
        client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")

        # Define raio grande
        client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")

        # Se o padrão não é vazio, adiciona o filtro
        if pattern:
            client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

        # Pequena pausa para garantir que o filtro foi aplicado
        time.sleep(0.1)

        # Tenta detectar
        detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

        num_detections = len(detections) if detections else 0
        results[pattern] = num_detections

        if num_detections > 0:
            print(f"   ✅ Padrão '{pattern if pattern else '(vazio)'}': {num_detections} detecções")
            # Mostra detalhes da primeira detecção
            det = detections[0]
            if hasattr(det, 'name'):
                print(f"      Primeiro objeto: {det.name}")
        else:
            print(f"   ❌ Padrão '{pattern if pattern else '(vazio)'}': 0 detecções")

    except Exception as e:
        print(f"   ⚠️ Erro com padrão '{pattern}': {e}")
        results[pattern] = -1

# Salva resultados
with open("mesh_detection_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Resultados salvos em mesh_detection_results.json")

# Se encontrou alguma detecção, tira screenshot
best_pattern = max(results, key=lambda k: results[k] if results[k] >= 0 else -1)
if results[best_pattern] > 0:
    print(f"\n📸 Capturando screenshot com padrão '{best_pattern}'...")

    # Configura com o melhor padrão
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    if best_pattern:
        client.simAddDetectionFilterMeshName(camera_name, image_type, best_pattern, vehicle_name="Ego")

    # Captura imagem
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
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
                x_min = int(det.box2D.min.x_val)
                y_min = int(det.box2D.min.y_val)
                x_max = int(det.box2D.max.x_val)
                y_max = int(det.box2D.max.y_val)

                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if hasattr(det, 'name'):
                    cv2.putText(img_bgr, det.name, (x_min, y_min-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite("mesh_detection_success.png", img_bgr)
        print("   ✅ Imagem salva: mesh_detection_success.png")

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
print("📋 RESUMO:")
print("="*70)

# Mostra resumo dos resultados
successful_patterns = {k: v for k, v in results.items() if v > 0}
if successful_patterns:
    print("✅ PADRÕES QUE FUNCIONARAM:")
    for pattern, count in sorted(successful_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"   '{pattern if pattern else '(vazio)'}': {count} detecções")
else:
    print("❌ NENHUM PADRÃO DETECTOU OBJETOS")
    print("\n   Isso sugere que:")
    print("   1. Os drones não têm mesh names configurados para detecção")
    print("   2. Ou precisam de configuração especial no Unreal Engine")
    print("   3. Ou a API precisa de inicialização diferente")