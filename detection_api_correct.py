#!/usr/bin/env python3
"""
Implementação CORRETA da API simGetDetections baseada na documentação oficial
Testa diferentes mesh names e configurações para detectar drones
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
import sys
import json

print("\n" + "="*70)
print("🚀 IMPLEMENTAÇÃO CORRETA DA API simGetDetections")
print("="*70)
print("Baseado na documentação oficial do AirSim")

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos disponíveis: {vehicles}")
except Exception as e:
    print(f"❌ Erro conectando: {e}")
    sys.exit(1)

# Prepara todos os drones
print("\n🚁 Preparando drones...")
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        print(f"   ✅ {v} preparado")
    except Exception as e:
        print(f"   ⚠️ Erro preparando {v}: {e}")

# Decola todos
print("\n🛫 Decolando todos os drones...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
        print(f"   ✅ {v} decolou")
    except Exception as e:
        print(f"   ⚠️ Erro decolando {v}: {e}")

time.sleep(3)

# Posiciona drones em formação visível
print("\n📍 Posicionando drones em formação...")
positions = {
    "Ego": {"x": 0, "y": 0, "z": -10},
    "Drone3": {"x": 10, "y": -3, "z": -10},
    "Drone4": {"x": 10, "y": 0, "z": -10},
    "Intruder1": {"x": 10, "y": 3, "z": -10}
}

for vehicle, pos in positions.items():
    if vehicle in vehicles:
        try:
            client.moveToPositionAsync(pos["x"], pos["y"], pos["z"], 5, vehicle_name=vehicle).join()
            print(f"   ✅ {vehicle} posicionado em ({pos['x']}, {pos['y']}, {pos['z']})")
        except:
            pass

time.sleep(2)

# IMPORTANTE: Baseado na documentação, os parâmetros corretos são:
# - camera_name: pode ser nome ou ID ("0", "1", etc)
# - image_type: tipo de imagem (Scene, DepthPlanar, etc)
# - vehicle_name: nome do veículo que tem a câmera

print("\n" + "="*70)
print("📊 TESTANDO DETECÇÃO COM DIFERENTES CONFIGURAÇÕES")
print("="*70)

# Lista de mesh names para testar baseado na documentação e exemplos
mesh_patterns_to_test = [
    # Padrões genéricos
    "*",                    # Detecta tudo
    ".*",                   # Regex para tudo

    # Padrões baseados em BP_FlyingPawn (blueprint padrão do AirSim)
    "BP_FlyingPawn",        # Nome exato sem wildcard
    "BP_FlyingPawn*",       # Com wildcard
    "BP_FlyingPawn.*",      # Com regex
    "*FlyingPawn*",         # Wildcard em ambos lados

    # Padrões baseados em SimpleFlight (tipo de veículo)
    "SimpleFlight",
    "SimpleFlight*",
    "*SimpleFlight*",

    # Padrões baseados nos nomes dos veículos
    "Drone*",               # Detecta Drone3, Drone4
    "Intruder*",            # Detecta Intruder1
    "*drone*",              # Case insensitive
    "*intruder*",

    # Padrões baseados em componentes
    "Pawn",
    "Pawn*",
    "*Pawn*",
    "Vehicle",
    "Vehicle*",

    # Padrões específicos do Unreal
    "SM_*",                 # Static Mesh prefix
    "BP_*",                 # Blueprint prefix
    "*_C",                  # Class instance suffix
]

# Configurações de câmera para testar
camera_configs = [
    ("0", airsim.ImageType.Scene),          # ID numérico
    ("front_center", airsim.ImageType.Scene),  # Nome da câmera
]

results = {}
best_config = None
max_detections = 0

for camera_name, image_type in camera_configs:
    print(f"\n📷 Testando com câmera '{camera_name}', tipo {image_type}")

    for pattern in mesh_patterns_to_test:
        try:
            # PASSO 1: Limpar filtros anteriores
            client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")

            # PASSO 2: Definir raio de detecção (em centímetros!)
            radius_cm = 100 * 100  # 100 metros = 10000 cm
            client.simSetDetectionFilterRadius(camera_name, image_type, radius_cm, vehicle_name="Ego")

            # PASSO 3: Adicionar padrão de mesh
            if pattern:  # Se não for vazio, adiciona o filtro
                client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

            # Pequena pausa para garantir que o filtro foi aplicado
            time.sleep(0.05)

            # PASSO 4: Obter detecções
            detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

            num_detections = len(detections) if detections else 0

            if num_detections > 0:
                print(f"   ✅ Padrão '{pattern}': {num_detections} detecções!")

                # Mostra detalhes das detecções
                for i, det in enumerate(detections[:3]):  # Mostra até 3
                    print(f"      Detecção {i+1}:")

                    # Nome do objeto
                    if hasattr(det, 'name'):
                        print(f"         Nome: {det.name}")

                    # Bounding box 2D
                    if hasattr(det, 'box2D'):
                        if hasattr(det.box2D.min, 'x_val'):
                            print(f"         Box2D: ({det.box2D.min.x_val:.0f}, {det.box2D.min.y_val:.0f}) a ({det.box2D.max.x_val:.0f}, {det.box2D.max.y_val:.0f})")
                        else:
                            print(f"         Box2D: ({det.box2D.min.x:.0f}, {det.box2D.min.y:.0f}) a ({det.box2D.max.x:.0f}, {det.box2D.max.y:.0f})")

                    # Posição relativa
                    if hasattr(det, 'relative_pose'):
                        print(f"         Posição relativa: ({det.relative_pose.position.x_val:.1f}, {det.relative_pose.position.y_val:.1f}, {det.relative_pose.position.z_val:.1f})")

                # Atualiza melhor configuração
                if num_detections > max_detections:
                    max_detections = num_detections
                    best_config = (camera_name, image_type, pattern)

                # Salva resultado
                key = f"{camera_name}_{pattern}"
                results[key] = num_detections

        except Exception as e:
            print(f"   ⚠️ Erro com padrão '{pattern}': {str(e)[:100]}")

# Se encontrou alguma detecção, captura screenshot com visualização
if best_config and max_detections > 0:
    print("\n" + "="*70)
    print("📸 CAPTURANDO VISUALIZAÇÃO COM MELHOR CONFIGURAÇÃO")
    print("="*70)

    camera_name, image_type, pattern = best_config
    print(f"Melhor configuração encontrada:")
    print(f"   Câmera: '{camera_name}'")
    print(f"   Tipo: {image_type}")
    print(f"   Padrão: '{pattern}'")
    print(f"   Detecções: {max_detections}")

    # Configura com a melhor configuração
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 100 * 100, vehicle_name="Ego")
    if pattern:
        client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    time.sleep(0.1)

    # Captura imagem
    actual_camera = "front_center" if camera_name == "0" else camera_name
    response = client.simGetImages([
        airsim.ImageRequest(actual_camera, airsim.ImageType.Scene, False, False)
    ], vehicle_name="Ego")[0]

    if response.image_data_uint8:
        # Converte para OpenCV
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(response.height, response.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Pega detecções novamente
        detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

        # Desenha bounding boxes
        for det in detections:
            if hasattr(det, 'box2D'):
                # Tenta diferentes formatos de acesso aos valores
                try:
                    if hasattr(det.box2D.min, 'x_val'):
                        x_min = int(det.box2D.min.x_val)
                        y_min = int(det.box2D.min.y_val)
                        x_max = int(det.box2D.max.x_val)
                        y_max = int(det.box2D.max.y_val)
                    else:
                        x_min = int(det.box2D.min.x)
                        y_min = int(det.box2D.min.y)
                        x_max = int(det.box2D.max.x)
                        y_max = int(det.box2D.max.y)

                    # Desenha retângulo
                    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Adiciona label
                    if hasattr(det, 'name'):
                        cv2.putText(img_bgr, det.name, (x_min, y_min-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"   ⚠️ Erro desenhando bbox: {e}")

        # Salva imagem
        cv2.imwrite("detection_working.png", img_bgr)
        print("\n✅ Imagem salva: detection_working.png")

else:
    print("\n" + "="*70)
    print("❌ NENHUMA CONFIGURAÇÃO DETECTOU OBJETOS")
    print("="*70)

# Salva resultados
with open("detection_results.json", "w") as f:
    json.dump({
        "results": results,
        "best_config": {
            "camera": best_config[0] if best_config else None,
            "image_type": str(best_config[1]) if best_config else None,
            "pattern": best_config[2] if best_config else None,
            "max_detections": max_detections
        }
    }, f, indent=2)

print("\n💾 Resultados salvos em detection_results.json")

# Pousa todos os drones
print("\n🛬 Pousando todos os drones...")
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
        print(f"   ✅ {v} pousou")
    except:
        pass

print("\n" + "="*70)
print("📋 ANÁLISE DOS RESULTADOS")
print("="*70)

if max_detections > 0:
    print("✅ DETECÇÃO FUNCIONOU!")
    print(f"\nMelhor configuração:")
    print(f"   • Câmera: '{best_config[0]}'")
    print(f"   • Tipo de imagem: {best_config[1]}")
    print(f"   • Padrão de mesh: '{best_config[2]}'")
    print(f"   • Número de detecções: {max_detections}")
    print("\n👉 Use essa configuração no seu dataset generator!")
else:
    print("❌ DETECÇÃO NÃO FUNCIONOU COM NENHUMA CONFIGURAÇÃO")
    print("\nPossíveis soluções:")
    print("1. Configure mesh names no Unreal Engine:")
    print("   • Adicione tags aos BP_FlyingPawn")
    print("   • Configure Stencil IDs")
    print("   • Habilite detecção nos blueprints")
    print("\n2. Use versão mais recente do AirSim:")
    print("   • Compile do código fonte")
    print("   • Verifique se a API está habilitada")
    print("\n3. Use método alternativo:")
    print("   • Posições conhecidas dos drones (ground truth)")
    print("   • Projeção 3D para 2D manual")