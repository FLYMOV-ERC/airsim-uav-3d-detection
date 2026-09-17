#!/usr/bin/env python3
"""
Descobre o mesh name EXATO dos drones
Vamos testar detecção e ver o nome retornado
"""

import cosysairsim as airsim
import time
import sys

print("\n" + "="*70)
print("🔍 DESCOBRINDO MESH NAME EXATO DOS DRONES")
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

# Prepara e posiciona drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass

time.sleep(3)

# Posiciona drones bem separados e visíveis
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(10, -5, -10, 3, vehicle_name="Drone3").join()
client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone4").join()
client.moveToPositionAsync(10, 5, -10, 3, vehicle_name="Intruder1").join()
time.sleep(2)

print("\n📊 TESTE 1: Detectar TUDO e ver os nomes")
print("-" * 50)

# Configura para detectar TUDO
camera_name = "front_center"
image_type = airsim.ImageType.Scene

client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera_name, image_type, "*", vehicle_name="Ego")

time.sleep(0.1)

# Detecta
detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")

if detections:
    print(f"Total de detecções: {len(detections)}")
    print("\n🔍 Analisando cada detecção:")

    drone_names = []

    for i, det in enumerate(detections):
        print(f"\n   Detecção {i+1}:")

        # Nome
        name = det.name if hasattr(det, 'name') else "SEM NOME"
        print(f"      Nome: '{name}'")

        # Tamanho da bbox
        if hasattr(det, 'box2D'):
            if hasattr(det.box2D.min, 'x_val'):
                width = det.box2D.max.x_val - det.box2D.min.x_val
                height = det.box2D.max.y_val - det.box2D.min.y_val
            else:
                width = det.box2D.max.x - det.box2D.min.x
                height = det.box2D.max.y - det.box2D.min.y

            print(f"      Tamanho: {width:.0f}x{height:.0f} pixels")

            # Tenta identificar se é drone pelo tamanho
            if 10 < width < 400 and 10 < height < 400:
                print(f"      → Provável DRONE! (tamanho pequeno/médio)")
                drone_names.append(name)
            elif width > 1000 or height > 500:
                print(f"      → Provável CHÃO/MONTANHA (muito grande)")
            else:
                print(f"      → Objeto desconhecido")

        # Posição 3D se disponível
        if hasattr(det, 'relative_pose'):
            x = det.relative_pose.position.x_val
            y = det.relative_pose.position.y_val
            z = det.relative_pose.position.z_val
            print(f"      Posição: ({x:.1f}, {y:.1f}, {z:.1f})")

    if drone_names:
        print(f"\n✅ Prováveis mesh names dos drones: {list(set(drone_names))}")

print("\n📊 TESTE 2: Testar padrões específicos")
print("-" * 50)

# Lista de padrões muito específicos para testar
test_patterns = [
    # Baseado nos nomes dos veículos
    "Drone3",
    "Drone4",
    "Intruder1",
    "Ego",

    # Combinações
    "Drone*",
    "Intruder*",
    "*3*",
    "*4*",

    # Possíveis nomes de mesh/blueprint
    "BP_FlyingPawn",
    "BP_FlyingPawn_C",
    "FlyingPawn",
    "SimpleFlight",
    "Multirotor",

    # Testa se os nomes são os próprios veículos
    "Vehicle*",
    "*Vehicle*",
]

best_pattern = None
best_count = 0

for pattern in test_patterns:
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    time.sleep(0.05)

    detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")
    count = len(detections) if detections else 0

    if count > 0:
        print(f"   ✅ '{pattern}': {count} detecções")

        # Verifica se detectou algo pequeno (provável drone)
        small_objects = 0
        for det in detections:
            if hasattr(det, 'box2D'):
                if hasattr(det.box2D.min, 'x_val'):
                    width = det.box2D.max.x_val - det.box2D.min.x_val
                    height = det.box2D.max.y_val - det.box2D.min.y_val
                else:
                    width = det.box2D.max.x - det.box2D.min.x
                    height = det.box2D.max.y - det.box2D.min.y

                if width < 400 and height < 400:
                    small_objects += 1

        if small_objects > 0:
            print(f"      → {small_objects} objetos pequenos (prováveis drones)")

            if small_objects > best_count:
                best_count = small_objects
                best_pattern = pattern

print("\n" + "="*70)
print("📋 CONCLUSÃO:")
print("="*70)

if best_pattern and best_count > 0:
    print(f"✅ MELHOR PADRÃO ENCONTRADO: '{best_pattern}'")
    print(f"   Detecta {best_count} prováveis drones")
    print(f"\n👉 Use este padrão no dataset generator!")
else:
    print("❌ Não encontrei um padrão que detecte APENAS drones")
    print("\n   Soluções:")
    print("   1. Configure mesh names no Unreal Engine")
    print("   2. Use detecção de TUDO + filtro por tamanho")
    print("   3. Use posições conhecidas dos drones")

# Pousa
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass