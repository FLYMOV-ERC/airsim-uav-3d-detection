#!/usr/bin/env python3
"""
Testa detecção usando os nomes EXATOS dos drones
"""

import cosysairsim as airsim
import time

print("\n✅ TESTE COM NOMES EXATOS DOS DRONES")

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
vehicles = client.listVehicles()

# Prepara drones
for v in vehicles:
    client.enableApiControl(True, v)
    client.armDisarm(True, v)
    client.takeoffAsync(vehicle_name=v).join()

time.sleep(3)

# Posiciona
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(10, -3, -10, 3, vehicle_name="Drone3").join()
client.moveToPositionAsync(10, 0, -10, 3, vehicle_name="Drone4").join()
client.moveToPositionAsync(10, 3, -10, 3, vehicle_name="Intruder1").join()
time.sleep(2)

camera_name = "front_center"
image_type = airsim.ImageType.Scene

print("\n📊 Testando padrões específicos:")

# Teste 1: Adicionar cada drone individualmente
print("\n1. Adicionando cada drone específico:")
client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")

# Adiciona os 3 drones
client.simAddDetectionFilterMeshName(camera_name, image_type, "Drone3", vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera_name, image_type, "Drone4", vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera_name, image_type, "Intruder1", vehicle_name="Ego")

time.sleep(0.1)
detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")
print(f"   Resultado: {len(detections) if detections else 0} detecções")

if detections:
    for det in detections:
        if hasattr(det, 'name'):
            print(f"      - {det.name}")

# Teste 2: Padrão com wildcard
print("\n2. Testando padrões com wildcard:")

patterns = ["Drone*", "Intruder*", "*rone*", "*ruder*"]

for pattern in patterns:
    client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
    client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
    client.simAddDetectionFilterMeshName(camera_name, image_type, pattern, vehicle_name="Ego")

    time.sleep(0.1)
    detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")
    print(f"   '{pattern}': {len(detections) if detections else 0} detecções")

# Teste 3: Combinação
print("\n3. Combinando padrões:")
client.simClearDetectionMeshNames(camera_name, image_type, vehicle_name="Ego")
client.simSetDetectionFilterRadius(camera_name, image_type, 10000, vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera_name, image_type, "Drone*", vehicle_name="Ego")
client.simAddDetectionFilterMeshName(camera_name, image_type, "Intruder*", vehicle_name="Ego")

time.sleep(0.1)
detections = client.simGetDetections(camera_name, image_type, vehicle_name="Ego")
print(f"   'Drone* + Intruder*': {len(detections) if detections else 0} detecções")

# Pousa
for v in vehicles:
    client.landAsync(vehicle_name=v).join()
    client.armDisarm(False, v)
    client.enableApiControl(False, v)

print("\n✅ Teste concluído!")