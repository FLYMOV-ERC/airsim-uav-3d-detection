#!/usr/bin/env python3
"""
Lista todos os objetos da cena para encontrar os drones
"""

import cosysairsim as airsim
import sys

print("\n🔍 LISTANDO OBJETOS DA CENA")

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

# Tenta listar objetos da cena
if hasattr(client, 'simListSceneObjects'):
    print("\n✅ simListSceneObjects disponível!")
    objects = client.simListSceneObjects()

    if objects:
        print(f"\nTotal de objetos: {len(objects)}")
        print("\nPrimeiros 50 objetos:")
        for i, obj in enumerate(objects[:50]):
            print(f"   {i+1}. {obj}")

        # Procura por padrões de drones
        print("\n🔍 Objetos que podem ser drones:")
        keywords = ['drone', 'Drone', 'intruder', 'Intruder', 'flying', 'Flying',
                   'pawn', 'Pawn', 'BP_', 'vehicle', 'Vehicle']

        for obj in objects:
            for keyword in keywords:
                if keyword.lower() in obj.lower():
                    print(f"   → {obj}")
                    break
    else:
        print("Nenhum objeto encontrado")
else:
    print("❌ simListSceneObjects não disponível")

# Testa se consegue pegar pose de objetos específicos
print("\n🔍 Testando simGetObjectPose com nomes conhecidos:")
test_names = ["Drone3", "Drone4", "Intruder1", "BP_FlyingPawn", "FlyingPawn"]

for name in test_names:
    try:
        pose = client.simGetObjectPose(name)
        print(f"   ✅ '{name}' encontrado! Posição: ({pose.position.x_val:.1f}, {pose.position.y_val:.1f}, {pose.position.z_val:.1f})")
    except:
        pass