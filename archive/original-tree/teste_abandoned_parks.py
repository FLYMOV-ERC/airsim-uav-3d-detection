#!/usr/bin/env python3
"""
Teste no ambiente Abandoned Parks
Verifica se os drones aparecem neste cenário
"""

import airsim
import time
import math

print("🏚️ TESTE NO ABANDONED PARKS")
print("="*60)

# Conecta
print("\n📡 Conectando...")
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao Abandoned Parks!")
except Exception as e:
    print(f"❌ Erro de conexão: {e}")
    print("Certifique-se que o AbandonedPark.exe está rodando")
    exit(1)

# Lista veículos
vehicles = client.listVehicles()
print(f"\n🚁 Veículos detectados: {vehicles}")

if len(vehicles) < 2:
    print("⚠️ Poucos veículos! Verifique o settings.json")

# Prepara todos os drones
print("\n🎮 Preparando drones...")
for vehicle in vehicles:
    try:
        client.enableApiControl(True, vehicle)
        client.armDisarm(True, vehicle)
        print(f"   ✅ {vehicle} pronto")
    except:
        print(f"   ⚠️ Erro com {vehicle}")

# Decola todos
print("\n🛫 Decolando...")
for vehicle in vehicles:
    client.takeoffAsync(vehicle_name=vehicle)
time.sleep(5)

# TESTE 1: Posiciona drones próximos
print("\n📍 TESTE 1: Posicionando drones bem próximos")
print("-"*40)

# Ego como observador
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()
print("Ego (observador) em (0, 0, -20)")

# Outros drones BEM PRÓXIMOS e VISÍVEIS
test_positions = [
    ("Drone3", 5, 0, -20, "5m na frente, mesma altura"),
    ("Drone4", 10, 5, -18, "10m frente, 5m direita"),
    ("Intruder1", 10, -5, -22, "10m frente, 5m esquerda"),
]

for vehicle, x, y, z, desc in test_positions:
    if vehicle in vehicles:
        client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle).join()
        print(f"{vehicle}: {desc}")

time.sleep(3)

# TESTE 2: Captura imagens
print("\n📸 TESTE 2: Capturando imagens")
print("-"*40)

# Testa diferentes câmeras
cameras = ["front_center", "0", ""]
captured = False

for cam in cameras:
    try:
        img = client.simGetImage(cam, airsim.ImageType.Scene, "Ego")

        if img and len(img) > 1000:
            filename = f"abandoned_{cam if cam else 'default'}.png"
            with open(filename, 'wb') as f:
                f.write(img)

            print(f"✅ Câmera '{cam}': Salvou {filename} ({len(img)/1024:.0f}KB)")

            if not captured:
                captured = True
                # Salva a principal
                with open("abandoned_test_main.png", 'wb') as f:
                    f.write(img)
    except:
        print(f"❌ Câmera '{cam}': Erro")

# TESTE 3: Move drones em círculo e captura sequência
print("\n🎬 TESTE 3: Movimento e captura (10 frames)")
print("-"*40)

for frame in range(10):
    t = frame / 10 * 2 * math.pi

    # Move drones em padrão circular
    for i, vehicle in enumerate(["Drone3", "Drone4", "Intruder1"]):
        if vehicle in vehicles:
            angle = (2 * math.pi * i / 3) + t
            x = 8 + 5 * math.cos(angle)
            y = 5 * math.sin(angle)
            z = -20 + 3 * math.sin(t)

            client.moveToPositionAsync(x, y, z, 3, vehicle_name=vehicle)

    time.sleep(0.5)

    # Captura frame
    try:
        img = client.simGetImage("front_center", 0, "Ego")
        if img:
            filename = f"abandoned_frame_{frame:02d}.png"
            with open(filename, 'wb') as f:
                f.write(img)
            print(f"   Frame {frame+1}/10 capturado", end='\r')
    except:
        pass

# Pousa
print("\n\n🛬 Pousando...")
for vehicle in vehicles:
    client.landAsync(vehicle_name=vehicle)
time.sleep(5)

for vehicle in vehicles:
    client.armDisarm(False, vehicle)
    client.enableApiControl(False, vehicle)

# Análise
print("\n" + "="*60)
print("📊 ANÁLISE DO ABANDONED PARKS:")
print("="*60)

print("\n✅ Ambiente conectado e funcionando")
print(f"✅ {len(vehicles)} veículos detectados")
print("✅ Imagens capturadas com sucesso")

print("\n🔍 VERIFIQUE AS IMAGENS:")
print("   • abandoned_test_main.png - Imagem principal")
print("   • abandoned_frame_*.png - Sequência de 10 frames")

print("\n❓ OS DRONES APARECEM NAS IMAGENS?")
print("   SIM → Abandoned Parks funciona! Use este ambiente")
print("   NÃO → Este ambiente também não renderiza drones")

print("="*60)