#!/usr/bin/env python3
"""
Diagnóstico: Por que os drones não aparecem nas imagens?
"""

import airsim
import time
import math

print("🔍 DIAGNÓSTICO DE VISIBILIDADE DOS DRONES\n")

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
drones = client.listVehicles()
print(f"✅ Conectado! Drones: {drones}\n")

# Prepara e decola
for drone in drones:
    client.enableApiControl(True, drone)
    client.armDisarm(True, drone)
    client.takeoffAsync(vehicle_name=drone)

time.sleep(5)

# TESTE 1: Verificar posições atuais
print("📍 TESTE 1: Posições atuais dos drones:")
print("-"*50)
for drone in drones:
    state = client.getMultirotorState(vehicle_name=drone)
    pos = state.kinematics_estimated.position
    print(f"{drone:12} -> X:{pos.x_val:7.2f}, Y:{pos.y_val:7.2f}, Z:{pos.z_val:7.2f}")

# TESTE 2: Colocar drones bem próximos e na frente do Ego
print("\n📍 TESTE 2: Posicionando drones BEM NA FRENTE do Ego...")
print("-"*50)

# Ego fica parado
client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()
print("Ego posicionado em (0, 0, -20)")

# Coloca outros drones BEM NA FRENTE
positions = {
    "Intruder1": (10, 0, -20),    # 10m na frente, mesma altura
    "Intruder2": (15, 5, -18),    # 15m na frente, 5m direita
    "Intruder3": (15, -5, -22),   # 15m na frente, 5m esquerda
    "Intruder4": (20, 0, -20),    # 20m na frente
}

for drone, pos in positions.items():
    if drone in drones:
        client.moveToPositionAsync(pos[0], pos[1], pos[2], 5, vehicle_name=drone)
        print(f"{drone} -> indo para {pos}")

time.sleep(5)

# TESTE 3: Capturar com diferentes câmeras
print("\n📸 TESTE 3: Capturando imagens...")
print("-"*50)

cameras_to_test = [
    ("0", "Câmera padrão (0)"),
    ("front_center", "Câmera frontal"),
    ("", "Câmera vazia"),
]

for cam_id, cam_name in cameras_to_test:
    try:
        png = client.simGetImage(cam_id, airsim.ImageType.Scene, vehicle_name="Ego")
        if png and len(png) > 1000:
            filename = f"teste_{cam_id if cam_id else 'default'}.png"
            with open(filename, 'wb') as f:
                f.write(png)
            print(f"✅ {cam_name}: Salvou {len(png)} bytes -> {filename}")
        else:
            print(f"❌ {cam_name}: Sem dados")
    except Exception as e:
        print(f"❌ {cam_name}: Erro - {e}")

# TESTE 4: Verificar distâncias
print("\n📏 TESTE 4: Distâncias entre Ego e outros drones:")
print("-"*50)
ego_state = client.getMultirotorState(vehicle_name="Ego")
ego_pos = ego_state.kinematics_estimated.position

for drone in drones:
    if drone != "Ego":
        state = client.getMultirotorState(vehicle_name=drone)
        pos = state.kinematics_estimated.position

        dx = pos.x_val - ego_pos.x_val
        dy = pos.y_val - ego_pos.y_val
        dz = pos.z_val - ego_pos.z_val
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        print(f"{drone:12} -> Distância: {dist:6.2f}m (ΔX:{dx:6.2f}, ΔY:{dy:6.2f}, ΔZ:{dz:6.2f})")

# TESTE 5: Mover um drone para MUITO PERTO
print("\n📍 TESTE 5: Colocando Intruder1 MUITO PERTO...")
print("-"*50)
client.moveToPositionAsync(5, 0, -20, 5, vehicle_name="Intruder1").join()
print("Intruder1 movido para apenas 5m na frente!")

time.sleep(2)

# Captura final
png = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name="Ego")
if png:
    with open("teste_muito_perto.png", 'wb') as f:
        f.write(png)
    print("📸 Salvou: teste_muito_perto.png")

# Pousa
print("\n🛬 Pousando...")
for drone in drones:
    client.landAsync(vehicle_name=drone)

print("\n" + "="*60)
print("ANÁLISE:")
print("="*60)
print("Se os drones não aparecem nas imagens:")
print("1. Câmera pode estar apontando para direção errada")
print("2. Drones podem estar fora do FOV (campo de visão)")
print("3. Problema de renderização no ambiente")
print("4. Drones muito pequenos/distantes")
print("\n🔍 Verifique as imagens geradas:")
print("   - teste_front_center.png")
print("   - teste_muito_perto.png")