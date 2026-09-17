#!/usr/bin/env python3
"""
Verifica o sistema de coordenadas do AirSim
E valida se Y negativo está correto
"""

import cosysairsim as airsim
import numpy as np
import time

print("\n" + "="*70)
print("🧭 VERIFICANDO SISTEMA DE COORDENADAS DO AIRSIM")
print("="*70)

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
vehicles = client.listVehicles()

print("\n📊 Sistema de Coordenadas do AirSim:")
print("-" * 50)
print("AirSim usa NED (North-East-Down):")
print("   • X = Norte (frente)")
print("   • Y = Leste (direita)")
print("   • Z = Baixo (down)")
print("\nNa câmera:")
print("   • X = Frente")
print("   • Y = Direita")
print("   • Z = Baixo")

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass

time.sleep(3)

print("\n🔍 TESTE DE POSICIONAMENTO:")
print("-" * 50)

# Teste 1: Drone à ESQUERDA (Y negativo no mundo)
print("\nTeste 1: Drone3 à ESQUERDA da câmera")
client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Ego").join()
client.moveToPositionAsync(10, -5, -10, 3, vehicle_name="Drone3").join()  # Y=-5 (esquerda)
time.sleep(2)

ego_pose = client.simGetVehiclePose(vehicle_name="Ego")
drone3_pose = client.simGetVehiclePose(vehicle_name="Drone3")

ego_pos = [ego_pose.position.x_val, ego_pose.position.y_val, ego_pose.position.z_val]
drone3_pos = [drone3_pose.position.x_val, drone3_pose.position.y_val, drone3_pose.position.z_val]

relative_pos = [drone3_pos[0] - ego_pos[0],
                drone3_pos[1] - ego_pos[1],
                drone3_pos[2] - ego_pos[2]]

print(f"   Ego posição: X={ego_pos[0]:.1f}, Y={ego_pos[1]:.1f}, Z={ego_pos[2]:.1f}")
print(f"   Drone3 posição: X={drone3_pos[0]:.1f}, Y={drone3_pos[1]:.1f}, Z={drone3_pos[2]:.1f}")
print(f"   Posição relativa: X={relative_pos[0]:.1f}, Y={relative_pos[1]:.1f}, Z={relative_pos[2]:.1f}")

if relative_pos[1] < 0:
    print("   ✅ Y negativo = Drone à ESQUERDA (correto no NED)")
else:
    print("   ❌ Y positivo = Drone à DIREITA (incorreto)")

# Teste 2: Drone à DIREITA (Y positivo no mundo)
print("\nTeste 2: Drone4 à DIREITA da câmera")
client.moveToPositionAsync(10, 5, -10, 3, vehicle_name="Drone4").join()  # Y=+5 (direita)
time.sleep(2)

drone4_pose = client.simGetVehiclePose(vehicle_name="Drone4")
drone4_pos = [drone4_pose.position.x_val, drone4_pose.position.y_val, drone4_pose.position.z_val]
relative_pos2 = [drone4_pos[0] - ego_pos[0],
                 drone4_pos[1] - ego_pos[1],
                 drone4_pos[2] - ego_pos[2]]

print(f"   Drone4 posição: X={drone4_pos[0]:.1f}, Y={drone4_pos[1]:.1f}, Z={drone4_pos[2]:.1f}")
print(f"   Posição relativa: X={relative_pos2[0]:.1f}, Y={relative_pos2[1]:.1f}, Z={relative_pos2[2]:.1f}")

if relative_pos2[1] > 0:
    print("   ✅ Y positivo = Drone à DIREITA (correto no NED)")
else:
    print("   ❌ Y negativo = Drone à ESQUERDA (incorreto)")

# Teste 3: Verificar na imagem
print("\n🖼️ VERIFICAÇÃO NA IMAGEM 2D:")
print("-" * 50)

# Parâmetros da câmera
image_width = 1280
image_height = 720
FOV_H = 90
fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = fx
cx = image_width / 2
cy = image_height / 2

# Projeta Drone3 (à esquerda)
if relative_pos[0] > 0:  # Se está na frente
    u1 = fx * relative_pos[1] / relative_pos[0] + cx
    v1 = fy * (-relative_pos[2]) / relative_pos[0] + cy

    print(f"Drone3 (Y={relative_pos[1]:.1f}):")
    print(f"   Pixel X = {u1:.0f} (de 0 a {image_width})")
    if u1 < cx:
        print(f"   ✅ Está à ESQUERDA da imagem (X < {cx})")
    else:
        print(f"   ❌ Está à DIREITA da imagem (X > {cx})")

# Projeta Drone4 (à direita)
if relative_pos2[0] > 0:  # Se está na frente
    u2 = fx * relative_pos2[1] / relative_pos2[0] + cx
    v2 = fy * (-relative_pos2[2]) / relative_pos2[0] + cy

    print(f"\nDrone4 (Y={relative_pos2[1]:.1f}):")
    print(f"   Pixel X = {u2:.0f} (de 0 a {image_width})")
    if u2 > cx:
        print(f"   ✅ Está à DIREITA da imagem (X > {cx})")
    else:
        print(f"   ❌ Está à ESQUERDA da imagem (X < {cx})")

print("\n" + "="*70)
print("📋 CONCLUSÃO:")
print("="*70)
print("""
No AirSim (sistema NED):
   • Y negativo = ESQUERDA
   • Y positivo = DIREITA
   • Z negativo = PARA CIMA (altitude)
   • Z positivo = PARA BAIXO

Isso é CORRETO e esperado!

Para treinar PointNet:
   ✅ As coordenadas 3D estão corretas
   ✅ Y negativo para drones à esquerda é normal
   ✅ O modelo vai aprender essa convenção

Se quiser converter para outro sistema:
   • ROS usa ENU (East-North-Up)
   • Unity usa sistema left-handed
   • Mas para treinar, use como está!
""")

# Pousa
for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass