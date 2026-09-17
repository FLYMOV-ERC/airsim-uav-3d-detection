#!/usr/bin/env python3
"""
Script para verificar EXATAMENTE de onde vem a nuvem de pontos
"""

import cosysairsim as airsim
import numpy as np
import json

print("\n" + "="*60)
print("🔍 VERIFICAÇÃO DO SENSOR LIDAR")
print("="*60)

# Conecta
print("\n🔌 Conectando ao AirSim...")
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()

vehicles = client.listVehicles()
print(f"✅ Drones disponíveis: {vehicles}")

# Verifica configuração do Ego
print("\n📡 VERIFICANDO SENSORES DO DRONE 'Ego':")
print("-"*40)

# Tenta descobrir sensores LiDAR configurados
possible_lidar_names = [
    "LidarSensor1", "LidarSensor2",
    "LidarFront", "LidarBack", "LidarTop", "LidarBottom",
    "Lidar1", "Lidar2",
    "LiDAR", "LIDAR",
    "lidar", "front_lidar"
]

lidar_found = []

for lidar_name in possible_lidar_names:
    try:
        # Tenta pegar dados do LiDAR
        lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")

        if lidar_data and hasattr(lidar_data, 'point_cloud'):
            points = np.array(lidar_data.point_cloud, dtype=np.float32)

            if len(points) > 3:
                points = points.reshape(-1, 3)
                lidar_found.append({
                    'name': lidar_name,
                    'points': len(points),
                    'data': lidar_data
                })
                print(f"✅ '{lidar_name}' ENCONTRADO - {len(points)} pontos")
    except:
        pass

if not lidar_found:
    print("❌ NENHUM sensor LiDAR encontrado no drone Ego!")
else:
    print(f"\n📊 SENSORES LIDAR ENCONTRADOS: {len(lidar_found)}")

    for lidar_info in lidar_found:
        print(f"\n🎯 Sensor: '{lidar_info['name']}'")
        print(f"   Pontos: {lidar_info['points']}")

        # Mostra informações do sensor
        data = lidar_info['data']
        if hasattr(data, 'pose'):
            print(f"   Posição no drone:")
            print(f"      X: {data.pose.position.x_val:.2f}")
            print(f"      Y: {data.pose.position.y_val:.2f}")
            print(f"      Z: {data.pose.position.z_val:.2f}")

        if hasattr(data, 'time_stamp'):
            print(f"   Timestamp: {data.time_stamp}")

# Verifica arquivo de configuração
print("\n📄 VERIFICANDO CONFIGURAÇÃO (settings.json):")
print("-"*40)

# Lê nosso settings.json local
try:
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    if 'Vehicles' in settings and 'Ego' in settings['Vehicles']:
        ego_config = settings['Vehicles']['Ego']

        if 'Sensors' in ego_config:
            print("✅ Sensores configurados no Ego:")
            for sensor_name, sensor_config in ego_config['Sensors'].items():
                sensor_type = sensor_config.get('SensorType', 'Unknown')
                if sensor_type == 6:  # LiDAR
                    print(f"   📡 {sensor_name} (LiDAR)")
                    print(f"      Canais: {sensor_config.get('NumberOfChannels', 'N/A')}")
                    print(f"      Pontos/seg: {sensor_config.get('PointsPerSecond', 'N/A')}")
                    print(f"      FOV Vertical: {sensor_config.get('VerticalFOVLower', 'N/A')}° a {sensor_config.get('VerticalFOVUpper', 'N/A')}°")
                    print(f"      FOV Horizontal: {sensor_config.get('HorizontalFOVStart', 'N/A')}° a {sensor_config.get('HorizontalFOVEnd', 'N/A')}°")
                    print(f"      Posição: X={sensor_config.get('X', 0)}, Y={sensor_config.get('Y', 0)}, Z={sensor_config.get('Z', 0)}")
        else:
            print("⚠️ Nenhum sensor configurado no Ego")
    else:
        print("⚠️ Ego não encontrado na configuração")

except FileNotFoundError:
    print("⚠️ Arquivo settings.json não encontrado localmente")

# Testa captura real
print("\n🎬 TESTE DE CAPTURA REAL:")
print("-"*40)

if lidar_found:
    lidar_name = lidar_found[0]['name']
    print(f"Usando sensor: '{lidar_name}'")

    # Prepara e decola Ego
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")
    client.takeoffAsync(vehicle_name="Ego").join()

    import time
    time.sleep(3)

    # Captura dados
    print("\nCapturando dados LiDAR...")
    lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")

    if lidar_data and hasattr(lidar_data, 'point_cloud'):
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3)

        print(f"\n📍 NUVEM DE PONTOS CAPTURADA:")
        print(f"   Total de pontos: {len(points)}")
        print(f"   Forma do array: {points.shape}")
        print(f"   Alcance X: {points[:,0].min():.1f}m a {points[:,0].max():.1f}m")
        print(f"   Alcance Y: {points[:,1].min():.1f}m a {points[:,1].max():.1f}m")
        print(f"   Alcance Z: {points[:,2].min():.1f}m a {points[:,2].max():.1f}m")

        # Salva amostra
        np.save('amostra_lidar.npy', points)
        print(f"\n💾 Amostra salva em 'amostra_lidar.npy'")

        # Mostra primeiros pontos
        print(f"\n🔍 Primeiros 5 pontos (X, Y, Z):")
        for i in range(min(5, len(points))):
            print(f"   Ponto {i}: [{points[i,0]:.2f}, {points[i,1]:.2f}, {points[i,2]:.2f}]")

    # Pousa
    client.landAsync(vehicle_name="Ego").join()
    client.armDisarm(False, "Ego")
    client.enableApiControl(False, "Ego")

else:
    print("❌ Não há LiDAR configurado no drone!")
    print("\n💡 A nuvem de pontos pode estar sendo gerada:")
    print("   1. De um sensor LiDAR virtual do AirSim")
    print("   2. Convertida do mapa de profundidade (depth map)")
    print("   3. De um sensor configurado no Windows mas não visível aqui")

print("\n" + "="*60)
print("CONCLUSÃO:")
print("="*60)

if lidar_found:
    print(f"✅ SIM! O drone Ego tem {len(lidar_found)} sensor(es) LiDAR acoplado(s)!")
    print(f"   Nome: '{lidar_found[0]['name']}'")
    print(f"   Gerando {lidar_found[0]['points']} pontos 3D por captura")
    print("\n📡 O LiDAR está FISICAMENTE simulado no drone,")
    print("   capturando o ambiente 3D ao redor!")
else:
    print("⚠️ Nenhum LiDAR detectado - verifique configuração no Windows")