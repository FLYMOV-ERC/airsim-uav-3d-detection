#!/usr/bin/env python3
"""
Script simplificado para garantir detecção de drones no LiDAR
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
import json
from pathlib import Path

def main():
    print("\n🎯 TESTE DE DETECÇÃO LIDAR SIMPLIFICADO")
    print("="*50)

    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"Drones: {vehicles}")

    # Prepara todos os drones
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except:
            pass

    # Decola
    for v in vehicles:
        try:
            client.takeoffAsync(vehicle_name=v)
        except:
            pass
    time.sleep(5)

    # Output dir
    output = Path("teste_deteccao")
    output.mkdir(exist_ok=True)

    print("\n📍 Posicionando drones estrategicamente...")

    # Ego drone no centro
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

    # Outros drones: colocamos MUITO próximos e em diferentes alturas
    # Drone3: muito próximo e baixo (onde sabemos que LiDAR detecta chão)
    client.moveToPositionAsync(5, 0, -0.5, 5, vehicle_name="Drone3").join()
    print("  Drone3: 5m frente, 0.5m altura (nível do chão)")

    # Drone4: próximo e médio
    client.moveToPositionAsync(8, 3, -8, 5, vehicle_name="Drone4").join()
    print("  Drone4: 8m frente, 8m altura")

    # Intruder1: próximo e alto
    client.moveToPositionAsync(10, -3, -15.5, 5, vehicle_name="Intruder1").join()
    print("  Intruder1: 10m frente, 15.5m altura (nível dos prédios)")

    time.sleep(2)

    print("\n📡 Capturando LiDAR...")

    # Captura LiDAR
    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    if not lidar_data or len(lidar_data.point_cloud) < 3:
        print("❌ Sem dados do LiDAR")
        return

    # Converte pontos
    points_raw = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    print(f"Total de pontos: {len(points_raw)}")

    # Analisa distribuição de alturas
    z_vals = points_raw[:,2]
    print(f"\nDistribuição de Z (altura):")
    print(f"  Min: {z_vals.min():.1f}m, Max: {z_vals.max():.1f}m")

    # Histograma de alturas
    hist, edges = np.histogram(z_vals, bins=20)
    print("\nPontos por faixa de altura:")
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"  {edges[i]:.1f} a {edges[i+1]:.1f}m: {hist[i]} pontos")

    # Filtra pontos frontais (FOV da câmera)
    x_positive = points_raw[:,0] > 0
    angles = np.degrees(np.arctan2(points_raw[:,1], points_raw[:,0]))
    in_fov = (angles > -45) & (angles < 45)
    points_fov = points_raw[x_positive & in_fov]

    print(f"\nPontos no FOV da câmera: {len(points_fov)}")

    # Obtém posições dos drones
    ego_state = client.getMultirotorState(vehicle_name="Ego")
    ego_pos = np.array([
        ego_state.kinematics_estimated.position.x_val,
        ego_state.kinematics_estimated.position.y_val,
        ego_state.kinematics_estimated.position.z_val
    ])

    print("\n🎯 DETECÇÃO DE DRONES:")
    print("-"*40)

    detections = []
    for v in vehicles:
        if v == "Ego":
            continue

        state = client.getMultirotorState(vehicle_name=v)
        drone_pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])

        # Posição relativa ao Ego
        rel_pos = drone_pos - ego_pos

        # Calcula distâncias dos pontos ao drone
        if len(points_fov) > 0:
            distances = np.linalg.norm(points_fov - rel_pos, axis=1)

            # Testa diferentes raios
            for radius in [1, 2, 3, 5, 10]:
                near_points = np.sum(distances < radius)
                if near_points > 0:
                    print(f"{v}:")
                    print(f"  Posição relativa: ({rel_pos[0]:.1f}, {rel_pos[1]:.1f}, {rel_pos[2]:.1f})m")
                    print(f"  Altura absoluta: {-drone_pos[2]:.1f}m")
                    print(f"  Pontos em raio {radius}m: {near_points}")
                    print(f"  Distância mínima ao ponto mais próximo: {distances.min():.2f}m")

                    # Se detectou, salva visualização
                    if near_points >= 3:
                        print(f"  ✅ DETECTADO! ({near_points} pontos)")
                        detections.append(v)
                    break
            else:
                print(f"{v}: ❌ NÃO DETECTADO (distância mínima: {distances.min():.1f}m)")

    # Visualização
    print("\n📊 Criando visualização...")

    # Vista de cima
    fig_size = (800, 600)
    img = np.zeros((fig_size[1], fig_size[0], 3), dtype=np.uint8)

    if len(points_fov) > 0:
        # Normaliza coordenadas para pixels
        x = points_fov[:, 0]
        y = points_fov[:, 1]

        x_range = 50  # Visualiza até 50m
        y_range = 50

        px = ((x / x_range) * fig_size[0] / 2 + fig_size[0] / 2).astype(int)
        py = ((y / y_range) * fig_size[1] / 2 + fig_size[1] / 2).astype(int)

        # Desenha pontos
        for i in range(len(points_fov)):
            if 0 <= px[i] < fig_size[0] and 0 <= py[i] < fig_size[1]:
                # Cor baseada na altura
                if points_fov[i, 2] < 2:  # Baixo
                    color = (255, 0, 0)  # Azul
                elif points_fov[i, 2] > 14:  # Alto
                    color = (0, 0, 255)  # Vermelho
                else:  # Médio
                    color = (0, 255, 0)  # Verde

                cv2.circle(img, (px[i], py[i]), 1, color, -1)

        # Marca posições dos drones
        for v in vehicles:
            if v == "Ego":
                # Ego no centro
                cv2.circle(img, (fig_size[0]//2, fig_size[1]//2), 10, (255, 255, 255), 2)
                cv2.putText(img, "EGO", (fig_size[0]//2-20, fig_size[1]//2-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                state = client.getMultirotorState(vehicle_name=v)
                drone_pos = np.array([
                    state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val
                ])
                rel_pos = drone_pos - ego_pos

                dpx = int((rel_pos[0] / x_range) * fig_size[0] / 2 + fig_size[0] / 2)
                dpy = int((rel_pos[1] / y_range) * fig_size[1] / 2 + fig_size[1] / 2)

                if 0 <= dpx < fig_size[0] and 0 <= dpy < fig_size[1]:
                    color = (0, 255, 0) if v in detections else (0, 0, 255)
                    cv2.circle(img, (dpx, dpy), 8, color, 2)
                    cv2.putText(img, v, (dpx-30, dpy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Salva visualização
    cv2.imwrite(str(output / "lidar_visualization.png"), img)

    # Salva pontos
    np.save(output / "points_raw.npy", points_raw)
    np.save(output / "points_fov.npy", points_fov)

    # Resumo
    print("\n" + "="*50)
    print("📊 RESUMO:")
    print(f"  Total de drones: {len(vehicles)-1}")
    print(f"  Drones detectados: {len(detections)}")
    print(f"  Taxa de detecção: {len(detections)/(len(vehicles)-1)*100:.0f}%")

    if len(detections) > 0:
        print(f"\n✅ SUCESSO! Detectados: {', '.join(detections)}")
    else:
        print("\n❌ Nenhum drone detectado. Possíveis causas:")
        print("  1. LiDAR não tem resolução suficiente")
        print("  2. Drones estão em alturas sem cobertura LiDAR")
        print("  3. Drones são muito pequenos para o LiDAR detectar")

    # Pousa
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
        except:
            pass

if __name__ == "__main__":
    main()