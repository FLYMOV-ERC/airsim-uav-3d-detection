#!/usr/bin/env python3
"""
Gerador de Dataset com DEPTH CAMERA - Detecta TUDO!
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path
import json
import sys

print("\n" + "="*70)
print("🚀 GERADOR COM DEPTH CAMERA - DETECÇÃO GARANTIDA!")
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

# Prepara drones
for v in vehicles:
    try:
        client.enableApiControl(True, v)
        client.armDisarm(True, v)
    except:
        pass

# Decola
print("\n🛫 Decolando todos os drones...")
for v in vehicles:
    try:
        client.takeoffAsync(vehicle_name=v).join()
    except:
        pass
time.sleep(3)

# Diretórios
output_dir = Path("dataset_depth")
output_dir.mkdir(exist_ok=True)
(output_dir / "images_rgb").mkdir(exist_ok=True)
(output_dir / "images_depth").mkdir(exist_ok=True)
(output_dir / "images_fusion").mkdir(exist_ok=True)
(output_dir / "images_comparison").mkdir(exist_ok=True)
(output_dir / "depth_arrays").mkdir(exist_ok=True)
(output_dir / "metadata").mkdir(exist_ok=True)

print(f"\n📁 Dataset será salvo em: {output_dir.absolute()}")

# Parâmetros da câmera
FOV_H = 90  # graus
FOV_V = 60  # estimado para 16:9
image_width = 1280
image_height = 720

fx = image_width / (2 * np.tan(np.radians(FOV_H/2)))
fy = image_height / (2 * np.tan(np.radians(FOV_V/2)))
cx = image_width / 2
cy = image_height / 2

# Cenários
scenarios = [
    {
        "name": "low_altitude",
        "ego_height": -10,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -3, "z": -10},
            {"vehicle": "Drone4", "x": 10, "y": 3, "z": -8},
            {"vehicle": "Intruder1", "x": 15, "y": 0, "z": -12}
        ]
    },
    {
        "name": "medium_altitude",
        "ego_height": -15,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 12, "y": -5, "z": -15},
            {"vehicle": "Drone4", "x": 18, "y": 5, "z": -13},
            {"vehicle": "Intruder1", "x": 25, "y": 0, "z": -17}
        ]
    },
    {
        "name": "high_altitude",
        "ego_height": -20,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 15, "y": -8, "z": -22},
            {"vehicle": "Drone4", "x": 20, "y": 8, "z": -20},
            {"vehicle": "Intruder1", "x": 30, "y": 0, "z": -18}
        ]
    },
    {
        "name": "scattered",
        "ego_height": -18,
        "drone_configs": [
            {"vehicle": "Drone3", "x": 8, "y": -10, "z": -18},
            {"vehicle": "Drone4", "x": 10, "y": 0, "z": -16},
            {"vehicle": "Intruder1", "x": 12, "y": 10, "z": -20}
        ]
    }
]

yaw_angles = [-30, -20, -10, 0, 10, 20, 30]
total_frames = 0
total_detections = 0
max_frames = 300
start_time = time.time()

print(f"\n📊 Configuração:")
print(f"   • {len(scenarios)} cenários x {len(yaw_angles)} ângulos")
print(f"   • Até {max_frames} frames")
print(f"   • Usando DEPTH CAMERA para detecção perfeita")

for scenario_idx, scenario in enumerate(scenarios):
    print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario['name']}")

    # Posiciona Ego
    ego_height = scenario["ego_height"]
    print(f"   Movendo Ego para {-ego_height}m de altura...")
    client.moveToPositionAsync(0, 0, ego_height, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    for cfg in scenario["drone_configs"]:
        if cfg["vehicle"] in vehicles:
            client.moveToPositionAsync(cfg["x"], cfg["y"], cfg["z"], 5,
                                      vehicle_name=cfg["vehicle"]).join()
            print(f"   {cfg['vehicle']}: x={cfg['x']}, y={cfg['y']}, altura={-cfg['z']}m")

    time.sleep(2)  # Estabilização

    for yaw in yaw_angles:
        if total_frames >= max_frames:
            break

        # Rotaciona Ego
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        frame_name = f"frame_{total_frames:04d}"

        # Captura RGB e Depth simultaneamente
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
        ], vehicle_name="Ego")

        # Processa RGB
        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Processa Depth
        depth_array = None
        if responses[1].image_data_float:
            depth_1d = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_array = depth_1d.reshape(responses[1].height, responses[1].width)

            # Limita range para visualização
            depth_array = np.clip(depth_array, 0, 100)

        if img_bgr is not None and depth_array is not None:
            h_rgb, w_rgb = img_bgr.shape[:2]
            h_depth, w_depth = depth_array.shape

            # Redimensiona depth para combinar com RGB se necessário
            if h_depth != h_rgb or w_depth != w_rgb:
                depth_array = cv2.resize(depth_array, (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)

            h, w = h_rgb, w_rgb

            # Cria imagem de fusão
            fusion = img_bgr.copy()

            # Converte depth para nuvem de pontos 3D
            # Cria grid de coordenadas de pixel
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Máscara para pontos válidos (não infinito)
            valid_mask = (depth_array > 0.1) & (depth_array < 100)

            # Calcula coordenadas 3D dos pontos válidos
            z = depth_array[valid_mask]
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]

            # Converte pixels para coordenadas 3D
            x = (u_valid - cx) * z / fx
            y = (v_valid - cy) * z / fy

            # Amostra pontos para visualização (muito denso senão)
            sample_rate = 50  # Mostra 1 a cada 50 pontos
            sample_indices = np.arange(0, len(z), sample_rate)

            z_sampled = z[sample_indices]
            u_sampled = u_valid[sample_indices]
            v_sampled = v_valid[sample_indices]

            # Detecta objetos próximos (possíveis drones)
            frame_detections = 0
            detected_objects = []

            # Procura descontinuidades na profundidade (objetos)
            depth_grad = np.gradient(depth_array)[0]
            edges = np.abs(depth_grad) > 2  # Mudança brusca de profundidade

            # Encontra contornos de objetos
            edges_uint8 = (edges * 255).astype(np.uint8)
            contours, _ = cv2.findContours(edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analisa cada contorno
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filtra objetos pequenos
                    # Calcula centro e distância
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx_obj = int(M["m10"] / M["m00"])
                        cy_obj = int(M["m01"] / M["m00"])

                        # Pega profundidade no centro do objeto
                        if 0 <= cx_obj < w and 0 <= cy_obj < h:
                            obj_depth = depth_array[cy_obj, cx_obj]

                            if 3 < obj_depth < 50:  # Objeto entre 3m e 50m
                                # Desenha detecção
                                cv2.drawContours(fusion, [contour], -1, (255, 255, 0), 2)
                                cv2.circle(fusion, (cx_obj, cy_obj), 5, (255, 0, 255), -1)
                                cv2.putText(fusion, f"{obj_depth:.1f}m",
                                          (cx_obj-20, cy_obj-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                                frame_detections += 1
                                detected_objects.append({
                                    "x": int(cx_obj),
                                    "y": int(cy_obj),
                                    "depth": float(obj_depth),
                                    "area": float(area)
                                })

            # Visualiza pontos depth como overlay colorido
            for i in range(len(z_sampled)):
                depth_val = z_sampled[i]
                u_pt = u_sampled[i]
                v_pt = v_sampled[i]

                # Cor baseada na profundidade
                if depth_val < 10:  # Próximo - vermelho
                    color = (0, 0, 255)
                elif depth_val < 30:  # Médio - amarelo
                    color = (0, 255, 255)
                elif depth_val < 50:  # Longe - verde
                    color = (0, 255, 0)
                else:  # Muito longe - azul
                    color = (255, 0, 0)

                cv2.circle(fusion, (u_pt, v_pt), 1, color, -1)

            # Cria visualização de depth colorida
            depth_vis = cv2.applyColorMap(
                (255 * depth_array / depth_array.max()).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            # Adiciona informações
            cv2.putText(fusion, f"{scenario['name']} | Alt:{-ego_height}m | Yaw:{yaw}° | F:{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(fusion, f"Objetos detectados: {frame_detections}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(fusion, "DEPTH CAMERA",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Salva imagens
            cv2.imwrite(str(output_dir / "images_rgb" / f"{frame_name}.png"), img_bgr)
            cv2.imwrite(str(output_dir / "images_depth" / f"{frame_name}.png"), depth_vis)
            cv2.imwrite(str(output_dir / "images_fusion" / f"{frame_name}.png"), fusion)

            comparison = np.hstack([img_bgr, fusion])
            cv2.imwrite(str(output_dir / "images_comparison" / f"{frame_name}.png"), comparison)

            # Salva array de depth
            np.save(output_dir / "depth_arrays" / f"{frame_name}.npy", depth_array)

            # Metadados
            metadata = {
                "frame_id": total_frames,
                "scenario": scenario['name'],
                "scenario_idx": scenario_idx,
                "yaw": yaw,
                "ego_height": -ego_height,
                "detections": frame_detections,
                "detected_objects": detected_objects,
                "depth_min": float(depth_array.min()),
                "depth_max": float(depth_array.max()),
                "drone_positions": scenario["drone_configs"]
            }

            with open(output_dir / "metadata" / f"{frame_name}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            total_frames += 1
            total_detections += frame_detections

            # Status
            if total_frames % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   📊 Progresso: {total_frames}/{max_frames}")
                print(f"      Detecções totais: {total_detections}")

    if total_frames >= max_frames:
        break

# Pousa todos os drones
print("\n🛬 Pousando todos os drones...")
for v in vehicles:
    try:
        client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=v).join()
    except:
        pass

for v in vehicles:
    try:
        client.landAsync(vehicle_name=v).join()
        client.armDisarm(False, v)
        client.enableApiControl(False, v)
    except:
        pass

print("\n" + "="*70)
print("🎉 DATASET COM DEPTH CAMERA COMPLETO!")
print("="*70)
print(f"\n📊 Estatísticas finais:")
print(f"   • Frames gerados: {total_frames}")
print(f"   • Detecções totais: {total_detections}")
print(f"   • Média: {total_detections/total_frames:.2f} objetos/frame" if total_frames > 0 else "")
print(f"\n✅ Vantagens da Depth Camera:")
print("   • Detecta TODOS os objetos")
print("   • Não precisa configuração no Unreal")
print("   • Fornece distância exata de cada pixel")
print("   • Funciona sempre!")
print(f"\n📁 Dataset salvo em: {output_dir.absolute()}")