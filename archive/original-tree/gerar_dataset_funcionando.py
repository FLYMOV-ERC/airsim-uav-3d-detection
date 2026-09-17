#!/usr/bin/env python3
"""
Script FUNCIONANDO para gerar dataset com drones no AirSim
Usa Cosys AirSim (Colosseum) que é compatível com AirSim v4
"""

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import math

def main():
    print("\n" + "="*60)
    print("🚀 GERADOR DE DATASET - AIRSIM V4 (COSYS/COLOSSEUM)")
    print("="*60)

    # Configuração
    output_dir = Path("dataset_drones")
    output_dir.mkdir(exist_ok=True)

    # Cria subdiretórios
    dirs = {
        'rgb': output_dir / 'rgb',
        'segmentation': output_dir / 'segmentation',
        'depth': output_dir / 'depth',
        'annotations': output_dir / 'annotations',
        'metadata': output_dir / 'metadata'
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True)

    print(f"\n📁 Salvando dataset em: {output_dir.absolute()}")

    # Conecta ao AirSim
    print("\n🔌 Conectando ao AirSim...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Drones: {vehicles}")

    # Prepara todos os drones
    print("\n🎮 Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
            print(f"   ✅ {vehicle} pronto")
        except Exception as e:
            print(f"   ⚠️ {vehicle}: {e}")

    # Decola todos
    print("\n🛫 Decolando drones...")
    for vehicle in vehicles:
        try:
            client.takeoffAsync(vehicle_name=vehicle)
        except:
            pass

    time.sleep(5)

    # Move Ego para posição de observação
    print("📍 Posicionando Ego como observador...")
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

    # Configuração de captura
    num_frames = 100  # Número de frames para capturar
    frames_captured = 0

    print(f"\n📸 Iniciando captura de {num_frames} frames...")
    print("   Os drones vão voar em padrões variados")

    for frame_idx in range(num_frames):
        # Calcula tempo para movimento suave
        t = frame_idx * 0.1

        # MOVIMENTO DOS DRONES
        # Padrão circular com variação de altura
        num_drones = len(vehicles) - 1  # Excluindo Ego
        drone_idx = 0

        for vehicle in vehicles:
            if vehicle == "Ego":
                # Ego rotaciona lentamente para observar
                yaw_angle = (frame_idx * 2) % 360
                client.rotateToYawAsync(yaw_angle, vehicle_name="Ego")
                continue

            # Outros drones voam em padrões
            if frame_idx < 33:
                # Padrão circular
                angle = (2 * math.pi * drone_idx / num_drones) + t
                radius = 20
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = -15 + 5 * math.sin(t * 0.5)

            elif frame_idx < 66:
                # Padrão em linha
                spacing = 15
                x = (drone_idx - num_drones/2) * spacing
                y = 25 + 10 * math.sin(t)
                z = -12 + 3 * math.sin(t * 0.7)

            else:
                # Padrão tipo enxame
                center_x = 15 * math.cos(t * 0.3)
                center_y = 15 * math.sin(t * 0.3)
                offset_angle = 2 * math.pi * drone_idx / num_drones
                x = center_x + 8 * math.cos(offset_angle + t)
                y = center_y + 8 * math.sin(offset_angle + t)
                z = -15 + 4 * math.sin(t * 0.5 + offset_angle)

            try:
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle)
            except:
                pass

            drone_idx += 1

        # Aguarda movimento
        time.sleep(0.3)

        # CAPTURA DE IMAGENS
        print(f"\r📸 Capturando frame {frame_idx+1}/{num_frames}...", end="", flush=True)

        try:
            # Captura múltiplos tipos de imagem
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),        # RGB
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False), # Segmentação
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)    # Profundidade
            ], vehicle_name="Ego")

            # Processa RGB
            if responses[0].image_data_uint8:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # Redimensiona para tamanho maior se necessário
                if responses[0].width < 640:
                    img_bgr = cv2.resize(img_bgr, (1280, 720))

                rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(rgb_file), img_bgr)

            # Processa Segmentação
            if len(responses) > 1 and responses[1].image_data_uint8:
                img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                if responses[1].width < 640:
                    img_seg = cv2.resize(img_seg, (1280, 720))

                seg_file = dirs['segmentation'] / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(seg_file), img_seg)

                # Gera anotações YOLO a partir da segmentação
                generate_yolo_annotations(img_seg, dirs['annotations'], frames_captured)

            # Processa Profundidade
            if len(responses) > 2 and responses[2].image_data_float:
                img_depth = airsim.list_to_2d_float_array(
                    responses[2].image_data_float,
                    responses[2].width,
                    responses[2].height
                )
                # Normaliza para visualização
                img_depth_normalized = np.clip(img_depth / 100.0, 0, 1) * 255
                img_depth_uint8 = img_depth_normalized.astype(np.uint8)

                if responses[2].width < 640:
                    img_depth_uint8 = cv2.resize(img_depth_uint8, (1280, 720))

                depth_file = dirs['depth'] / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(depth_file), img_depth_uint8)

            # Salva metadata
            metadata = {
                'frame': frames_captured,
                'timestamp': datetime.now().isoformat(),
                'vehicles': {}
            }

            for vehicle in vehicles:
                try:
                    state = client.getMultirotorState(vehicle_name=vehicle)
                    metadata['vehicles'][vehicle] = {
                        'position': {
                            'x': state.kinematics_estimated.position.x_val,
                            'y': state.kinematics_estimated.position.y_val,
                            'z': state.kinematics_estimated.position.z_val
                        },
                        'velocity': {
                            'x': state.kinematics_estimated.linear_velocity.x_val,
                            'y': state.kinematics_estimated.linear_velocity.y_val,
                            'z': state.kinematics_estimated.linear_velocity.z_val
                        }
                    }
                except:
                    pass

            meta_file = dirs['metadata'] / f"frame_{frames_captured:06d}.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            frames_captured += 1

        except Exception as e:
            print(f"\n   ⚠️ Erro no frame {frame_idx}: {e}")
            continue

        # Status a cada 10 frames
        if (frame_idx + 1) % 10 == 0:
            print(f"\n   ✅ Progresso: {frame_idx+1}/{num_frames} frames ({frames_captured} capturados com sucesso)")

    # FINALIZAÇÃO
    print(f"\n\n🛬 Pousando todos os drones...")
    for vehicle in vehicles:
        try:
            client.landAsync(vehicle_name=vehicle)
        except:
            pass

    time.sleep(5)

    for vehicle in vehicles:
        try:
            client.armDisarm(False, vehicle)
            client.enableApiControl(False, vehicle)
        except:
            pass

    # ESTATÍSTICAS FINAIS
    print("\n" + "="*60)
    print("✅ DATASET GERADO COM SUCESSO!")
    print("="*60)

    rgb_count = len(list(dirs['rgb'].glob("*.png")))
    seg_count = len(list(dirs['segmentation'].glob("*.png")))
    depth_count = len(list(dirs['depth'].glob("*.png")))
    anno_count = len(list(dirs['annotations'].glob("*.txt")))
    meta_count = len(list(dirs['metadata'].glob("*.json")))

    print(f"\n📊 Estatísticas do Dataset:")
    print(f"   Frames totais capturados: {frames_captured}/{num_frames}")
    print(f"   Imagens RGB: {rgb_count}")
    print(f"   Máscaras de segmentação: {seg_count}")
    print(f"   Mapas de profundidade: {depth_count}")
    print(f"   Anotações YOLO: {anno_count}")
    print(f"   Arquivos de metadata: {meta_count}")

    print(f"\n📁 Dataset completo em: {output_dir.absolute()}")

    if frames_captured > 0:
        print("\n🎉 SUCESSO! Dataset pronto para treino!")
        print("   Use as imagens RGB e anotações YOLO para treinar detectores")
        print("   As máscaras de segmentação podem ser usadas para validação")
        print("   Os mapas de profundidade fornecem informação 3D")
    else:
        print("\n⚠️ Nenhuma imagem foi capturada")

def generate_yolo_annotations(seg_image, anno_dir, frame_idx):
    """Gera anotações YOLO a partir da imagem de segmentação"""
    h, w = seg_image.shape[:2]
    annotations = []

    # Cores típicas de drones na segmentação (ajuste conforme necessário)
    drone_colors = [
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
    ]

    for color in drone_colors:
        # Cria máscara para a cor
        lower = np.array(color) - 20
        upper = np.array(color) + 20
        mask = cv2.inRange(seg_image, lower, upper)

        if mask.any():
            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Ignora objetos muito pequenos
                    continue

                x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

                # Formato YOLO: class_id center_x center_y width height (normalizado)
                cx = (x + bbox_w/2) / w
                cy = (y + bbox_h/2) / h
                nw = bbox_w / w
                nh = bbox_h / h

                # Classe 0 = drone
                annotations.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if annotations:
        anno_file = anno_dir / f"frame_{frame_idx:06d}.txt"
        with open(anno_file, 'w') as f:
            f.write('\n'.join(annotations))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro fatal: {e}")