#!/usr/bin/env python3
"""
Gerador de dataset FINAL - Usando câmera front_center em 720p
A melhor qualidade disponível sem modificar settings.json
"""

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import math
from tqdm import tqdm

def main():
    print("\n" + "="*60)
    print("🚀 DATASET FINAL - 720p HD (Melhor Disponível)")
    print("="*60)

    # Configuração
    output_dir = Path("dataset_final_720p")
    output_dir.mkdir(exist_ok=True)

    dirs = {
        'rgb': output_dir / 'rgb',
        'rgb_1080p': output_dir / 'rgb_1080p',  # Versão upscaled
        'segmentation': output_dir / 'segmentation',
        'depth': output_dir / 'depth',
        'annotations': output_dir / 'annotations',
        'metadata': output_dir / 'metadata'
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True)

    print(f"\n📁 Dataset em: {output_dir.absolute()}")

    # Conecta
    print("\n🔌 Conectando...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Drones: {vehicles}")

    # Prepara drones
    print("\n🎮 Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
            print(f"   ✅ {vehicle}")
        except Exception as e:
            print(f"   ⚠️ {vehicle}: {e}")

    # Decola
    print("\n🛫 Decolando...")
    for vehicle in vehicles:
        try:
            client.takeoffAsync(vehicle_name=vehicle)
        except:
            pass
    time.sleep(5)

    # Posiciona Ego
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

    # Configuração
    num_frames = 200  # Dataset completo
    frames_captured = 0

    print(f"\n📸 Capturando {num_frames} frames em 720p HD")
    print("   Usando câmera 'front_center' (1280x720)")
    print("   Gerando também versão 1080p upscaled\n")

    with tqdm(total=num_frames, desc="Progresso") as pbar:
        for frame_idx in range(num_frames):
            t = frame_idx * 0.1

            # MOVIMENTO VARIADO DOS DRONES
            num_drones = len(vehicles) - 1
            drone_idx = 0

            for vehicle in vehicles:
                if vehicle == "Ego":
                    # Ego rotaciona e varia altura
                    yaw = (frame_idx * 2) % 360
                    client.rotateToYawAsync(yaw, vehicle_name="Ego")

                    # Varia altura suavemente
                    ego_z = -15 + 5 * math.sin(t * 0.2)
                    client.moveToZAsync(ego_z, 2, vehicle_name="Ego")
                    continue

                # Padrões variados para outros drones
                if frame_idx < 50:
                    # Círculo expandindo
                    angle = (2 * math.pi * drone_idx / num_drones) + t
                    radius = 10 + frame_idx * 0.3
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = -12 + 3 * math.sin(t * 0.5)

                elif frame_idx < 100:
                    # Figura oito
                    angle = t + drone_idx * math.pi / 2
                    x = 20 * math.cos(angle)
                    y = 10 * math.sin(2 * angle)
                    z = -10 + 4 * math.sin(t * 0.3)

                elif frame_idx < 150:
                    # Espiral
                    angle = t * 2 + drone_idx * 2 * math.pi / num_drones
                    radius = 5 + t * 0.5
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = -20 + t * 0.2

                else:
                    # Movimento caótico controlado
                    x = 15 * math.sin(t * (1 + drone_idx * 0.2))
                    y = 15 * math.cos(t * (1.3 + drone_idx * 0.1))
                    z = -12 + 6 * math.sin(t * 0.4 + drone_idx)

                try:
                    client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle)
                except:
                    pass

                drone_idx += 1

            # Pequena pausa para movimento
            time.sleep(0.2)

            # CAPTURA - Sempre usa front_center que é 720p
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
                    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
                ], vehicle_name="Ego")

                # RGB 720p
                if responses[0].image_data_uint8:
                    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    # Salva 720p original
                    rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_file), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                    # Cria versão 1080p com upscaling de qualidade
                    img_1080p = cv2.resize(img_bgr, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

                    # Aplica leve sharpening para melhorar detalhes
                    kernel = np.array([[0, -0.5, 0],
                                      [-0.5, 3, -0.5],
                                      [0, -0.5, 0]])
                    img_1080p = cv2.filter2D(img_1080p, -1, kernel)

                    # Salva 1080p
                    rgb_1080_file = dirs['rgb_1080p'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_1080_file), img_1080p, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                # Segmentação
                if len(responses) > 1 and responses[1].image_data_uint8:
                    img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                    img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                    # Upscale segmentação também
                    img_seg = cv2.resize(img_seg, (1920, 1080), interpolation=cv2.INTER_NEAREST)

                    seg_file = dirs['segmentation'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(seg_file), img_seg)

                    # Gera anotações YOLO
                    generate_yolo_annotations(img_seg, dirs['annotations'], frames_captured)

                # Profundidade
                if len(responses) > 2 and responses[2].image_data_float:
                    img_depth = airsim.list_to_2d_float_array(
                        responses[2].image_data_float,
                        responses[2].width,
                        responses[2].height
                    )

                    # Normalização melhorada
                    img_depth_normalized = np.clip(img_depth / 40.0, 0, 1) * 255
                    img_depth_uint8 = img_depth_normalized.astype(np.uint8)

                    # Upscale e aplica colormap
                    img_depth_uint8 = cv2.resize(img_depth_uint8, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                    img_depth_colored = cv2.applyColorMap(img_depth_uint8, cv2.COLORMAP_TURBO)

                    depth_file = dirs['depth'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(depth_file), img_depth_colored)

                # Metadata
                metadata = {
                    'frame': frames_captured,
                    'timestamp': datetime.now().isoformat(),
                    'resolution': {'width': 1280, 'height': 720, 'upscaled': '1920x1080'},
                    'vehicles': {}
                }

                for vehicle in vehicles:
                    try:
                        state = client.getMultirotorState(vehicle_name=vehicle)
                        metadata['vehicles'][vehicle] = {
                            'position': {
                                'x': float(state.kinematics_estimated.position.x_val),
                                'y': float(state.kinematics_estimated.position.y_val),
                                'z': float(state.kinematics_estimated.position.z_val)
                            }
                        }
                    except:
                        pass

                meta_file = dirs['metadata'] / f"frame_{frames_captured:06d}.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                frames_captured += 1
                pbar.update(1)

            except Exception as e:
                pbar.write(f"⚠️ Erro frame {frame_idx}: {e}")
                continue

    # Pousa
    print("\n🛬 Pousando...")
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

    # ESTATÍSTICAS
    print("\n" + "="*60)
    print("✅ DATASET COMPLETO GERADO!")
    print("="*60)

    rgb_count = len(list(dirs['rgb'].glob("*.png")))
    rgb_1080_count = len(list(dirs['rgb_1080p'].glob("*.png")))
    seg_count = len(list(dirs['segmentation'].glob("*.png")))
    depth_count = len(list(dirs['depth'].glob("*.png")))
    anno_count = len(list(dirs['annotations'].glob("*.txt")))

    print(f"\n📊 Estatísticas:")
    print(f"   Frames capturados: {frames_captured}/{num_frames}")
    print(f"   Imagens RGB 720p: {rgb_count}")
    print(f"   Imagens RGB 1080p (upscaled): {rgb_1080_count}")
    print(f"   Segmentações: {seg_count}")
    print(f"   Mapas de profundidade: {depth_count}")
    print(f"   Anotações YOLO: {anno_count}")

    print(f"\n📁 Dataset em: {output_dir.absolute()}")
    print("\n✨ Use as imagens da pasta 'rgb_1080p' para melhor qualidade!")
    print("   Elas foram upscaled com Lanczos4 + sharpening")

def generate_yolo_annotations(seg_image, anno_dir, frame_idx):
    """Gera anotações YOLO precisas"""
    h, w = seg_image.shape[:2]
    annotations = []

    # Cores típicas de objetos no AirSim
    object_colors = {
        (255, 0, 255): 0,  # Drone - Magenta
        (0, 255, 255): 0,  # Drone - Cyan
        (255, 255, 0): 0,  # Drone - Yellow
        (255, 0, 0): 0,    # Drone - Red
        (0, 255, 0): 1,    # Outro objeto - Green
        (0, 0, 255): 1,    # Outro objeto - Blue
    }

    for color, class_id in object_colors.items():
        lower = np.array(color) - 15
        upper = np.array(color) + 15
        mask = cv2.inRange(seg_image, lower, upper)

        if mask.any():
            # Limpa ruído
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Ignora muito pequeno
                    continue

                x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

                # Formato YOLO
                cx = (x + bbox_w/2) / w
                cy = (y + bbox_h/2) / h
                nw = bbox_w / w
                nh = bbox_h / h

                annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if annotations:
        anno_file = anno_dir / f"frame_{frame_idx:06d}.txt"
        with open(anno_file, 'w') as f:
            f.write('\n'.join(annotations))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")