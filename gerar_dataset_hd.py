#!/usr/bin/env python3
"""
Script para gerar dataset em ALTA QUALIDADE com AirSim
Captura imagens em resolução máxima sem redimensionamento
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
    print("🚀 GERADOR DE DATASET HD - ALTA QUALIDADE")
    print("="*60)

    # Configuração
    output_dir = Path("dataset_hd")
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

    print(f"\n📁 Salvando dataset HD em: {output_dir.absolute()}")

    # Conecta ao AirSim
    print("\n🔌 Conectando ao AirSim...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Drones: {vehicles}")

    # CONFIGURAÇÃO DE CÂMERA PARA ALTA QUALIDADE
    print("\n📸 Configurando câmera para alta resolução...")

    # Tenta obter informação da câmera
    try:
        camera_info = client.simGetCameraInfo("0", vehicle_name="Ego")
        print(f"   Câmera detectada - FOV: {camera_info.fov}")
    except:
        print("   Usando configuração padrão de câmera")

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
    num_frames = 50  # Menos frames mas com melhor qualidade
    frames_captured = 0

    print(f"\n📸 Iniciando captura HD de {num_frames} frames...")
    print("   ⚠️ IMPORTANTE: Capturando em RESOLUÇÃO MÁXIMA")
    print("   💡 Dica: Se as imagens ainda estiverem pixeladas,")
    print("      verifique settings.json no Windows para aumentar resolução")

    for frame_idx in range(num_frames):
        # Calcula tempo para movimento suave
        t = frame_idx * 0.15

        # MOVIMENTO DOS DRONES
        num_drones = len(vehicles) - 1
        drone_idx = 0

        for vehicle in vehicles:
            if vehicle == "Ego":
                # Ego rotaciona e ajusta altura
                yaw_angle = (frame_idx * 3) % 360
                client.rotateToYawAsync(yaw_angle, vehicle_name="Ego")

                # Varia altura do Ego para diferentes perspectivas
                ego_z = -15 + 3 * math.sin(t * 0.3)
                client.moveToZAsync(ego_z, 2, vehicle_name="Ego")
                continue

            # Outros drones em padrões variados
            if frame_idx < 25:
                # Padrão circular próximo
                angle = (2 * math.pi * drone_idx / num_drones) + t
                radius = 15 + 5 * math.sin(t * 0.7)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = -12 + 4 * math.sin(t * 0.5 + drone_idx)
            else:
                # Padrão cruzado dinâmico
                if drone_idx % 2 == 0:
                    x = 20 * math.cos(t + drone_idx)
                    y = 20 * math.sin(t * 1.5 + drone_idx)
                else:
                    x = 15 * math.sin(t + drone_idx)
                    y = 15 * math.cos(t * 1.3 + drone_idx)
                z = -10 + 5 * math.sin(t * 0.4 + drone_idx * 0.5)

            try:
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle)
            except:
                pass

            drone_idx += 1

        # Aguarda movimento e estabilização
        time.sleep(0.5)

        # CAPTURA DE IMAGENS EM ALTA QUALIDADE
        print(f"\r📸 Capturando frame HD {frame_idx+1}/{num_frames}...", end="", flush=True)

        try:
            # IMPORTANTE: Captura sem compressão para máxima qualidade
            # pixels_as_float=False para imagem não comprimida
            # compress=False para evitar compressão PNG/JPG
            responses = client.simGetImages([
                # RGB em alta qualidade
                airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False),
                # Segmentação
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, pixels_as_float=False, compress=False),
                # Profundidade
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)
            ], vehicle_name="Ego")

            # Processa RGB
            if responses[0].image_data_uint8:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # NÃO REDIMENSIONA - mantém resolução original
                print(f" (Original: {responses[0].width}x{responses[0].height})", end="", flush=True)

                # Salva com máxima qualidade PNG
                rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(rgb_file), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Sem compressão

                # Alternativa: Tenta capturar com diferentes câmeras/resoluções
                if responses[0].width < 1280:
                    print(f"\n   ⚠️ Resolução baixa detectada: {responses[0].width}x{responses[0].height}")

                    # Tenta captura alternativa com front_center
                    try:
                        response_hd = client.simGetImages([
                            airsim.ImageRequest("front_center", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                        ], vehicle_name="Ego")

                        if response_hd and response_hd[0].image_data_uint8:
                            img_1d_hd = np.frombuffer(response_hd[0].image_data_uint8, dtype=np.uint8)
                            img_rgb_hd = img_1d_hd.reshape(response_hd[0].height, response_hd[0].width, 3)
                            img_bgr_hd = cv2.cvtColor(img_rgb_hd, cv2.COLOR_RGB2BGR)

                            if response_hd[0].width > responses[0].width:
                                img_bgr = img_bgr_hd
                                print(f"   ✅ Usando front_center: {response_hd[0].width}x{response_hd[0].height}")
                    except:
                        pass

            # Processa Segmentação
            if len(responses) > 1 and responses[1].image_data_uint8:
                img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                seg_file = dirs['segmentation'] / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(seg_file), img_seg, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Gera anotações YOLO
                generate_yolo_annotations(img_seg, dirs['annotations'], frames_captured)

            # Processa Profundidade
            if len(responses) > 2 and responses[2].image_data_float:
                img_depth = airsim.list_to_2d_float_array(
                    responses[2].image_data_float,
                    responses[2].width,
                    responses[2].height
                )

                # Melhor normalização para profundidade
                img_depth_normalized = np.clip(img_depth / 50.0, 0, 1) * 255
                img_depth_uint8 = img_depth_normalized.astype(np.uint8)

                # Aplica colormap para melhor visualização
                img_depth_colored = cv2.applyColorMap(img_depth_uint8, cv2.COLORMAP_JET)

                depth_file = dirs['depth'] / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(depth_file), img_depth_colored, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # Salva metadata detalhado
            metadata = {
                'frame': frames_captured,
                'timestamp': datetime.now().isoformat(),
                'image_info': {
                    'width': responses[0].width if responses else 0,
                    'height': responses[0].height if responses else 0,
                    'camera': '0'
                },
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
                        'orientation': {
                            'w': state.kinematics_estimated.orientation.w_val,
                            'x': state.kinematics_estimated.orientation.x_val,
                            'y': state.kinematics_estimated.orientation.y_val,
                            'z': state.kinematics_estimated.orientation.z_val
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
            print(f"\n   ✅ Progresso: {frame_idx+1}/{num_frames} frames HD")

    # TESTE DE CAPTURA EM SUPER ALTA RESOLUÇÃO
    print("\n\n🔬 Testando captura em múltiplas resoluções...")

    test_cameras = ["0", "front_center", "1", "high_res"]
    best_resolution = 0
    best_camera = "0"

    for cam in test_cameras:
        try:
            test_response = client.simGetImages([
                airsim.ImageRequest(cam, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
            ], vehicle_name="Ego")

            if test_response and test_response[0].image_data_uint8:
                resolution = test_response[0].width * test_response[0].height
                print(f"   Câmera '{cam}': {test_response[0].width}x{test_response[0].height}")

                if resolution > best_resolution:
                    best_resolution = resolution
                    best_camera = cam
        except:
            pass

    print(f"\n   🏆 Melhor câmera: '{best_camera}' com resolução máxima")

    # FINALIZAÇÃO
    print(f"\n🛬 Pousando todos os drones...")
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
    print("✅ DATASET HD GERADO COM SUCESSO!")
    print("="*60)

    rgb_count = len(list(dirs['rgb'].glob("*.png")))
    seg_count = len(list(dirs['segmentation'].glob("*.png")))
    depth_count = len(list(dirs['depth'].glob("*.png")))
    anno_count = len(list(dirs['annotations'].glob("*.txt")))
    meta_count = len(list(dirs['metadata'].glob("*.json")))

    print(f"\n📊 Estatísticas do Dataset HD:")
    print(f"   Frames totais: {frames_captured}/{num_frames}")
    print(f"   Imagens RGB HD: {rgb_count}")
    print(f"   Segmentações: {seg_count}")
    print(f"   Profundidade: {depth_count}")
    print(f"   Anotações YOLO: {anno_count}")
    print(f"   Metadata: {meta_count}")

    # Verifica qualidade das imagens
    if rgb_count > 0:
        sample_img = cv2.imread(str(list(dirs['rgb'].glob("*.png"))[0]))
        print(f"\n📐 Resolução das imagens: {sample_img.shape[1]}x{sample_img.shape[0]}")

        if sample_img.shape[1] < 1280:
            print("\n⚠️ ATENÇÃO: Imagens ainda em baixa resolução!")
            print("\n📝 Para melhorar a qualidade:")
            print("   1. No Windows, edite: Documents/AirSim/settings.json")
            print("   2. Adicione nas CaptureSettings de cada câmera:")
            print('      "Width": 1920,')
            print('      "Height": 1080,')
            print("   3. Reinicie o AirSim")
            print("\n   Exemplo de configuração:")
            print('''
      "CaptureSettings": [
        {
          "ImageType": 0,
          "Width": 1920,
          "Height": 1080,
          "FOV_Degrees": 90
        }
      ]
            ''')
        else:
            print("   ✅ Qualidade HD confirmada!")

    print(f"\n📁 Dataset HD completo em: {output_dir.absolute()}")

def generate_yolo_annotations(seg_image, anno_dir, frame_idx):
    """Gera anotações YOLO com detecção melhorada"""
    h, w = seg_image.shape[:2]
    annotations = []

    # Cores de drones na segmentação - ajustadas para Blocks
    drone_colors = [
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 0),  # Yellow
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (128, 0, 128),  # Purple
    ]

    for color in drone_colors:
        # Tolerância menor para detecção mais precisa
        lower = np.array(color) - 10
        upper = np.array(color) + 10
        mask = cv2.inRange(seg_image, lower, upper)

        if mask.any():
            # Aplica operações morfológicas para limpar ruído
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 30:  # Ignora objetos muito pequenos
                    continue

                x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

                # Formato YOLO
                cx = (x + bbox_w/2) / w
                cy = (y + bbox_h/2) / h
                nw = bbox_w / w
                nh = bbox_h / h

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
        print(f"\n\n❌ Erro: {e}")