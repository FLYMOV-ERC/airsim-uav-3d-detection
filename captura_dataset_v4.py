#!/usr/bin/env python3
"""
Script para captura de dataset - compatível com AirSim v4
"""

import os
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import math

import airsim

def main():
    print("\n" + "="*50)
    print("🚀 CAPTURA DE DATASET - AIRSIM V4")
    print("="*50)

    # Configuração
    output_dir = Path("dataset_capturado")
    output_dir.mkdir(exist_ok=True)

    rgb_dir = output_dir / "rgb"
    seg_dir = output_dir / "segmentation"
    meta_dir = output_dir / "metadata"

    rgb_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    print(f"\n📁 Diretório: {output_dir.absolute()}")

    # Conecta
    print("\n🔌 Conectando...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Drones: {vehicles}")

    # Prepara drones
    print("\n🎮 Preparando drones...")
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
            print(f"   ✅ {v}")
        except Exception as e:
            print(f"   ⚠️ {v}: {e}")

    # Decola
    print("\n🛫 Decolando...")
    for v in vehicles:
        try:
            client.takeoffAsync(vehicle_name=v)
        except:
            pass
    time.sleep(5)

    # Move Ego para posição de observação
    try:
        client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()
    except:
        pass

    # Captura frames
    num_frames = 30
    print(f"\n📸 Capturando {num_frames} frames...")

    frames_captured = 0

    for i in range(num_frames):
        print(f"\nFrame {i+1}/{num_frames}:")

        # Move outros drones
        t = i * 0.3
        for idx, v in enumerate(vehicles):
            if v != "Ego":
                try:
                    angle = (2 * math.pi * idx / len(vehicles)) + t
                    x = 15 * math.cos(angle)
                    y = 15 * math.sin(angle)
                    z = -10 + 3 * math.sin(t * 0.5)
                    client.moveToPositionAsync(x, y, z, 5, vehicle_name=v)
                except:
                    pass

        time.sleep(0.3)

        # Tenta diferentes métodos de captura
        success = False

        # Método 1: simGetImages com external=False
        try:
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ], vehicle_name="Ego", external=False)

            if responses[0].image_data_uint8:
                # RGB
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                rgb_file = rgb_dir / f"frame_{frames_captured:06d}.png"
                cv2.imwrite(str(rgb_file), img_bgr)
                print(f"   ✅ RGB salva ({responses[0].width}x{responses[0].height})")

                # Segmentação
                if len(responses) > 1 and responses[1].image_data_uint8:
                    img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                    img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                    seg_file = seg_dir / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(seg_file), img_seg)
                    print(f"   ✅ Segmentação salva")

                success = True

        except Exception as e1:
            # Método 2: simGetImages sem external
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
                ], vehicle_name="Ego")

                if responses[0].image_data_uint8:
                    # RGB
                    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    rgb_file = rgb_dir / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_file), img_bgr)
                    print(f"   ✅ RGB salva (método 2)")

                    success = True

            except Exception as e2:
                # Método 3: simGetImage único
                try:
                    # Tenta com external=False
                    img_data = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name="Ego", external=False)

                    if img_data:
                        # Decodifica PNG/JPG
                        nparr = np.frombuffer(img_data, np.uint8)
                        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if img_bgr is not None:
                            rgb_file = rgb_dir / f"frame_{frames_captured:06d}.png"
                            cv2.imwrite(str(rgb_file), img_bgr)
                            print(f"   ✅ RGB salva (método 3)")
                            success = True

                except Exception as e3:
                    # Método 4: simGetImage sem external
                    try:
                        img_data = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name="Ego")

                        if img_data:
                            nparr = np.frombuffer(img_data, np.uint8)
                            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if img_bgr is not None:
                                rgb_file = rgb_dir / f"frame_{frames_captured:06d}.png"
                                cv2.imwrite(str(rgb_file), img_bgr)
                                print(f"   ✅ RGB salva (método 4)")
                                success = True

                    except Exception as e4:
                        print(f"   ❌ Todos os métodos falharam")
                        print(f"      Erro: {e4}")

        if success:
            frames_captured += 1

            # Salva metadata
            try:
                metadata = {
                    'frame': frames_captured - 1,
                    'timestamp': datetime.now().isoformat(),
                    'drones': {}
                }

                for v in vehicles:
                    try:
                        state = client.getMultirotorState(vehicle_name=v)
                        metadata['drones'][v] = {
                            'x': state.kinematics_estimated.position.x_val,
                            'y': state.kinematics_estimated.position.y_val,
                            'z': state.kinematics_estimated.position.z_val
                        }
                    except:
                        pass

                meta_file = meta_dir / f"frame_{frames_captured-1:06d}.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            except:
                pass

    # Pousa
    print("\n🛬 Pousando...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
        except:
            pass

    time.sleep(3)

    for v in vehicles:
        try:
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    # Resumo
    print("\n" + "="*50)
    print("✅ CAPTURA COMPLETA!")
    print("="*50)

    rgb_count = len(list(rgb_dir.glob("*.png")))
    seg_count = len(list(seg_dir.glob("*.png")))
    meta_count = len(list(meta_dir.glob("*.json")))

    print(f"\n📊 Resultados:")
    print(f"   Frames capturados: {frames_captured}/{num_frames}")
    print(f"   Imagens RGB: {rgb_count}")
    print(f"   Segmentações: {seg_count}")
    print(f"   Metadados: {meta_count}")
    print(f"\n📁 Dataset em: {output_dir.absolute()}")

    if frames_captured > 0:
        print("\n✅ SUCESSO! Dataset gerado com imagens!")
    else:
        print("\n⚠️ Nenhuma imagem foi capturada. Verifique a configuração.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")