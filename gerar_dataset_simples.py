#!/usr/bin/env python3
"""
Script simplificado para gerar dataset - compatível com nova versão AirSim
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
    print("🚀 GERADOR DE DATASET AIRSIM - BLOCKS")
    print("="*50)

    # Configuração
    output_dir = Path("dataset_blocks_novo")
    output_dir.mkdir(exist_ok=True)

    # Cria subdiretórios
    rgb_dir = output_dir / "rgb"
    seg_dir = output_dir / "segmentation"
    meta_dir = output_dir / "metadata"

    rgb_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    print(f"\n📁 Salvando em: {output_dir.absolute()}")

    # Conecta
    print("\n🔌 Conectando ao AirSim...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    # Lista drones
    vehicles = client.listVehicles()
    print(f"✅ Drones encontrados: {vehicles}")

    # Prepara drones
    print("\n🎮 Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
            print(f"   ✅ {vehicle} pronto")
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

    # Move para posições iniciais
    print("📍 Posicionando drones...")
    try:
        client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()
    except:
        pass

    # Captura frames
    num_frames = 20
    print(f"\n📸 Capturando {num_frames} frames...")

    for i in range(num_frames):
        print(f"\n   Frame {i+1}/{num_frames}:")

        # Move outros drones em círculo
        t = i * 0.2
        for idx, vehicle in enumerate(vehicles):
            if vehicle != "Ego":
                try:
                    angle = (2 * math.pi * idx / len(vehicles)) + t
                    x = 10 * math.cos(angle)
                    y = 10 * math.sin(angle)
                    z = -10 + 2 * math.sin(t * 0.5)
                    client.moveToPositionAsync(x, y, z, 3, vehicle_name=vehicle)
                except:
                    pass

        time.sleep(0.5)

        # Tenta capturar com diferentes métodos
        try:
            # Método 1: Sem especificar external
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ], vehicle_name="Ego", external=False)

            process_images(responses, i, rgb_dir, seg_dir, meta_dir, vehicles, client)

        except Exception as e1:
            # Método 2: Sem o parâmetro external
            try:
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
                ], vehicle_name="Ego")

                process_images(responses, i, rgb_dir, seg_dir, meta_dir, vehicles, client)

            except Exception as e2:
                # Método 3: Captura individual
                try:
                    print("      Tentando método alternativo...")

                    # RGB
                    response = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name="Ego")
                    if response:
                        img_1d = np.frombuffer(response, dtype=np.uint8)
                        # Assume 720p
                        img = img_1d.reshape(720, 1280, 3)
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        filename = rgb_dir / f"frame_{i:06d}.png"
                        cv2.imwrite(str(filename), img_bgr)
                        print(f"      ✅ RGB salva")

                    # Segmentação
                    response = client.simGetImage("0", airsim.ImageType.Segmentation, vehicle_name="Ego")
                    if response:
                        img_1d = np.frombuffer(response, dtype=np.uint8)
                        img = img_1d.reshape(720, 1280, 3)

                        filename = seg_dir / f"frame_{i:06d}.png"
                        cv2.imwrite(str(filename), img)
                        print(f"      ✅ Segmentação salva")

                except Exception as e3:
                    print(f"      ❌ Erro: {e3}")

        # Salva metadata
        try:
            metadata = {
                'frame': i,
                'timestamp': datetime.now().isoformat(),
                'drones': {}
            }

            for vehicle in vehicles:
                try:
                    state = client.getMultirotorState(vehicle_name=vehicle)
                    metadata['drones'][vehicle] = {
                        'x': state.kinematics_estimated.position.x_val,
                        'y': state.kinematics_estimated.position.y_val,
                        'z': state.kinematics_estimated.position.z_val
                    }
                except:
                    pass

            meta_file = meta_dir / f"frame_{i:06d}.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except:
            pass

    # Pousa
    print("\n🛬 Pousando drones...")
    for vehicle in vehicles:
        try:
            client.landAsync(vehicle_name=vehicle)
        except:
            pass

    time.sleep(3)

    for vehicle in vehicles:
        try:
            client.armDisarm(False, vehicle)
            client.enableApiControl(False, vehicle)
        except:
            pass

    # Estatísticas
    print("\n" + "="*50)
    print("✅ DATASET GERADO!")
    print("="*50)

    rgb_count = len(list(rgb_dir.glob("*.png")))
    seg_count = len(list(seg_dir.glob("*.png")))
    meta_count = len(list(meta_dir.glob("*.json")))

    print(f"\n📊 Resultados:")
    print(f"   Imagens RGB: {rgb_count}")
    print(f"   Segmentações: {seg_count}")
    print(f"   Metadados: {meta_count}")
    print(f"\n📁 Salvo em: {output_dir.absolute()}")

def process_images(responses, frame_idx, rgb_dir, seg_dir, meta_dir, vehicles, client):
    """Processa respostas de imagem"""
    # RGB
    if responses[0].image_data_uint8:
        img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        filename = rgb_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(filename), img_bgr)
        print(f"      ✅ RGB salva")

    # Segmentação
    if len(responses) > 1 and responses[1].image_data_uint8:
        img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

        filename = seg_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(filename), img_seg)
        print(f"      ✅ Segmentação salva")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")