#!/usr/bin/env python3
"""
Script para gerar dataset de drones no ambiente Blocks
"""

import os
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import random
import math

import airsim

class DatasetGenerator:
    def __init__(self, output_dir="dataset_blocks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Cria subdiretórios
        self.dirs = {
            'rgb': self.output_dir / 'rgb',
            'segmentation': self.output_dir / 'segmentation',
            'annotations': self.output_dir / 'annotations',
            'metadata': self.output_dir / 'metadata'
        }

        for d in self.dirs.values():
            d.mkdir(exist_ok=True)

        # Conecta ao AirSim
        print("🔌 Conectando ao AirSim...")
        self.client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
        self.client.confirmConnection()

        # Lista drones
        self.drones = self.client.listVehicles()
        print(f"✅ Drones detectados: {self.drones}")

        self.frame_count = 0

    def setup_drones(self):
        """Prepara todos os drones"""
        print("\n🎮 Preparando drones...")

        for drone in self.drones:
            try:
                self.client.enableApiControl(True, drone)
                self.client.armDisarm(True, drone)
                print(f"   ✅ {drone} preparado")
            except Exception as e:
                print(f"   ⚠️ Erro preparando {drone}: {e}")

        # Decola drones
        print("\n🛫 Decolando drones...")
        for drone in self.drones:
            try:
                self.client.takeoffAsync(vehicle_name=drone)
            except:
                pass

        time.sleep(5)
        print("✅ Drones prontos!")

    def move_drones_pattern(self, t):
        """Move drones em padrão circular"""
        num_drones = len(self.drones)
        radius = 15

        for i, drone in enumerate(self.drones):
            try:
                if drone == "Ego":
                    # Ego fica no centro observando
                    self.client.moveToPositionAsync(
                        0, 0, -10, 3, vehicle_name=drone
                    )
                else:
                    # Outros drones voam em círculo
                    angle = (2 * math.pi * i / num_drones) + t * 0.5
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = -10 + 3 * math.sin(t * 0.3)

                    self.client.moveToPositionAsync(
                        x, y, z, 3, vehicle_name=drone
                    )
            except:
                pass

    def capture_frame(self):
        """Captura imagens e gera anotações"""
        print(f"\n📸 Capturando frame {self.frame_count}...")

        # Captura do drone Ego (observador)
        try:
            # Requisições de imagem
            requests = [
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ]

            responses = self.client.simGetImages(requests, vehicle_name="Ego")

            # Processa RGB
            if responses[0].image_data_uint8:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                rgb_path = self.dirs['rgb'] / f"frame_{self.frame_count:06d}.png"
                cv2.imwrite(str(rgb_path), img_bgr)
                print(f"   ✅ RGB salva")

            # Processa Segmentação
            if len(responses) > 1 and responses[1].image_data_uint8:
                img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                seg_path = self.dirs['segmentation'] / f"frame_{self.frame_count:06d}.png"
                cv2.imwrite(str(seg_path), img_seg)
                print(f"   ✅ Segmentação salva")

                # Gera anotações YOLO
                self.generate_annotations(img_seg)

            # Salva metadata
            self.save_metadata()

            self.frame_count += 1

        except Exception as e:
            print(f"   ❌ Erro na captura: {e}")

    def generate_annotations(self, seg_image):
        """Gera anotações YOLO a partir da segmentação"""
        h, w = seg_image.shape[:2]
        annotations = []

        # Cores típicas de drones na segmentação
        drone_colors = [
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
        ]

        for color in drone_colors:
            # Cria máscara para a cor
            lower = np.array(color) - 10
            upper = np.array(color) + 10
            mask = cv2.inRange(seg_image, lower, upper)

            if mask.any():
                # Encontra contornos
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:  # Ignora objetos muito pequenos
                        continue

                    x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

                    # Formato YOLO
                    cx = (x + bbox_w/2) / w
                    cy = (y + bbox_h/2) / h
                    nw = bbox_w / w
                    nh = bbox_h / h

                    annotations.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if annotations:
            anno_path = self.dirs['annotations'] / f"frame_{self.frame_count:06d}.txt"
            with open(anno_path, 'w') as f:
                f.write('\n'.join(annotations))
            print(f"   ✅ {len(annotations)} anotações salvas")

    def save_metadata(self):
        """Salva posições dos drones"""
        metadata = {
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'drones': {}
        }

        for drone in self.drones:
            try:
                state = self.client.getMultirotorState(vehicle_name=drone)
                metadata['drones'][drone] = {
                    'position': {
                        'x': state.kinematics_estimated.position.x_val,
                        'y': state.kinematics_estimated.position.y_val,
                        'z': state.kinematics_estimated.position.z_val
                    }
                }
            except:
                pass

        meta_path = self.dirs['metadata'] / f"frame_{self.frame_count:06d}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def run(self, num_frames=50):
        """Executa geração do dataset"""
        print(f"\n🎬 Gerando dataset com {num_frames} frames")

        self.setup_drones()

        for i in range(num_frames):
            # Tempo para movimento suave
            t = i * 0.1

            # Move drones
            self.move_drones_pattern(t)
            time.sleep(0.5)

            # Captura frame
            self.capture_frame()

            # Status
            if (i + 1) % 10 == 0:
                print(f"\n📊 Progresso: {i+1}/{num_frames} frames")

        # Pousa drones
        print("\n🛬 Pousando drones...")
        for drone in self.drones:
            try:
                self.client.landAsync(vehicle_name=drone)
                time.sleep(0.5)
                self.client.armDisarm(False, drone)
                self.client.enableApiControl(False, drone)
            except:
                pass

        # Resumo
        print("\n" + "="*50)
        print("✅ DATASET GERADO COM SUCESSO!")
        print("="*50)
        print(f"\n📊 Estatísticas:")
        print(f"   Frames capturados: {self.frame_count}")
        print(f"   Imagens RGB: {len(list(self.dirs['rgb'].glob('*.png')))}")
        print(f"   Segmentações: {len(list(self.dirs['segmentation'].glob('*.png')))}")
        print(f"   Anotações: {len(list(self.dirs['annotations'].glob('*.txt')))}")
        print(f"\n📁 Dataset salvo em: {self.output_dir.absolute()}")

def main():
    try:
        generator = DatasetGenerator(output_dir="dataset_blocks")
        generator.run(num_frames=50)  # Gera 50 frames
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    main()