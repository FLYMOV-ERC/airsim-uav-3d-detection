#!/usr/bin/env python3
"""
Script avançado para geração de dataset com AirSim
Combina as melhores práticas dos scripts existentes
"""

import os
import json
import time
import math
import argparse
import threading
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import random

try:
    import airsim
except ImportError:
    import cosysairsim as airsim


class DatasetGenerator:
    """Gerador avançado de dataset para treino de detecção de drones"""

    def __init__(self, ip="172.19.80.1", port=41451, output_dir="dataset_novo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Cria subdiretórios
        self.dirs = {
            'rgb': self.output_dir / 'rgb',
            'seg': self.output_dir / 'segmentation',
            'depth': self.output_dir / 'depth',
            'annotations': self.output_dir / 'annotations',
            'metadata': self.output_dir / 'metadata'
        }

        for d in self.dirs.values():
            d.mkdir(exist_ok=True)

        # Conecta ao AirSim
        print(f"🔌 Conectando ao AirSim em {ip}:{port}")
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()

        # Descobre drones disponíveis
        self.drones = self.client.listVehicles()
        print(f"✅ Drones detectados: {self.drones}")

        # Configurações de câmeras
        self.cameras = ["front_center", "back_center", "bottom_center"]
        self.image_width = 1280
        self.image_height = 720

        # Estado de execução
        self.running = True
        self.frame_count = 0

    def setup_drones(self):
        """Habilita e prepara todos os drones"""
        print("\n🎮 Preparando drones...")
        for drone in self.drones:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)

        # Decola todos
        print("🛫 Decolando drones...")
        tasks = []
        for drone in self.drones:
            tasks.append(self.client.takeoffAsync(vehicle_name=drone))

        for task in tasks:
            task.join()

        # Posiciona em altitude inicial
        altitude = -15
        for drone in self.drones:
            pos = self.client.getMultirotorState(drone).kinematics_estimated.position
            self.client.moveToPositionAsync(
                pos.x_val, pos.y_val, altitude, 5, vehicle_name=drone
            ).join()

        print("✅ Drones prontos!")
        time.sleep(2)

    def move_drones_formation(self, pattern="circle", t=0):
        """Move drones em diferentes formações"""
        num_drones = len(self.drones)

        if pattern == "circle":
            # Movimento circular
            radius = 20
            for i, drone in enumerate(self.drones):
                if drone == "Ego":
                    # Ego fica no centro como observador
                    self.client.moveToPositionAsync(0, 0, -15, 3, vehicle_name=drone)
                else:
                    angle = (2 * math.pi * i / num_drones) + t
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = -15 + 5 * math.sin(t * 0.5)
                    self.client.moveToPositionAsync(x, y, z, 3, vehicle_name=drone)

        elif pattern == "line":
            # Formação em linha
            spacing = 10
            for i, drone in enumerate(self.drones):
                x = (i - num_drones/2) * spacing
                y = 20 + 5 * math.sin(t)
                z = -15 + 3 * math.sin(t * 0.7)
                self.client.moveToPositionAsync(x, y, z, 3, vehicle_name=drone)

        elif pattern == "random":
            # Movimento aleatório mas controlado
            for drone in self.drones:
                if drone == "Ego":
                    continue

                state = self.client.getMultirotorState(drone).kinematics_estimated.position
                x = state.x_val + random.uniform(-5, 5)
                y = state.y_val + random.uniform(-5, 5)
                z = state.z_val + random.uniform(-2, 2)

                # Limita área de movimento
                x = max(-50, min(50, x))
                y = max(-50, min(50, y))
                z = max(-30, min(-5, z))

                self.client.moveToPositionAsync(x, y, z, 3, vehicle_name=drone)

        elif pattern == "swarm":
            # Movimento tipo enxame
            center_x = 20 * math.cos(t * 0.3)
            center_y = 20 * math.sin(t * 0.3)

            for i, drone in enumerate(self.drones):
                if drone == "Ego":
                    continue

                offset_angle = 2 * math.pi * i / num_drones
                x = center_x + 10 * math.cos(offset_angle + t)
                y = center_y + 10 * math.sin(offset_angle + t)
                z = -15 + 5 * math.sin(t * 0.5 + offset_angle)

                self.client.moveToPositionAsync(x, y, z, 3, vehicle_name=drone)

    def capture_images(self, drone="Ego", camera="front_center"):
        """Captura imagens RGB, Segmentação e Profundidade"""
        requests = [
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False)
        ]

        responses = self.client.simGetImages(requests, vehicle_name=drone)

        images = {}

        # RGB
        if responses[0].image_data_uint8:
            img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img = img.reshape(responses[0].height, responses[0].width, 3)
            images['rgb'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Segmentation
        if responses[1].image_data_uint8:
            img = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
            img = img.reshape(responses[1].height, responses[1].width, 3)
            images['seg'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Depth
        if responses[2].image_data_float:
            img = airsim.list_to_2d_float_array(
                responses[2].image_data_float,
                responses[2].width,
                responses[2].height
            )
            # Normaliza para visualização
            img_normalized = np.clip(img / 100.0, 0, 1) * 255
            images['depth'] = img_normalized.astype(np.uint8)

        return images

    def generate_annotations(self, seg_image):
        """Gera anotações YOLO a partir da segmentação"""
        annotations = []

        # Define cores de interesse (drones)
        drone_colors = [
            (255, 0, 255),  # Magenta - comum para drones
            (0, 255, 255),  # Cyan
        ]

        h, w = seg_image.shape[:2]

        for color in drone_colors:
            # Cria máscara para cor específica
            mask = cv2.inRange(seg_image, color, color)

            if mask.any():
                # Encontra contornos
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, bbox_w, bbox_h = cv2.boundingRect(contour)

                    # Ignora detecções muito pequenas
                    if bbox_w * bbox_h < 100:
                        continue

                    # Formato YOLO: class_id center_x center_y width height (normalizado)
                    center_x = (x + bbox_w / 2) / w
                    center_y = (y + bbox_h / 2) / h
                    norm_w = bbox_w / w
                    norm_h = bbox_h / h

                    annotations.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        return annotations

    def save_metadata(self):
        """Salva metadados de todos os drones"""
        metadata = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'drones': {}
        }

        for drone in self.drones:
            state = self.client.getMultirotorState(vehicle_name=drone)
            metadata['drones'][drone] = {
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

        meta_file = self.dirs['metadata'] / f"frame_{self.frame_count:06d}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def set_environment(self, weather="clear", time_of_day=12):
        """Configura ambiente (clima e hora)"""
        weather_presets = {
            'clear': {'Fog': 0, 'Rain': 0, 'Snow': 0, 'Dust': 0},
            'foggy': {'Fog': 0.5, 'Rain': 0, 'Snow': 0, 'Dust': 0},
            'rainy': {'Fog': 0.1, 'Rain': 0.7, 'Snow': 0, 'Dust': 0},
            'snowy': {'Fog': 0.2, 'Rain': 0, 'Snow': 0.8, 'Dust': 0},
            'dusty': {'Fog': 0, 'Rain': 0, 'Snow': 0, 'Dust': 0.6}
        }

        if weather in weather_presets:
            params = weather_presets[weather]
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, params['Fog'])
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, params['Rain'])
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, params['Snow'])
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, params['Dust'])

        # Define hora do dia
        time_string = f"2024-01-01 {time_of_day:02d}:00:00"
        self.client.simSetTimeOfDay(True, time_string)

    def collect_frame(self):
        """Coleta um frame completo de dados"""
        # Captura de todas as câmeras do Ego
        for camera in self.cameras[:1]:  # Por enquanto só front_center
            images = self.capture_images("Ego", camera)

            if 'rgb' in images:
                rgb_path = self.dirs['rgb'] / f"frame_{self.frame_count:06d}.png"
                cv2.imwrite(str(rgb_path), images['rgb'])

            if 'seg' in images:
                seg_path = self.dirs['seg'] / f"frame_{self.frame_count:06d}.png"
                cv2.imwrite(str(seg_path), images['seg'])

                # Gera anotações YOLO
                annotations = self.generate_annotations(images['seg'])
                if annotations:
                    anno_path = self.dirs['annotations'] / f"frame_{self.frame_count:06d}.txt"
                    with open(anno_path, 'w') as f:
                        f.write('\n'.join(annotations))

            if 'depth' in images:
                depth_path = self.dirs['depth'] / f"frame_{self.frame_count:06d}.png"
                cv2.imwrite(str(depth_path), images['depth'])

        # Salva metadata
        self.save_metadata()

        self.frame_count += 1

    def run(self, num_frames=1000, patterns=['circle', 'line', 'swarm', 'random']):
        """Executa coleta de dataset"""
        self.setup_drones()

        frames_per_pattern = num_frames // len(patterns)
        weather_options = ['clear', 'foggy', 'rainy']
        time_options = [6, 10, 14, 18, 22]  # Diferentes horas do dia

        print(f"\n🎬 Iniciando coleta de {num_frames} frames")
        print(f"   Padrões: {patterns}")
        print(f"   Frames por padrão: {frames_per_pattern}")

        with tqdm(total=num_frames, desc="Coletando") as pbar:
            for pattern_idx, pattern in enumerate(patterns):
                print(f"\n📐 Padrão: {pattern}")

                for frame_in_pattern in range(frames_per_pattern):
                    # Tempo para movimento suave
                    t = (pattern_idx * frames_per_pattern + frame_in_pattern) * 0.1

                    # Varia ambiente a cada 50 frames
                    if self.frame_count % 50 == 0:
                        weather = random.choice(weather_options)
                        tod = random.choice(time_options)
                        self.set_environment(weather, tod)

                    # Move drones
                    self.move_drones_formation(pattern, t)

                    # Espera movimento
                    time.sleep(0.1)

                    # Coleta frame
                    self.collect_frame()

                    pbar.update(1)

                    # Estatísticas periódicas
                    if self.frame_count % 100 == 0:
                        print(f"\n📊 Frames coletados: {self.frame_count}")
                        print(f"   RGB: {len(list(self.dirs['rgb'].glob('*.png')))}")
                        print(f"   Segmentação: {len(list(self.dirs['seg'].glob('*.png')))}")
                        print(f"   Anotações: {len(list(self.dirs['annotations'].glob('*.txt')))}")

        print(f"\n✅ Coleta completa!")
        print(f"   Total de frames: {self.frame_count}")
        print(f"   Diretório: {self.output_dir.absolute()}")

        # Pousa drones
        print("\n🛬 Pousando drones...")
        for drone in self.drones:
            self.client.landAsync(vehicle_name=drone)

        time.sleep(5)

        for drone in self.drones:
            self.client.armDisarm(False, drone)
            self.client.enableApiControl(False, drone)

        # Salva resumo
        self.save_summary()

    def save_summary(self):
        """Salva resumo do dataset"""
        summary = {
            'dataset_info': {
                'total_frames': self.frame_count,
                'creation_date': datetime.now().isoformat(),
                'drones': self.drones,
                'cameras': self.cameras,
                'image_size': [self.image_width, self.image_height]
            },
            'file_counts': {
                'rgb_images': len(list(self.dirs['rgb'].glob('*.png'))),
                'seg_images': len(list(self.dirs['seg'].glob('*.png'))),
                'depth_images': len(list(self.dirs['depth'].glob('*.png'))),
                'annotations': len(list(self.dirs['annotations'].glob('*.txt'))),
                'metadata': len(list(self.dirs['metadata'].glob('*.json')))
            }
        }

        with open(self.output_dir / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n📄 Resumo salvo em dataset_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Gerador de Dataset AirSim")
    parser.add_argument('--ip', default='172.19.80.1', help='IP do AirSim')
    parser.add_argument('--port', type=int, default=41451, help='Porta do AirSim')
    parser.add_argument('--frames', type=int, default=500, help='Número de frames')
    parser.add_argument('--output', default='dataset_novo', help='Diretório de saída')
    parser.add_argument('--patterns', nargs='+',
                       default=['circle', 'line', 'swarm', 'random'],
                       help='Padrões de movimento')

    args = parser.parse_args()

    # Cria gerador
    generator = DatasetGenerator(
        ip=args.ip,
        port=args.port,
        output_dir=args.output
    )

    try:
        # Executa coleta
        generator.run(
            num_frames=args.frames,
            patterns=args.patterns
        )
    except KeyboardInterrupt:
        print("\n⚠️ Coleta interrompida!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        # Garante que drones pousem
        try:
            for drone in generator.drones:
                generator.client.landAsync(vehicle_name=drone)
                generator.client.armDisarm(False, drone)
                generator.client.enableApiControl(False, drone)
        except:
            pass


if __name__ == "__main__":
    main()