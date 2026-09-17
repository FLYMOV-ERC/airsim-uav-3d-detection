#!/usr/bin/env python3
"""
Gerador de Dataset para o ambiente Mountains
Coleta imagens RGB com múltiplas variações
"""

import airsim
import numpy as np
import cv2
import time
import json
from pathlib import Path
from datetime import datetime
import math

class DatasetGenerator:
    def __init__(self):
        self.output_dir = Path("dataset_mountains_completo")
        self.output_dir.mkdir(exist_ok=True)

        # Subpastas
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        # Conecta ao AirSim
        print("📡 Conectando ao Mountains...")
        self.client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
        self.client.confirmConnection()

        # Descobre drones
        self.drones = self.client.listVehicles()
        print(f"✅ Conectado! Drones: {self.drones}\n")

        self.frame_count = 0
        self.start_time = time.time()

    def setup_drones(self):
        """Prepara todos os drones"""
        print("🚁 Preparando drones...")
        for drone in self.drones:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)

        # Decola todos
        for drone in self.drones:
            self.client.takeoffAsync(vehicle_name=drone)
        time.sleep(5)
        print("✅ Drones no ar!\n")

    def apply_weather(self, preset):
        """Aplica configurações de clima"""
        weather_presets = {
            "clear": {"fog": 0, "rain": 0, "snow": 0, "dust": 0},
            "foggy": {"fog": 0.7, "rain": 0, "snow": 0, "dust": 0},
            "rainy": {"fog": 0.2, "rain": 0.8, "snow": 0, "dust": 0},
            "snowy": {"fog": 0.3, "rain": 0, "snow": 0.9, "dust": 0},
            "dusty": {"fog": 0, "rain": 0, "snow": 0, "dust": 0.6},
        }

        if preset in weather_presets:
            w = weather_presets[preset]
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, w["fog"])
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, w["rain"])
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, w["snow"])
            self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, w["dust"])
            return True
        return False

    def apply_time(self, hour):
        """Define horário do dia"""
        time_str = f"2024-01-01 {hour:02d}:00:00"
        self.client.simSetTimeOfDay(True, time_str)

    def move_drones_pattern(self, pattern_name, t):
        """Move drones em diferentes padrões"""
        patterns = {
            "circle": self._pattern_circle,
            "line": self._pattern_line,
            "vformation": self._pattern_vformation,
            "random": self._pattern_random,
            "spiral": self._pattern_spiral
        }

        if pattern_name in patterns:
            positions = patterns[pattern_name](t)
            for drone, pos in zip(self.drones, positions):
                self.client.moveToPositionAsync(
                    pos[0], pos[1], pos[2], 5, vehicle_name=drone
                )

    def _pattern_circle(self, t):
        """Padrão circular"""
        positions = []
        num_drones = len(self.drones)
        for i in range(num_drones):
            angle = (2 * math.pi * i / num_drones) + t
            x = 20 * math.cos(angle)
            y = 20 * math.sin(angle)
            z = -15 - i * 3
            positions.append((x, y, z))
        return positions

    def _pattern_line(self, t):
        """Padrão em linha"""
        positions = []
        for i in range(len(self.drones)):
            x = i * 15 + math.sin(t) * 5
            y = math.cos(t) * 10
            z = -20 - i * 2
            positions.append((x, y, z))
        return positions

    def _pattern_vformation(self, t):
        """Formação em V"""
        positions = []
        for i in range(len(self.drones)):
            if i == 0:
                x, y, z = 10 + t * 2, 0, -25
            else:
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                x = 10 + t * 2 - row * 8
                y = side * row * 8
                z = -25 + row * 2
            positions.append((x, y, z))
        return positions

    def _pattern_random(self, t):
        """Padrão aleatório"""
        np.random.seed(int(t * 100))
        positions = []
        for i in range(len(self.drones)):
            x = np.random.uniform(-30, 30)
            y = np.random.uniform(-30, 30)
            z = np.random.uniform(-10, -30)
            positions.append((x, y, z))
        return positions

    def _pattern_spiral(self, t):
        """Padrão espiral"""
        positions = []
        for i in range(len(self.drones)):
            r = 5 + t * 3 + i * 5
            angle = t * 2 + i * 1.57
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = -10 - t * 2 - i * 2
            positions.append((x, y, z))
        return positions

    def capture_frame(self, scenario_info):
        """Captura imagens e salva metadata"""

        # Captura de cada drone
        for drone in self.drones:
            try:
                # Captura imagem RGB
                png = self.client.simGetImage("0", airsim.ImageType.Scene, vehicle_name=drone)

                if png and len(png) > 1000:
                    # Salva imagem
                    img_filename = f"frame_{self.frame_count:06d}_{drone}.png"
                    img_path = self.output_dir / "images" / img_filename
                    with open(img_path, 'wb') as f:
                        f.write(png)

                    # Metadata do frame
                    state = self.client.getMultirotorState(vehicle_name=drone)
                    pos = state.kinematics_estimated.position
                    ori = state.kinematics_estimated.orientation

                    metadata = {
                        "frame": self.frame_count,
                        "drone": drone,
                        "image": img_filename,
                        "scenario": scenario_info,
                        "position": {
                            "x": pos.x_val,
                            "y": pos.y_val,
                            "z": pos.z_val
                        },
                        "orientation": {
                            "w": ori.w_val,
                            "x": ori.x_val,
                            "y": ori.y_val,
                            "z": ori.z_val
                        },
                        "timestamp": time.time()
                    }

                    # Salva metadata
                    meta_filename = f"frame_{self.frame_count:06d}_{drone}.json"
                    meta_path = self.output_dir / "metadata" / meta_filename
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

            except Exception as e:
                print(f"⚠️ Erro capturando {drone}: {e}")

        self.frame_count += 1

    def generate_dataset(self, total_frames=500):
        """Gera o dataset completo"""

        print("🎬 INICIANDO GERAÇÃO DO DATASET")
        print("="*60)
        print(f"📊 Total de frames planejados: {total_frames}")
        print(f"🚁 Drones: {len(self.drones)}")
        print(f"📁 Salvando em: {self.output_dir.absolute()}\n")

        self.setup_drones()

        # Cenários variados
        scenarios = [
            {"weather": "clear", "time": 12, "pattern": "circle"},
            {"weather": "foggy", "time": 8, "pattern": "line"},
            {"weather": "clear", "time": 18, "pattern": "vformation"},
            {"weather": "snowy", "time": 10, "pattern": "random"},
            {"weather": "dusty", "time": 15, "pattern": "spiral"},
            {"weather": "rainy", "time": 20, "pattern": "circle"},
            {"weather": "clear", "time": 6, "pattern": "line"},
        ]

        frames_per_scenario = total_frames // len(scenarios)

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}")
            print(f"   Clima: {scenario['weather']}")
            print(f"   Horário: {scenario['time']}:00")
            print(f"   Padrão: {scenario['pattern']}")

            # Aplica configurações
            self.apply_weather(scenario['weather'])
            self.apply_time(scenario['time'])

            # Coleta frames
            for frame in range(frames_per_scenario):
                # Calcula tempo para animação
                t = frame / frames_per_scenario * math.pi * 2

                # Move drones
                self.move_drones_pattern(scenario['pattern'], t)
                time.sleep(0.5)  # Aguarda movimento

                # Captura frame
                self.capture_frame(scenario)

                # Progresso
                total_progress = scenario_idx * frames_per_scenario + frame + 1
                pct = (total_progress / total_frames) * 100
                print(f"   📸 Frame {total_progress}/{total_frames} ({pct:.1f}%)", end='\r')

        # Pousa drones
        print("\n\n🛬 Pousando drones...")
        for drone in self.drones:
            self.client.landAsync(vehicle_name=drone)
        time.sleep(5)

        for drone in self.drones:
            self.client.armDisarm(False, drone)
            self.client.enableApiControl(False, drone)

        # Estatísticas finais
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("✅ DATASET GERADO COM SUCESSO!")
        print("="*60)
        print(f"📊 Total de frames: {self.frame_count}")
        print(f"📁 Imagens salvas: {len(list((self.output_dir / 'images').glob('*.png')))}")
        print(f"⏱️ Tempo total: {elapsed/60:.1f} minutos")
        print(f"📍 Local: {self.output_dir.absolute()}")

        # Cria arquivo de resumo
        summary = {
            "environment": "Mountains",
            "total_frames": self.frame_count,
            "drones": self.drones,
            "scenarios": scenarios,
            "generation_time": elapsed,
            "timestamp": str(datetime.now())
        }

        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    generator = DatasetGenerator()

    # Pergunta quantos frames
    print("💾 GERADOR DE DATASET - MOUNTAINS")
    print("-"*60)
    print("Quantos frames você quer gerar?")
    print("  [1] Teste rápido (50 frames)")
    print("  [2] Dataset pequeno (200 frames)")
    print("  [3] Dataset médio (500 frames)")
    print("  [4] Dataset grande (1000 frames)")
    print("  [5] Personalizado")
    print("-"*60)

    choice = input("Escolha (1-5) [padrão=2]: ") or "2"

    frames_map = {
        "1": 50,
        "2": 200,
        "3": 500,
        "4": 1000
    }

    if choice in frames_map:
        num_frames = frames_map[choice]
    elif choice == "5":
        num_frames = int(input("Número de frames: ") or 200)
    else:
        num_frames = 200

    try:
        generator.generate_dataset(num_frames)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompido pelo usuário")
        print("🛬 Pousando drones...")
        for drone in generator.drones:
            generator.client.landAsync(vehicle_name=drone)
            generator.client.armDisarm(False, drone)
            generator.client.enableApiControl(False, drone)