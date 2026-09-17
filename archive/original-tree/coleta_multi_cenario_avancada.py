#!/usr/bin/env python3
"""
Sistema avançado de coleta de dataset multi-drone e multi-cenário
Suporta número ilimitado de drones e mudanças dinâmicas de ambiente
"""

import airsim
import numpy as np
import os
import json
import time
import math
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiScenarioDataCollector:
    """Coletor avançado de dados multi-drone e multi-cenário"""

    def __init__(self, output_dir: str = "dataset_avancado"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Conecta ao AirSim
        self.client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
        self.client.confirmConnection()

        # Descobre todos os drones disponíveis
        self.drones = self.client.listVehicles()
        print(f"🚁 Drones detectados: {self.drones}")

        # Configurações de cenário
        self.current_scenario = None
        self.frame_count = 0

    def setup_drones(self):
        """Habilita controle API para todos os drones"""
        print("\n🎮 Habilitando controle dos drones...")
        for drone in self.drones:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
        print("✅ Todos os drones prontos!")

    def takeoff_all(self, altitude: float = -10):
        """Decola todos os drones"""
        print(f"\n🛫 Decolando {len(self.drones)} drones...")
        tasks = []
        for drone in self.drones:
            tasks.append(self.client.takeoffAsync(vehicle_name=drone))

        # Aguarda todos decolarem
        for task in tasks:
            task.join()

        # Move para altitude desejada
        tasks = []
        for drone in self.drones:
            pos = self.client.getMultirotorState(drone).kinematics_estimated.position
            tasks.append(
                self.client.moveToPositionAsync(
                    pos.x_val, pos.y_val, altitude, 5, vehicle_name=drone
                )
            )

        for task in tasks:
            task.join()
        print("✅ Todos no ar!")

    # ============= CENÁRIOS PREDEFINIDOS =============

    def cenario_inspecao(self, alvo_pos: Tuple[float, float, float] = (0, 0, -20)):
        """Cenário de inspeção: drones circulam objeto"""
        print("\n🔍 Cenário: INSPEÇÃO DE ESTRUTURA")

        num_drones = len(self.drones)
        radius = 20
        height_variation = 5

        for i, drone in enumerate(self.drones):
            angle = (2 * math.pi * i) / num_drones
            x = alvo_pos[0] + radius * math.cos(angle)
            y = alvo_pos[1] + radius * math.sin(angle)
            z = alvo_pos[2] + random.uniform(-height_variation, height_variation)

            self.client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone)

    def cenario_busca_resgate(self, area_size: float = 100):
        """Cenário de busca: padrão de grade"""
        print("\n🔦 Cenário: BUSCA E RESGATE")

        num_drones = len(self.drones)
        grid_size = int(math.sqrt(num_drones))
        spacing = area_size / grid_size

        drone_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if drone_idx < num_drones:
                    x = -area_size/2 + i * spacing
                    y = -area_size/2 + j * spacing
                    z = -15 + random.uniform(-3, 3)

                    self.client.moveToPositionAsync(
                        x, y, z, 5, vehicle_name=self.drones[drone_idx]
                    )
                    drone_idx += 1

    def cenario_formacao_v(self):
        """Cenário de formação em V"""
        print("\n✈️ Cenário: FORMAÇÃO EM V")

        leader_pos = (20, 0, -15)
        spacing = 10
        angle = 30  # graus

        for i, drone in enumerate(self.drones):
            if i == 0:
                # Líder
                x, y, z = leader_pos
            else:
                # Seguidores
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                x = leader_pos[0] - row * spacing * math.cos(math.radians(angle))
                y = leader_pos[1] + side * row * spacing * math.sin(math.radians(angle))
                z = leader_pos[2] + random.uniform(-2, 2)

            self.client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone)

    def cenario_enxame_aleatorio(self, center: Tuple = (0, 0, -20), radius: float = 30):
        """Cenário de enxame aleatório"""
        print("\n🐝 Cenário: ENXAME ALEATÓRIO")

        for drone in self.drones:
            x = center[0] + random.uniform(-radius, radius)
            y = center[1] + random.uniform(-radius, radius)
            z = center[2] + random.uniform(-radius/2, radius/2)

            self.client.moveToPositionAsync(x, y, z, 3, vehicle_name=drone)

    # ============= VARIAÇÕES AMBIENTAIS =============

    def set_weather(self, weather_type: str):
        """Define condições climáticas"""
        weather_params = {
            "clear": {"Fog": 0, "Rain": 0, "Snow": 0, "Dust": 0},
            "foggy": {"Fog": 0.7, "Rain": 0, "Snow": 0, "Dust": 0},
            "rainy": {"Fog": 0.2, "Rain": 0.8, "Snow": 0, "Dust": 0},
            "snowy": {"Fog": 0.3, "Rain": 0, "Snow": 0.9, "Dust": 0},
            "dusty": {"Fog": 0, "Rain": 0, "Snow": 0, "Dust": 0.8},
        }

        if weather_type in weather_params:
            print(f"🌤️ Mudando clima para: {weather_type}")
            params = weather_params[weather_type]

            self.client.simSetWeatherParameter(
                airsim.WeatherParameter.Fog, params["Fog"]
            )
            self.client.simSetWeatherParameter(
                airsim.WeatherParameter.Rain, params["Rain"]
            )
            self.client.simSetWeatherParameter(
                airsim.WeatherParameter.Snow, params["Snow"]
            )
            self.client.simSetWeatherParameter(
                airsim.WeatherParameter.Dust, params["Dust"]
            )

    def set_time_of_day(self, hour: int):
        """Define horário do dia (0-23)"""
        print(f"🕐 Mudando horário para: {hour:02d}:00")
        time_string = f"2024-01-01 {hour:02d}:00:00"
        self.client.simSetTimeOfDay(True, time_string)

    # ============= COLETA DE DADOS =============

    def collect_frame_data(self):
        """Coleta dados de todos os sensores de todos os drones"""
        frame_data = {
            "timestamp": time.time(),
            "frame": self.frame_count,
            "scenario": self.current_scenario,
            "drones": {}
        }

        for drone in self.drones:
            # Estado do drone
            state = self.client.getMultirotorState(vehicle_name=drone)

            drone_data = {
                "position": {
                    "x": state.kinematics_estimated.position.x_val,
                    "y": state.kinematics_estimated.position.y_val,
                    "z": state.kinematics_estimated.position.z_val
                },
                "orientation": {
                    "w": state.kinematics_estimated.orientation.w_val,
                    "x": state.kinematics_estimated.orientation.x_val,
                    "y": state.kinematics_estimated.orientation.y_val,
                    "z": state.kinematics_estimated.orientation.z_val
                }
            }

            # Coleta imagens se o drone tiver câmeras
            if drone == "Ego":  # Assumindo que Ego tem câmeras
                responses = self.client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
                ], vehicle_name=drone)

                # Salva imagens
                for idx, response in enumerate(responses):
                    if response.pixels_as_float:
                        img = airsim.get_pfm_array(response)
                    else:
                        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                        img = img.reshape(response.height, response.width, 3)

                    img_type = "rgb" if idx == 0 else "seg"
                    filename = f"{drone}_{img_type}_{self.frame_count:06d}.png"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), img)

                    drone_data[f"image_{img_type}"] = filename

            frame_data["drones"][drone] = drone_data

        # Salva metadata
        metadata_file = self.output_dir / f"frame_{self.frame_count:06d}.json"
        with open(metadata_file, 'w') as f:
            json.dump(frame_data, f, indent=2)

        self.frame_count += 1
        return frame_data

    # ============= SEQUÊNCIAS DE COLETA =============

    def run_sequence(self, num_frames: int = 100, scenarios: List[str] = None):
        """Executa sequência de coleta com múltiplos cenários"""

        if scenarios is None:
            scenarios = ["inspecao", "busca", "formacao", "enxame"]

        self.setup_drones()
        self.takeoff_all()

        frames_per_scenario = num_frames // len(scenarios)

        for scenario in scenarios:
            self.current_scenario = scenario

            # Aplica cenário
            if scenario == "inspecao":
                self.cenario_inspecao()
            elif scenario == "busca":
                self.cenario_busca_resgate()
            elif scenario == "formacao":
                self.cenario_formacao_v()
            elif scenario == "enxame":
                self.cenario_enxame_aleatorio()

            # Varia condições ambientais
            weather_options = ["clear", "foggy", "rainy"]
            time_options = [8, 12, 18, 22]  # Manhã, meio-dia, tarde, noite

            for frame in range(frames_per_scenario):
                # Muda clima a cada 20 frames
                if frame % 20 == 0:
                    self.set_weather(random.choice(weather_options))

                # Muda horário a cada 30 frames
                if frame % 30 == 0:
                    self.set_time_of_day(random.choice(time_options))

                # Coleta dados
                self.collect_frame_data()

                # Pequeno movimento aleatório
                if frame % 5 == 0:
                    for drone in self.drones:
                        state = self.client.getMultirotorState(drone)
                        pos = state.kinematics_estimated.position

                        # Adiciona pequena perturbação
                        new_x = pos.x_val + random.uniform(-2, 2)
                        new_y = pos.y_val + random.uniform(-2, 2)
                        new_z = pos.z_val + random.uniform(-1, 1)

                        self.client.moveToPositionAsync(
                            new_x, new_y, new_z, 2, vehicle_name=drone
                        )

                print(f"📸 Frame {self.frame_count}/{num_frames} - Cenário: {scenario}")
                time.sleep(0.1)  # Pequena pausa

        # Pousa todos os drones
        print("\n🛬 Pousando todos os drones...")
        for drone in self.drones:
            self.client.landAsync(vehicle_name=drone)

        print(f"\n✅ Coleta completa! {self.frame_count} frames salvos em {self.output_dir}")

# ============= EXECUÇÃO PRINCIPAL =============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coleta avançada multi-drone multi-cenário")
    parser.add_argument("--frames", type=int, default=200, help="Número de frames")
    parser.add_argument("--output", type=str, default="dataset_avancado", help="Diretório de saída")
    parser.add_argument("--scenarios", nargs="+", default=["inspecao", "busca", "formacao", "enxame"],
                       help="Cenários a executar")

    args = parser.parse_args()

    # Cria coletor e executa
    collector = MultiScenarioDataCollector(output_dir=args.output)

    try:
        collector.run_sequence(num_frames=args.frames, scenarios=args.scenarios)
    except KeyboardInterrupt:
        print("\n⚠️ Coleta interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
    finally:
        # Garante que drones pousem
        for drone in collector.drones:
            collector.client.landAsync(vehicle_name=drone)
            collector.client.armDisarm(False, drone)
            collector.client.enableApiControl(False, drone)