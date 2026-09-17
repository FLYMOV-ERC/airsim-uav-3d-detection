#!/usr/bin/env python3
"""
Coleta automática de dados em múltiplos ambientes AirSim
Instrui quando trocar de ambiente e coleta dados de cada um
"""

import airsim
import time
import json
import os
from pathlib import Path
from datetime import datetime

class MultiEnvironmentCollector:
    def __init__(self):
        self.output_base = Path("dataset_multi_env")
        self.output_base.mkdir(exist_ok=True)

    def wait_for_environment(self, env_name: str):
        """Aguarda usuário iniciar novo ambiente"""
        print("\n" + "="*60)
        print(f"🚀 PRÓXIMO AMBIENTE: {env_name}")
        print("="*60)
        print(f"1. No Windows: FECHE o ambiente atual")
        print(f"2. EXECUTE o {env_name}.exe")
        print(f"3. AGUARDE carregar completamente")
        print(f"4. Pressione ENTER aqui quando pronto...")
        input()

    def test_connection(self):
        """Testa conexão com ambiente atual"""
        try:
            client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
            client.confirmConnection()
            return client
        except:
            return None

    def collect_environment_data(self, env_name: str, num_frames: int = 50):
        """Coleta dados de um ambiente específico"""
        print(f"\n📊 Coletando dados do ambiente: {env_name}")

        # Conecta
        client = self.test_connection()
        if not client:
            print(f"❌ Não consegui conectar ao {env_name}")
            return False

        # Cria diretório para este ambiente
        env_dir = self.output_base / env_name.lower()
        env_dir.mkdir(exist_ok=True)

        # Descobre drones
        drones = client.listVehicles()
        print(f"✅ Conectado! Drones: {drones}")

        # Prepara drones
        for drone in drones:
            client.enableApiControl(True, drone)
            client.armDisarm(True, drone)

        # Decola
        for drone in drones:
            client.takeoffAsync(vehicle_name=drone)
        time.sleep(5)

        # Coleta dados com variações
        scenarios = ["clear", "foggy", "night"]
        frames_per_scenario = num_frames // len(scenarios)

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"\n🎬 Cenário {scenario_idx+1}/{len(scenarios)}: {scenario}")

            # Aplica configurações do cenário
            if scenario == "clear":
                client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
                client.simSetTimeOfDay(True, "2024-01-01 12:00:00")
            elif scenario == "foggy":
                client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.8)
                client.simSetTimeOfDay(True, "2024-01-01 08:00:00")
            elif scenario == "night":
                client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
                client.simSetTimeOfDay(True, "2024-01-01 22:00:00")

            # Move drones em padrão
            for i, drone in enumerate(drones):
                x = i * 10 * (scenario_idx + 1)
                y = i * 5
                z = -10 - (scenario_idx * 5)
                client.moveToPositionAsync(x, y, z, 5, vehicle_name=drone)

            time.sleep(3)

            # Coleta frames
            for frame in range(frames_per_scenario):
                frame_data = {
                    "environment": env_name,
                    "scenario": scenario,
                    "frame": frame,
                    "timestamp": time.time(),
                    "drones": {}
                }

                # Coleta posições
                for drone in drones:
                    state = client.getMultirotorState(vehicle_name=drone)
                    pos = state.kinematics_estimated.position
                    frame_data["drones"][drone] = {
                        "x": pos.x_val,
                        "y": pos.y_val,
                        "z": pos.z_val
                    }

                # Salva metadata
                filename = f"{env_name}_{scenario}_{frame:04d}.json"
                with open(env_dir / filename, 'w') as f:
                    json.dump(frame_data, f)

                print(f"   Frame {frame+1}/{frames_per_scenario}", end='\r')
                time.sleep(0.1)

        # Pousa drones
        print(f"\n🛬 Pousando drones...")
        for drone in drones:
            client.landAsync(vehicle_name=drone)
        time.sleep(5)

        # Desarma
        for drone in drones:
            client.armDisarm(False, drone)
            client.enableApiControl(False, drone)

        print(f"✅ Coleta do {env_name} completa! Dados em: {env_dir}")
        return True

# ============= EXECUÇÃO PRINCIPAL =============

def main():
    print("🎮 COLETA AUTOMÁTICA MULTI-AMBIENTE AIRSIM")
    print("="*60)

    collector = MultiEnvironmentCollector()

    # Lista de ambientes para coletar
    environments = [
        "Blocks",      # Ambiente básico
        "AirSimNH",    # Neighborhood (se você baixou este)
        # "Mountains",   # Descomente se baixou
        # "City",        # Descomente se baixou
    ]

    print(f"\n📋 Vamos coletar dados de {len(environments)} ambientes:")
    for env in environments:
        print(f"   - {env}")

    print("\n⚠️  IMPORTANTE: Você precisa trocar manualmente entre ambientes!")
    print("Vamos começar...\n")

    results = {}

    for env in environments:
        # Pede para usuário trocar ambiente
        collector.wait_for_environment(env)

        # Coleta dados
        success = collector.collect_environment_data(env, num_frames=60)
        results[env] = "✅ Sucesso" if success else "❌ Falhou"

    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DA COLETA:")
    print("="*60)
    for env, status in results.items():
        print(f"{status} {env}")
    print(f"\n📁 Todos os dados salvos em: {collector.output_base.absolute()}")

if __name__ == "__main__":
    main()