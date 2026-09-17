#!/usr/bin/env python3
"""
Coleta dataset nos 3 ambientes baixados
Guia você passo a passo na troca entre ambientes
"""

import airsim
import numpy as np
import time
import json
import cv2
from pathlib import Path
from datetime import datetime

class CollectorTresAmbientes:
    def __init__(self):
        self.output_dir = Path("dataset_3_ambientes")
        self.output_dir.mkdir(exist_ok=True)
        self.client = None

    def conectar(self):
        """Tenta conectar ao AirSim"""
        try:
            self.client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
            self.client.confirmConnection()
            return True
        except:
            return False

    def coletar_ambiente(self, nome_ambiente: str, num_frames: int = 100):
        """Coleta dados de um ambiente específico"""

        print(f"\n{'='*60}")
        print(f"📊 COLETANDO DADOS: {nome_ambiente}")
        print(f"{'='*60}")

        # Cria pasta para este ambiente
        env_dir = self.output_dir / nome_ambiente.lower().replace(' ', '_')
        env_dir.mkdir(exist_ok=True)

        # Conecta
        if not self.conectar():
            print("❌ Falha na conexão")
            return False

        # Descobre drones
        drones = self.client.listVehicles()
        print(f"✅ Conectado! Drones: {drones}")

        # Prepara drones
        for drone in drones:
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)

        # Decola
        print("🛫 Decolando...")
        for drone in drones:
            self.client.takeoffAsync(vehicle_name=drone)
        time.sleep(5)

        # Define cenários para este ambiente
        cenarios = [
            {"nome": "dia_claro", "fog": 0, "hora": "12:00"},
            {"nome": "neblina", "fog": 0.7, "hora": "08:00"},
            {"nome": "noite", "fog": 0, "hora": "22:00"},
        ]

        frames_por_cenario = num_frames // len(cenarios)
        frame_global = 0

        for cenario in cenarios:
            print(f"\n🎬 Cenário: {cenario['nome']}")

            # Aplica configurações do cenário
            self.client.simSetWeatherParameter(
                airsim.WeatherParameter.Fog, cenario['fog']
            )
            self.client.simSetTimeOfDay(
                True, f"2024-01-01 {cenario['hora']}:00"
            )

            # Coleta frames
            for frame_num in range(frames_por_cenario):
                # Move drones em padrão variado
                t = frame_num / frames_por_cenario * 2 * 3.14159
                for i, drone in enumerate(drones):
                    # Movimento circular
                    radius = 10 + i * 5
                    x = radius * np.cos(t + i * 1.57)
                    y = radius * np.sin(t + i * 1.57)
                    z = -10 - i * 2

                    self.client.moveToPositionAsync(
                        x, y, z, 3, vehicle_name=drone
                    )

                # Coleta dados do frame
                frame_data = {
                    "ambiente": nome_ambiente,
                    "cenario": cenario['nome'],
                    "frame": frame_global,
                    "timestamp": time.time(),
                    "drones": {}
                }

                # Posições dos drones
                for drone in drones:
                    state = self.client.getMultirotorState(vehicle_name=drone)
                    pos = state.kinematics_estimated.position
                    ori = state.kinematics_estimated.orientation

                    frame_data["drones"][drone] = {
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
                        }
                    }

                # Coleta imagens do drone principal (Ego)
                if "Ego" in drones:
                    try:
                        # Request Scene image
                        responses = self.client.simGetImages([
                            airsim.ImageRequest(
                                "front_center",
                                airsim.ImageType.Scene,
                                False, False
                            )
                        ], vehicle_name="Ego")

                        if responses and len(responses) > 0:
                            response = responses[0]
                            if response.image_data_uint8:
                                # Converte para numpy array
                                img = np.frombuffer(
                                    response.image_data_uint8, dtype=np.uint8
                                )
                                img = img.reshape(
                                    response.height, response.width, 3
                                )

                                # Salva imagem
                                img_name = f"frame_{frame_global:06d}_rgb.png"
                                img_path = env_dir / img_name
                                cv2.imwrite(str(img_path), img)
                                frame_data["image_rgb"] = img_name
                    except:
                        pass  # Ignora erro de imagem

                # Salva metadata
                json_name = f"frame_{frame_global:06d}.json"
                with open(env_dir / json_name, 'w') as f:
                    json.dump(frame_data, f, indent=2)

                frame_global += 1
                print(f"   Frame {frame_num+1}/{frames_por_cenario}", end='\r')

                time.sleep(0.1)

        # Pousa drones
        print("\n🛬 Pousando...")
        for drone in drones:
            self.client.landAsync(vehicle_name=drone)
        time.sleep(5)

        # Desarma
        for drone in drones:
            self.client.armDisarm(False, drone)
            self.client.enableApiControl(False, drone)

        print(f"✅ Coleta completa! {frame_global} frames salvos em {env_dir}")
        return True

# ============= EXECUÇÃO PRINCIPAL =============

def main():
    print("🚁 COLETA DE DATASET EM 3 AMBIENTES")
    print("="*60)

    collector = CollectorTresAmbientes()

    # Ambientes para coletar
    ambientes = [
        "Ambiente 1",
        "Ambiente 2",
        "Ambiente 3"
    ]

    print("\n📋 Vamos coletar dados de 3 ambientes diferentes")
    print("⚠️  Você precisará trocar manualmente entre os ambientes\n")

    resultados = []

    for i, ambiente in enumerate(ambientes, 1):
        print("\n" + "🔄"*30)
        print(f"\n🎯 AMBIENTE {i} de {len(ambientes)}")
        print("-"*60)
        print("INSTRUÇÕES:")
        print(f"1. No Windows: FECHE o ambiente anterior (se houver)")
        print(f"2. EXECUTE o Ambiente {i} (.exe)")
        print(f"3. AGUARDE carregar completamente")
        print(f"4. Pressione ENTER aqui quando pronto...")
        print("-"*60)

        input("Pressione ENTER quando o ambiente estiver rodando... ")

        # Pergunta o nome real do ambiente
        nome_real = input(f"Qual o nome deste ambiente? (ex: Neighborhood, City, Mountains): ")

        if not nome_real:
            nome_real = f"Ambiente_{i}"

        # Coleta dados
        sucesso = collector.coletar_ambiente(nome_real, num_frames=90)

        resultados.append({
            "ambiente": nome_real,
            "status": "✅ Sucesso" if sucesso else "❌ Falhou"
        })

    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DA COLETA")
    print("="*60)

    for resultado in resultados:
        print(f"{resultado['status']} - {resultado['ambiente']}")

    print(f"\n📁 Todos os dados salvos em: {collector.output_dir.absolute()}")
    print("\n🎉 COLETA COMPLETA!")

    # Cria arquivo resumo
    resumo = {
        "data_coleta": str(datetime.now()),
        "ambientes": resultados,
        "total_ambientes": len(ambientes),
        "output_dir": str(collector.output_dir.absolute())
    }

    with open(collector.output_dir / "resumo_coleta.json", 'w') as f:
        json.dump(resumo, f, indent=2)

if __name__ == "__main__":
    main()