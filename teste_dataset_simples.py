#!/usr/bin/env python3
"""
Script simples para testar geração de dataset com AirSim
"""

import os
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# Tenta importar airsim
try:
    import airsim
    print("✅ Módulo airsim importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar airsim: {e}")
    print("Tentando cosysairsim...")
    try:
        import cosysairsim as airsim
        print("✅ Módulo cosysairsim importado")
    except ImportError as e2:
        print(f"❌ Erro ao importar cosysairsim: {e2}")
        exit(1)

def main():
    print("\n" + "="*50)
    print("🚀 TESTE DE CAPTURA DE DATASET AIRSIM")
    print("="*50)

    # Configurações
    ip = "172.19.80.1"
    port = 41451
    output_dir = Path("dataset_teste_novo")

    # Cria diretórios
    output_dir.mkdir(exist_ok=True)
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "segmentation").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    print(f"\n📁 Diretório de saída: {output_dir.absolute()}")
    print(f"🔌 Tentando conectar em {ip}:{port}...")

    try:
        # Conecta ao AirSim com timeout
        client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=5)

        # Tenta confirmar conexão
        print("   Confirmando conexão...")
        client.confirmConnection()
        print("✅ Conectado ao AirSim!")

        # Lista veículos disponíveis
        print("\n📡 Verificando drones disponíveis...")
        vehicles = client.listVehicles()

        if not vehicles:
            print("⚠️  Nenhum drone encontrado! Tentando com drone padrão 'Drone0'")
            vehicles = ["Drone0"]
        else:
            print(f"✅ Drones encontrados: {vehicles}")

        # Usa o primeiro drone disponível
        drone_name = vehicles[0] if vehicles else ""
        print(f"\n🎮 Usando drone: '{drone_name}'")

        # Tenta habilitar controle
        print("   Habilitando controle API...")
        client.enableApiControl(True, vehicle_name=drone_name)

        print("   Armando motores...")
        client.armDisarm(True, vehicle_name=drone_name)

        print("✅ Drone preparado!")

        # Tenta decolar (sem esperar muito)
        print("\n🛫 Tentando decolar...")
        takeoff_task = client.takeoffAsync(vehicle_name=drone_name)
        takeoff_task.join(timeout_sec=5)  # Timeout de 5 segundos

        print("⏱️  Aguardando estabilização...")
        time.sleep(2)

        # Captura algumas imagens
        print("\n📸 Iniciando captura de imagens...")
        num_frames = 5

        for i in range(num_frames):
            print(f"\n   Frame {i+1}/{num_frames}:")

            # Requisições de imagem
            requests = [
                # RGB
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                # Segmentação
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ]

            print("      Solicitando imagens...")
            responses = client.simGetImages(requests, vehicle_name=drone_name)

            # Processa RGB
            if responses[0].image_data_uint8:
                img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                filename = output_dir / "rgb" / f"frame_{i:04d}.png"
                cv2.imwrite(str(filename), img_bgr)
                print(f"      ✅ RGB salva: {filename.name}")
            else:
                print("      ⚠️  Sem dados RGB")

            # Processa Segmentação
            if len(responses) > 1 and responses[1].image_data_uint8:
                img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                filename = output_dir / "segmentation" / f"frame_{i:04d}.png"
                cv2.imwrite(str(filename), img_seg)
                print(f"      ✅ Segmentação salva: {filename.name}")
            else:
                print("      ⚠️  Sem dados de segmentação")

            # Salva metadados
            metadata = {
                'frame': i,
                'timestamp': datetime.now().isoformat(),
                'drone': drone_name
            }

            # Tenta obter posição
            try:
                state = client.getMultirotorState(vehicle_name=drone_name)
                metadata['position'] = {
                    'x': state.kinematics_estimated.position.x_val,
                    'y': state.kinematics_estimated.position.y_val,
                    'z': state.kinematics_estimated.position.z_val
                }
            except:
                metadata['position'] = None

            # Salva metadata
            import json
            meta_file = output_dir / "metadata" / f"frame_{i:04d}.json"
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"      ✅ Metadata salvo")

            time.sleep(0.5)  # Pequeno delay entre frames

        # Pousa
        print("\n🛬 Pousando drone...")
        land_task = client.landAsync(vehicle_name=drone_name)
        land_task.join(timeout_sec=5)

        # Desabilita
        client.armDisarm(False, vehicle_name=drone_name)
        client.enableApiControl(False, vehicle_name=drone_name)

        # Estatísticas finais
        print("\n" + "="*50)
        print("✅ TESTE COMPLETO!")
        print("="*50)

        rgb_files = list((output_dir / "rgb").glob("*.png"))
        seg_files = list((output_dir / "segmentation").glob("*.png"))
        meta_files = list((output_dir / "metadata").glob("*.json"))

        print(f"\n📊 Arquivos gerados:")
        print(f"   RGB: {len(rgb_files)} imagens")
        print(f"   Segmentação: {len(seg_files)} imagens")
        print(f"   Metadata: {len(meta_files)} arquivos")
        print(f"\n📁 Local: {output_dir.absolute()}")

        # Mostra primeiras imagens capturadas
        if rgb_files:
            print(f"\n🖼️  Primeira imagem RGB: {rgb_files[0].name}")
            print(f"   Tamanho: {os.path.getsize(rgb_files[0])} bytes")

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        print("\n🔍 Possíveis problemas:")
        print("   1. AirSim não está rodando")
        print("   2. IP/Porta incorretos")
        print("   3. Ambiente não carregado (verifique se está em 'Blocks')")
        print("   4. Firewall bloqueando conexão")
        print("   5. Nome do drone incorreto")

        # Tenta diagnosticar
        print("\n🔧 Tentando diagnosticar...")
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((ip, port))
            sock.close()

            if result == 0:
                print(f"   ✅ Porta {port} está aberta em {ip}")
                print("   → AirSim parece estar rodando, mas não responde corretamente")
            else:
                print(f"   ❌ Não foi possível conectar em {ip}:{port}")
                print("   → Verifique se o AirSim está rodando")
        except:
            pass

if __name__ == "__main__":
    main()