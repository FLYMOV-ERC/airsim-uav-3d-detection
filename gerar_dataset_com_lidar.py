#!/usr/bin/env python3
"""
Dataset com Imagens RGB + Nuvem de Pontos LiDAR
Captura sincronizada de câmera e LiDAR 3D
"""

import cosysairsim as airsim
import numpy as np
import cv2
from pathlib import Path
import time
import json
from datetime import datetime
import math
from tqdm import tqdm
import open3d as o3d  # Para visualização 3D (opcional)

def main():
    print("\n" + "="*60)
    print("🚀 DATASET COM NUVEM DE PONTOS 3D (RGB + LiDAR)")
    print("="*60)

    # Configuração
    output_dir = Path("dataset_rgb_lidar")
    output_dir.mkdir(exist_ok=True)

    dirs = {
        'rgb': output_dir / 'rgb',
        'segmentation': output_dir / 'segmentation',
        'depth': output_dir / 'depth',
        'lidar': output_dir / 'lidar',  # Nuvem de pontos
        'lidar_viz': output_dir / 'lidar_viz',  # Visualização do LiDAR
        'metadata': output_dir / 'metadata'
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True)

    print(f"\n📁 Dataset em: {output_dir.absolute()}")

    # Conecta
    print("\n🔌 Conectando ao AirSim...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Drones: {vehicles}")

    # Verifica se LiDAR está disponível
    print("\n📡 Verificando sensores LiDAR...")
    lidar_names = ["LidarSensor1", "LidarFront", "Lidar1", "LiDAR"]
    lidar_available = None

    for lidar_name in lidar_names:
        try:
            # Tenta obter dados do LiDAR
            lidar_data = client.getLidarData(lidar_name, vehicle_name="Ego")
            if lidar_data:
                lidar_available = lidar_name
                print(f"   ✅ LiDAR '{lidar_name}' encontrado!")
                break
        except:
            continue

    if not lidar_available:
        print("   ⚠️ LiDAR não encontrado - usando depth camera para gerar nuvem de pontos")

    # Prepara drones
    print("\n🎮 Preparando drones...")
    for vehicle in vehicles:
        try:
            client.enableApiControl(True, vehicle)
            client.armDisarm(True, vehicle)
            print(f"   ✅ {vehicle}")
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

    # Posiciona Ego
    client.moveToPositionAsync(0, 0, -15, 5, vehicle_name="Ego").join()

    # Configuração
    num_frames = 100
    frames_captured = 0

    print(f"\n📸 Capturando {num_frames} frames com nuvem de pontos 3D")
    print("   RGB: 1280x720 (front_center)")
    if lidar_available:
        print(f"   LiDAR: {lidar_available}")
    else:
        print("   Nuvem de pontos: Gerada de depth map")
    print()

    with tqdm(total=num_frames, desc="Capturando") as pbar:
        for frame_idx in range(num_frames):
            t = frame_idx * 0.1

            # MOVIMENTO DOS DRONES
            num_drones = len(vehicles) - 1
            drone_idx = 0

            for vehicle in vehicles:
                if vehicle == "Ego":
                    # Ego rotaciona lentamente
                    yaw = (frame_idx * 3) % 360
                    client.rotateToYawAsync(yaw, vehicle_name="Ego")

                    # Varia altura
                    ego_z = -15 + 3 * math.sin(t * 0.3)
                    client.moveToZAsync(ego_z, 2, vehicle_name="Ego")
                    continue

                # Movimento circular dos outros drones
                angle = (2 * math.pi * drone_idx / num_drones) + t
                radius = 15 + 5 * math.sin(t * 0.5)
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = -12 + 4 * math.sin(t * 0.7 + drone_idx)

                try:
                    client.moveToPositionAsync(x, y, z, 5, vehicle_name=vehicle)
                except:
                    pass

                drone_idx += 1

            time.sleep(0.3)

            # CAPTURA SINCRONIZADA
            try:
                # 1. CAPTURA IMAGENS
                responses = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
                    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False)
                ], vehicle_name="Ego")

                # RGB
                if responses[0].image_data_uint8:
                    img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    rgb_file = dirs['rgb'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(rgb_file), img_bgr)

                # Segmentação
                if len(responses) > 1 and responses[1].image_data_uint8:
                    img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                    img_seg = img_1d.reshape(responses[1].height, responses[1].width, 3)

                    seg_file = dirs['segmentation'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(seg_file), img_seg)

                # Depth (para backup de nuvem de pontos)
                depth_array = None
                if len(responses) > 2 and responses[2].image_data_float:
                    depth_array = airsim.list_to_2d_float_array(
                        responses[2].image_data_float,
                        responses[2].width,
                        responses[2].height
                    )

                    # Salva visualização do depth
                    depth_viz = np.clip(depth_array / 50.0, 0, 1) * 255
                    depth_viz = cv2.applyColorMap(depth_viz.astype(np.uint8), cv2.COLORMAP_TURBO)
                    depth_file = dirs['depth'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(depth_file), depth_viz)

                # 2. CAPTURA NUVEM DE PONTOS
                point_cloud = None

                if lidar_available:
                    # Usa LiDAR real
                    try:
                        lidar_data = client.getLidarData(lidar_available, vehicle_name="Ego")

                        if lidar_data and len(lidar_data.point_cloud) > 3:
                            # Converte para array de pontos 3D
                            points = np.array(lidar_data.point_cloud, dtype=np.float32)
                            points = points.reshape(-1, 3)  # [N, 3] formato XYZ

                            point_cloud = points
                            pbar.write(f"   LiDAR: {len(points)} pontos capturados")

                    except Exception as e:
                        pbar.write(f"   ⚠️ Erro LiDAR: {e}")

                # Se não tem LiDAR ou falhou, gera do depth map
                if point_cloud is None and depth_array is not None:
                    point_cloud = depth_to_point_cloud(
                        depth_array,
                        width=responses[2].width,
                        height=responses[2].height,
                        fov=90  # Campo de visão da câmera
                    )
                    pbar.write(f"   Depth->Points: {len(point_cloud)} pontos gerados")

                # 3. SALVA NUVEM DE PONTOS
                if point_cloud is not None and len(point_cloud) > 0:
                    # Formato NumPy (.npy) - rápido e eficiente
                    npy_file = dirs['lidar'] / f"frame_{frames_captured:06d}.npy"
                    np.save(npy_file, point_cloud)

                    # Formato PLY (Point Cloud Library) - compatível com mais software
                    ply_file = dirs['lidar'] / f"frame_{frames_captured:06d}.ply"
                    save_point_cloud_ply(point_cloud, ply_file)

                    # Cria visualização 2D da nuvem de pontos
                    viz_img = visualize_point_cloud_2d(point_cloud)
                    viz_file = dirs['lidar_viz'] / f"frame_{frames_captured:06d}.png"
                    cv2.imwrite(str(viz_file), viz_img)

                # 4. METADATA
                metadata = {
                    'frame': frames_captured,
                    'timestamp': datetime.now().isoformat(),
                    'sensors': {
                        'camera': 'front_center',
                        'lidar': lidar_available if lidar_available else 'depth_generated',
                        'point_cloud_size': len(point_cloud) if point_cloud is not None else 0
                    },
                    'vehicles': {}
                }

                # Posições dos veículos
                for vehicle in vehicles:
                    try:
                        state = client.getMultirotorState(vehicle_name=vehicle)
                        metadata['vehicles'][vehicle] = {
                            'position': {
                                'x': float(state.kinematics_estimated.position.x_val),
                                'y': float(state.kinematics_estimated.position.y_val),
                                'z': float(state.kinematics_estimated.position.z_val)
                            },
                            'orientation': {
                                'w': float(state.kinematics_estimated.orientation.w_val),
                                'x': float(state.kinematics_estimated.orientation.x_val),
                                'y': float(state.kinematics_estimated.orientation.y_val),
                                'z': float(state.kinematics_estimated.orientation.z_val)
                            }
                        }
                    except:
                        pass

                meta_file = dirs['metadata'] / f"frame_{frames_captured:06d}.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                frames_captured += 1
                pbar.update(1)

            except Exception as e:
                pbar.write(f"⚠️ Erro frame {frame_idx}: {e}")
                continue

    # Pousa
    print("\n🛬 Pousando...")
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

    # ESTATÍSTICAS
    print("\n" + "="*60)
    print("✅ DATASET RGB + NUVEM DE PONTOS COMPLETO!")
    print("="*60)

    rgb_count = len(list(dirs['rgb'].glob("*.png")))
    seg_count = len(list(dirs['segmentation'].glob("*.png")))
    depth_count = len(list(dirs['depth'].glob("*.png")))
    npy_count = len(list(dirs['lidar'].glob("*.npy")))
    ply_count = len(list(dirs['lidar'].glob("*.ply")))
    viz_count = len(list(dirs['lidar_viz'].glob("*.png")))

    print(f"\n📊 Estatísticas:")
    print(f"   Frames totais: {frames_captured}/{num_frames}")
    print(f"   Imagens RGB: {rgb_count}")
    print(f"   Segmentações: {seg_count}")
    print(f"   Mapas de profundidade: {depth_count}")
    print(f"   Nuvens de pontos (.npy): {npy_count}")
    print(f"   Nuvens de pontos (.ply): {ply_count}")
    print(f"   Visualizações LiDAR: {viz_count}")

    print(f"\n📁 Dataset em: {output_dir.absolute()}")

    print("\n📝 Como usar as nuvens de pontos:")
    print("   - Arquivos .npy: np.load('frame_000000.npy') -> array [N, 3]")
    print("   - Arquivos .ply: Compatível com CloudCompare, MeshLab, Open3D")
    print("   - Visualizações: Projeção 2D colorida da nuvem de pontos")

    # Exemplo de uso
    print("\n📌 Exemplo de código para carregar:")
    print("""
    import numpy as np

    # Carregar nuvem de pontos
    points = np.load('dataset_rgb_lidar/lidar/frame_000000.npy')
    print(f'Pontos 3D: {points.shape}')  # [N, 3] com coordenadas XYZ

    # Carregar imagem RGB correspondente
    import cv2
    img = cv2.imread('dataset_rgb_lidar/rgb/frame_000000.png')
    """)

def depth_to_point_cloud(depth_array, width, height, fov):
    """Converte depth map em nuvem de pontos 3D"""
    # Calcula parâmetros da câmera
    f = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    cx = width / 2.0
    cy = height / 2.0

    # Cria grade de coordenadas
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # Converte para coordenadas 3D
    z = depth_array
    x = (xx - cx) * z / f
    y = (yy - cy) * z / f

    # Filtra pontos inválidos
    valid = (z > 0.1) & (z < 100)  # Entre 10cm e 100m

    # Empilha em array [N, 3]
    points = np.stack([
        x[valid].flatten(),
        y[valid].flatten(),
        z[valid].flatten()
    ], axis=-1)

    return points

def save_point_cloud_ply(points, filename):
    """Salva nuvem de pontos no formato PLY"""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""

    with open(filename, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def visualize_point_cloud_2d(points, img_size=(800, 600)):
    """Cria visualização 2D da nuvem de pontos"""
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    if len(points) == 0:
        return img

    # Projeta pontos no plano XY (vista de cima)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Normaliza para caber na imagem
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = np.zeros_like(x)

    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = np.zeros_like(y)

    if z_max > z_min:
        z_norm = (z - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z)

    # Converte para coordenadas de pixel
    px = (x_norm * (img_size[0] - 20) + 10).astype(int)
    py = (y_norm * (img_size[1] - 20) + 10).astype(int)

    # Usa Z para colorir (mais alto = mais vermelho)
    colors = cv2.applyColorMap((z_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Desenha pontos
    for i in range(len(points)):
        if 0 <= px[i] < img_size[0] and 0 <= py[i] < img_size[1]:
            color = colors[i][0].tolist()
            cv2.circle(img, (px[i], py[i]), 2, color, -1)

    # Adiciona escala
    cv2.putText(img, f"Points: {len(points)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Z: {z_min:.1f}m to {z_max:.1f}m", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido")
    except Exception as e:
        print(f"\n❌ Erro: {e}")