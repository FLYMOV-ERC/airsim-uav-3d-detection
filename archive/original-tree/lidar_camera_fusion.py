#!/usr/bin/env python3
"""
Fusão LiDAR-Câmera: Projeta nuvem de pontos na imagem RGB
Cria visualização tipo "frustum" com pontos coloridos por profundidade
"""

import numpy as np
import cv2
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import cosysairsim as airsim
import time

class LidarCameraFusion:
    def __init__(self, fov_horizontal=90, fov_vertical=60, img_width=1280, img_height=720):
        """
        Inicializa parâmetros de fusão

        Args:
            fov_horizontal: Campo de visão horizontal em graus
            fov_vertical: Campo de visão vertical em graus
            img_width: Largura da imagem
            img_height: Altura da imagem
        """
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = np.radians(fov_vertical)
        self.img_width = img_width
        self.img_height = img_height

        # Calcula parâmetros intrínsecos da câmera (aproximado)
        self.fx = img_width / (2 * np.tan(self.fov_h / 2))
        self.fy = img_height / (2 * np.tan(self.fov_v / 2))
        self.cx = img_width / 2
        self.cy = img_height / 2

        # Matriz intrínseca da câmera
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        print(f"📷 Parâmetros da câmera:")
        print(f"   FOV: {fov_horizontal}° x {fov_vertical}°")
        print(f"   Resolução: {img_width}x{img_height}")
        print(f"   Focal: fx={self.fx:.1f}, fy={self.fy:.1f}")

    def project_points_to_image(self, points_3d):
        """
        Projeta pontos 3D para coordenadas de imagem 2D

        No AirSim:
        - X: frente (positivo = à frente do drone)
        - Y: direita (positivo = à direita)
        - Z: baixo (positivo = abaixo)

        Para câmera:
        - u (horizontal): direita
        - v (vertical): baixo
        """
        if len(points_3d) == 0:
            return np.array([]), np.array([])

        # Filtra pontos atrás da câmera (X <= 0)
        valid_mask = points_3d[:, 0] > 0.1  # Pontos à frente
        points_valid = points_3d[valid_mask]

        if len(points_valid) == 0:
            return np.array([]), np.array([])

        # Converte para sistema de coordenadas da câmera
        # Câmera olha para +X, com Y para direita e Z para baixo
        x_cam = points_valid[:, 0]  # Profundidade (frente)
        y_cam = points_valid[:, 1]  # Lateral (direita)
        z_cam = points_valid[:, 2]  # Vertical (baixo)

        # Projeta para plano da imagem
        u = (self.fx * y_cam / x_cam) + self.cx
        v = (self.fy * z_cam / x_cam) + self.cy

        # Filtra pontos fora da imagem
        in_image = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)

        u = u[in_image].astype(int)
        v = v[in_image].astype(int)
        depths = x_cam[in_image]

        return np.column_stack((u, v)), depths

    def colorize_by_depth(self, depths, min_depth=0, max_depth=50):
        """
        Mapeia profundidade para cores usando colormap
        """
        if len(depths) == 0:
            return np.array([])

        # Normaliza profundidades
        depths_norm = np.clip((depths - min_depth) / (max_depth - min_depth), 0, 1)

        # Usa colormap (jet: azul=perto, vermelho=longe)
        colormap = cm.get_cmap('jet')
        colors = colormap(depths_norm)[:, :3] * 255  # RGB apenas, sem alpha

        return colors.astype(np.uint8)

    def create_fusion_image(self, img_rgb, points_3d, point_size=3, opacity=0.7):
        """
        Cria imagem fundida com pontos LiDAR projetados

        Args:
            img_rgb: Imagem RGB base
            points_3d: Nuvem de pontos 3D
            point_size: Tamanho dos pontos projetados
            opacity: Opacidade da sobreposição
        """
        # Cria cópia da imagem
        fusion_img = img_rgb.copy()
        overlay = np.zeros_like(img_rgb)

        # Projeta pontos
        points_2d, depths = self.project_points_to_image(points_3d)

        if len(points_2d) > 0:
            # Coloriza por profundidade
            colors = self.colorize_by_depth(depths)

            # Desenha pontos na overlay
            for i, (u, v) in enumerate(points_2d):
                color = colors[i].tolist()
                cv2.circle(overlay, (u, v), point_size, color, -1)

            # Funde com transparência
            fusion_img = cv2.addWeighted(fusion_img, 1-opacity, overlay, opacity, 0)

            # Adiciona informações
            cv2.putText(fusion_img, f"LiDAR Points: {len(points_2d)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(fusion_img, f"Depth Range: {depths.min():.1f}-{depths.max():.1f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Adiciona barra de cores
            self.add_colorbar(fusion_img, depths.min(), depths.max())

        return fusion_img

    def add_colorbar(self, img, min_val, max_val):
        """Adiciona barra de cores indicando escala de profundidade"""
        h, w = img.shape[:2]

        # Cria barra de cores
        bar_width = 30
        bar_height = 200
        bar_x = w - 50
        bar_y = h - bar_height - 50

        # Gradiente de cores
        for i in range(bar_height):
            depth_norm = i / bar_height
            color = cm.get_cmap('jet')(1 - depth_norm)[:3]
            color = tuple([int(c * 255) for c in color])
            cv2.rectangle(img, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i + 1), color, -1)

        # Bordas
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Labels
        cv2.putText(img, f"{max_val:.0f}m", (bar_x - 45, bar_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"{min_val:.0f}m", (bar_x - 45, bar_y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Depth", (bar_x - 15, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def create_frustum_visualization(self, points_3d, img_shape=(720, 1280)):
        """
        Cria visualização 3D do frustum da câmera com pontos
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Filtra pontos no FOV
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        # Pontos no frustum (frente da câmera)
        in_frustum = x > 0
        x_f = x[in_frustum]
        y_f = y[in_frustum]
        z_f = z[in_frustum]

        # Coloriza por profundidade
        if len(x_f) > 0:
            depths = x_f
            colors = cm.jet((depths - depths.min()) / (depths.max() - depths.min() + 0.001))

            # Plota pontos
            scatter = ax.scatter(x_f, y_f, z_f, c=colors, s=1, alpha=0.5)

            # Desenha frustum da câmera
            max_depth = min(50, x_f.max())

            # Calcula corners do frustum
            aspect = img_shape[1] / img_shape[0]
            h_half = max_depth * np.tan(self.fov_h / 2)
            v_half = max_depth * np.tan(self.fov_v / 2)

            # Corners do frustum na profundidade máxima
            corners = np.array([
                [max_depth, -h_half, -v_half],  # Top-left
                [max_depth, h_half, -v_half],   # Top-right
                [max_depth, h_half, v_half],    # Bottom-right
                [max_depth, -h_half, v_half],   # Bottom-left
            ])

            # Origem (posição da câmera)
            origin = np.array([0, 0, 0])

            # Desenha linhas do frustum
            for corner in corners:
                ax.plot([origin[0], corner[0]],
                       [origin[1], corner[1]],
                       [origin[2], corner[2]], 'g-', alpha=0.3)

            # Conecta corners
            for i in range(4):
                j = (i + 1) % 4
                ax.plot([corners[i, 0], corners[j, 0]],
                       [corners[i, 1], corners[j, 1]],
                       [corners[i, 2], corners[j, 2]], 'g-', alpha=0.3)

            # Marca origem
            ax.scatter([0], [0], [0], c='red', s=100, marker='o', label='Camera')

            # Labels
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (Right)')
            ax.set_zlabel('Z (Down)')
            ax.set_title('Camera Frustum with LiDAR Points')

            # Ajusta visualização
            ax.view_init(elev=20, azim=45)

            # Limites
            ax.set_xlim([0, max_depth])
            ax.set_ylim([-h_half, h_half])
            ax.set_zlim([-v_half, v_half])

            plt.colorbar(scatter, ax=ax, label='Depth (m)')

        return fig


def process_dataset_with_fusion():
    """Processa dataset existente e cria visualizações fundidas"""

    print("\n" + "="*60)
    print("🔄 FUSÃO LIDAR-CÂMERA COM PROJEÇÃO")
    print("="*60)

    # Diretórios
    dataset_dir = Path("dataset_lidar_final")
    output_dir = Path("lidar_camera_fusion_output")
    output_dir.mkdir(exist_ok=True)

    # Inicializa fusão
    fusion = LidarCameraFusion()

    # Processa frames disponíveis
    rgb_files = sorted(dataset_dir.glob("rgb/*.png"))

    if not rgb_files:
        print("❌ Nenhuma imagem RGB encontrada!")
        return

    print(f"\n📁 Processando {len(rgb_files)} frames...")

    for i, rgb_file in enumerate(rgb_files[:10]):  # Processa até 10 frames
        frame_num = rgb_file.stem.split('_')[1]

        print(f"\n🖼️ Frame {frame_num}:")

        # Carrega imagem RGB
        img_rgb = cv2.imread(str(rgb_file))

        # Carrega pontos LiDAR filtrados
        lidar_file = dataset_dir / f"lidar_filtered/frame_{frame_num}.npy"
        if not lidar_file.exists():
            print(f"   ⚠️ LiDAR não encontrado para frame {frame_num}")
            continue

        points_3d = np.load(lidar_file)
        print(f"   📊 Pontos LiDAR: {len(points_3d)}")

        # Cria fusão
        fusion_img = fusion.create_fusion_image(img_rgb, points_3d, point_size=3, opacity=0.7)

        # Salva resultado
        output_file = output_dir / f"fusion_{frame_num}.png"
        cv2.imwrite(str(output_file), fusion_img)
        print(f"   ✅ Salvo: {output_file}")

        # Cria visualização 3D do frustum para o primeiro frame
        if i == 0:
            print("   📐 Criando visualização 3D do frustum...")
            fig = fusion.create_frustum_visualization(points_3d)
            frustum_file = output_dir / "frustum_3d.png"
            plt.savefig(frustum_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Frustum salvo: {frustum_file}")

    print("\n" + "="*60)
    print("✅ FUSÃO COMPLETA!")
    print("="*60)
    print(f"\n📁 Visualizações salvas em: {output_dir.absolute()}")
    print("\n📝 Conteúdo gerado:")
    print("   • fusion_*.png - Imagens RGB com pontos LiDAR projetados")
    print("   • frustum_3d.png - Visualização 3D do frustum da câmera")
    print("\n🎨 Código de cores:")
    print("   • Azul: Objetos próximos")
    print("   • Verde/Amarelo: Distância média")
    print("   • Vermelho: Objetos distantes")


def capture_and_fuse_realtime():
    """Captura e funde em tempo real do AirSim"""

    print("\n" + "="*60)
    print("🔄 CAPTURA E FUSÃO EM TEMPO REAL")
    print("="*60)

    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Drones: {vehicles}")

    # Prepara Ego
    client.enableApiControl(True, "Ego")
    client.armDisarm(True, "Ego")

    # Decola
    print("🛫 Decolando...")
    client.takeoffAsync(vehicle_name="Ego").join()
    client.moveToPositionAsync(0, 0, -10, 5, vehicle_name="Ego").join()

    # Posiciona outros drones
    if "Drone3" in vehicles:
        client.enableApiControl(True, "Drone3")
        client.armDisarm(True, "Drone3")
        client.takeoffAsync(vehicle_name="Drone3")
        client.moveToPositionAsync(15, 5, -1, 5, vehicle_name="Drone3")

    if "Drone4" in vehicles:
        client.enableApiControl(True, "Drone4")
        client.armDisarm(True, "Drone4")
        client.takeoffAsync(vehicle_name="Drone4")
        client.moveToPositionAsync(20, -5, -1.5, 5, vehicle_name="Drone4")

    time.sleep(3)

    # Inicializa fusão
    fusion = LidarCameraFusion()

    output_dir = Path("realtime_fusion")
    output_dir.mkdir(exist_ok=True)

    print("\n📸 Capturando e fundindo...")

    for i in range(5):
        print(f"\n🔄 Captura {i+1}/5:")

        # Captura RGB
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Captura LiDAR
            lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

            if lidar_data and len(lidar_data.point_cloud) > 3:
                points_raw = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

                # Filtra FOV
                x = points_raw[:, 0]
                y = points_raw[:, 1]
                in_front = x > 0
                angles = np.degrees(np.arctan2(y, x))
                in_fov = (angles >= -45) & (angles <= 45)
                points_filtered = points_raw[in_front & in_fov]

                print(f"   Pontos LiDAR: {len(points_filtered)}")

                # Cria fusão
                fusion_img = fusion.create_fusion_image(img_bgr, points_filtered)

                # Salva
                output_file = output_dir / f"realtime_fusion_{i:03d}.png"
                cv2.imwrite(str(output_file), fusion_img)
                print(f"   ✅ Salvo: {output_file}")

                # Salva também imagem original para comparação
                original_file = output_dir / f"original_{i:03d}.png"
                cv2.imwrite(str(original_file), img_bgr)

        # Rotaciona Ego para variar vista
        client.rotateToYawAsync((i+1) * 30, vehicle_name="Ego")
        time.sleep(1)

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

    print("\n✅ CAPTURA E FUSÃO COMPLETAS!")
    print(f"📁 Resultados em: {output_dir.absolute()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "realtime":
        # Modo tempo real
        capture_and_fuse_realtime()
    else:
        # Processa dataset existente
        process_dataset_with_fusion()