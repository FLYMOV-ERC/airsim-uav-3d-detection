#!/usr/bin/env python3
"""
Fusão LiDAR-Câmera CORRIGIDA
Projeção adequada considerando o sistema de coordenadas correto
"""

import numpy as np
import cv2
import cosysairsim as airsim
import time
from pathlib import Path
import matplotlib.pyplot as plt

class CorrectedLidarCameraFusion:
    def __init__(self):
        # Parâmetros da câmera (1280x720, FOV 90°x60°)
        self.img_width = 1280
        self.img_height = 720
        self.fov_h = np.radians(90)
        self.fov_v = np.radians(60)

        # Calcula parâmetros intrínsecos
        self.fx = self.img_width / (2 * np.tan(self.fov_h / 2))
        self.fy = self.img_height / (2 * np.tan(self.fov_v / 2))
        self.cx = self.img_width / 2
        self.cy = self.img_height / 2

        print(f"📷 Câmera configurada:")
        print(f"   Resolução: {self.img_width}x{self.img_height}")
        print(f"   FOV: {np.degrees(self.fov_h):.0f}° x {np.degrees(self.fov_v):.0f}°")
        print(f"   Focal: fx={self.fx:.1f}, fy={self.fy:.1f}")

    def transform_lidar_to_camera(self, points_lidar):
        """
        Transforma pontos do sistema LiDAR para sistema da câmera

        LiDAR no AirSim (confirmado):
        - X: frente (positivo = à frente)
        - Y: direita (positivo = à direita)
        - Z: baixo (positivo = abaixo do sensor, por isso só vemos o chão!)

        Câmera (convenção típica):
        - X: direita
        - Y: baixo
        - Z: frente

        Transformação necessária:
        - X_lidar -> Z_camera (frente)
        - Y_lidar -> X_camera (direita)
        - Z_lidar -> Y_camera (baixo)
        """

        # IMPORTANTE: O Z do LiDAR é positivo para BAIXO
        # Por isso estamos vendo apenas o chão!
        # Vamos inverter Z para ter objetos acima do sensor

        points_camera = np.column_stack([
            points_lidar[:, 1],   # Y_lidar -> X_camera (direita)
            points_lidar[:, 2],   # Z_lidar -> Y_camera (baixo)
            points_lidar[:, 0]    # X_lidar -> Z_camera (frente)
        ])

        return points_camera

    def project_to_image(self, points_camera):
        """
        Projeta pontos 3D do sistema da câmera para pixels 2D
        """
        # Filtra pontos atrás da câmera
        valid = points_camera[:, 2] > 0.5  # Z > 0.5m (frente)
        points = points_camera[valid]

        if len(points) == 0:
            return np.array([]), np.array([])

        # Projeção perspectiva
        x = points[:, 0]  # lateral
        y = points[:, 1]  # vertical
        z = points[:, 2]  # profundidade

        # Projeta para plano da imagem
        u = (self.fx * x / z) + self.cx
        v = (self.fy * y / z) + self.cy

        # Filtra pontos dentro da imagem
        in_image = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)

        pixels = np.column_stack([u[in_image], v[in_image]]).astype(int)
        depths = z[in_image]

        return pixels, depths

    def create_fusion_image(self, img_rgb, points_lidar):
        """
        Cria imagem fundida com pontos LiDAR projetados corretamente
        """
        # Transforma pontos para sistema da câmera
        points_camera = self.transform_lidar_to_camera(points_lidar)

        # Projeta para imagem
        pixels, depths = self.project_to_image(points_camera)

        # Cria visualização
        fusion_img = img_rgb.copy()

        if len(pixels) > 0:
            # Normaliza profundidades para cores
            depth_min = max(0.5, depths.min())
            depth_max = min(50, depths.max())
            depths_norm = np.clip((depths - depth_min) / (depth_max - depth_min), 0, 1)

            # Mapeia para cores (jet colormap)
            colors = plt.cm.jet(depths_norm)[:, :3] * 255

            # Desenha pontos
            for i, (u, v) in enumerate(pixels):
                # Tamanho baseado na distância (próximo = maior)
                size = max(1, int(5 * (1 - depths_norm[i])))
                color = colors[i].astype(int).tolist()
                cv2.circle(fusion_img, (u, v), size, color, -1)

            # Adiciona informações
            self.add_info_overlay(fusion_img, len(pixels), depth_min, depth_max)

        return fusion_img

    def add_info_overlay(self, img, num_points, depth_min, depth_max):
        """Adiciona informações na imagem"""

        # Painel de informações
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        cv2.putText(img, "LiDAR-Camera Fusion (CORRECTED)", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"Projected Points: {num_points}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Depth Range: {depth_min:.1f} - {depth_max:.1f}m", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Barra de cores
        h, w = img.shape[:2]
        bar_height = 200
        bar_width = 30
        bar_x = w - 60
        bar_y = h - bar_height - 60

        for i in range(bar_height):
            depth_norm = 1 - (i / bar_height)
            color = plt.cm.jet(depth_norm)[:3]
            color_bgr = tuple([int(c * 255) for c in color])
            cv2.line(img, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i), color_bgr, 1)

        cv2.rectangle(img, (bar_x-1, bar_y-1), (bar_x+bar_width+1, bar_y+bar_height+1), (255,255,255), 2)
        cv2.putText(img, f"{depth_max:.0f}m", (bar_x-45, bar_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, f"{depth_min:.0f}m", (bar_x-45, bar_y+bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


def test_corrected_fusion():
    """Testa a fusão corrigida em tempo real"""

    print("\n" + "="*70)
    print("🔧 TESTE DE FUSÃO LIDAR-CÂMERA CORRIGIDA")
    print("="*70)

    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"\n✅ Conectado! Veículos: {vehicles}")

    # Prepara drones
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except:
            pass

    # Decola
    print("\n🛫 Decolando...")
    for v in vehicles:
        try:
            client.takeoffAsync(vehicle_name=v)
        except:
            pass
    time.sleep(5)

    # Posiciona drones para teste
    print("📍 Posicionando drones...")

    # Ego: observador em altura média
    client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

    # Outros drones em diferentes posições e ALTURAS
    # IMPORTANTE: Colocar drones em DIFERENTES ALTURAS para ver se aparecem
    if "Drone3" in vehicles:
        client.moveToPositionAsync(15, -5, -20, 5, vehicle_name="Drone3").join()  # Mesma altura
        print("   Drone3: 15m frente, 5m esquerda, mesma altura")

    if "Drone4" in vehicles:
        client.moveToPositionAsync(20, 0, -15, 5, vehicle_name="Drone4").join()  # Mais alto
        print("   Drone4: 20m frente, centro, 5m acima")

    if "Intruder1" in vehicles:
        client.moveToPositionAsync(25, 5, -25, 5, vehicle_name="Intruder1").join()  # Mais baixo
        print("   Intruder1: 25m frente, 5m direita, 5m abaixo")

    time.sleep(3)

    # Inicializa fusão
    fusion = CorrectedLidarCameraFusion()

    output_dir = Path("corrected_fusion_output")
    output_dir.mkdir(exist_ok=True)

    print("\n📸 Capturando e fundindo...")

    for frame in range(5):
        print(f"\n🔄 Frame {frame+1}/5:")

        # Rotaciona Ego para variar perspectiva
        yaw = frame * 20
        client.rotateToYawAsync(yaw, vehicle_name="Ego")

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

                # Filtra pontos no FOV aproximado da câmera
                x = points_raw[:, 0]  # frente
                y = points_raw[:, 1]  # lateral

                # FOV horizontal
                angles = np.degrees(np.arctan2(y, x))
                in_fov = (x > 0) & (angles > -45) & (angles < 45)
                points_fov = points_raw[in_fov]

                print(f"   Pontos totais: {len(points_raw)}")
                print(f"   Pontos no FOV: {len(points_fov)}")

                # Análise rápida
                if len(points_fov) > 0:
                    z_range = points_fov[:, 2]
                    print(f"   Z range (vertical): {z_range.min():.2f} a {z_range.max():.2f}")

                    # Verifica se há pontos acima do chão
                    above_ground = np.sum(z_range < 0)  # Z negativo = acima no mundo
                    print(f"   Pontos acima do chão (Z<0): {above_ground}")

                # Cria fusão
                fusion_img = fusion.create_fusion_image(img_bgr, points_fov)

                # Salva resultados
                cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)
                cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion_img)

                # Comparação lado a lado
                h, w = img_bgr.shape[:2]
                comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
                comparison[:, :w] = img_bgr
                comparison[:, w+20:] = fusion_img
                cv2.putText(comparison, "Original", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(comparison, "Fusion Corrected", (w+30, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comparison)
                print(f"   ✅ Salvo!")

        time.sleep(0.5)

    # Pousa
    print("\n🛬 Finalizando...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    print("\n" + "="*70)
    print("✅ TESTE COMPLETO!")
    print("="*70)
    print(f"\n📁 Resultados em: {output_dir.absolute()}")
    print("\n⚠️ NOTA IMPORTANTE:")
    print("   O LiDAR do AirSim captura principalmente o CHÃO")
    print("   Por isso vemos muitos pontos do terreno e poucos objetos aéreos")
    print("   Isso é uma limitação da configuração padrão do sensor")


if __name__ == "__main__":
    test_corrected_fusion()