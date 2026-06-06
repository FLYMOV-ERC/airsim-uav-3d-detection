#!/usr/bin/env python3
"""
Fusão final otimizada - trabalha com o que o LiDAR consegue detectar
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

def create_enhanced_fusion(img, points):
    """Cria fusão melhorada com destaque para objetos não-chão"""
    h, w = img.shape[:2]
    fusion = img.copy()

    # Calibração da câmera
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    # Filtra pontos frontais
    front_mask = points[:, 0] > 0.5
    points_front = points[front_mask]

    if len(points_front) > 0:
        # Filtra FOV horizontal
        angles = np.degrees(np.arctan2(points_front[:, 1], points_front[:, 0]))
        fov_mask = (angles > -45) & (angles < 45)
        points_fov = points_front[fov_mask]

        if len(points_fov) > 0:
            X = points_fov[:, 0]
            Y = points_fov[:, 1]
            Z = points_fov[:, 2]

            # Projeta na imagem
            u = (fx * Y / X + cx).astype(int)
            v = (fy * Z / X + cy).astype(int)

            # Filtra pontos dentro da imagem
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[valid]
            v = v[valid]
            depths = X[valid]
            heights = Z[valid]

            # Cria máscara de profundidade para blend
            depth_mask = np.zeros((h, w), dtype=np.float32)

            # Desenha pontos com cores baseadas em altura e distância
            for i in range(len(u)):
                # Normaliza profundidade
                depth_norm = min(1.0, depths[i] / 50)

                # Determina cor baseada na altura
                if heights[i] < -2:  # Muito acima
                    color = (255, 0, 255)  # Magenta
                    size = 5
                    alpha = 0.9
                elif heights[i] < 0:  # Pouco acima
                    color = (255, 100, 255)  # Rosa
                    size = 4
                    alpha = 0.8
                elif heights[i] < 2:  # Nível
                    color = (0, 255, 255)  # Amarelo
                    size = 3
                    alpha = 0.7
                elif heights[i] < 5:  # Pouco abaixo
                    color = (0, 200, 255)  # Laranja
                    size = 2
                    alpha = 0.6
                elif heights[i] < 20:  # Chão próximo
                    # Gradiente verde baseado na distância
                    green = int(255 * (1 - depth_norm))
                    color = (0, green, 0)
                    size = 1
                    alpha = 0.4
                else:  # Chão distante
                    color = (50, 150, 50)  # Verde escuro
                    size = 1
                    alpha = 0.3

                # Desenha com transparência simulada
                cv2.circle(fusion, (u[i], v[i]), size, color, -1)
                depth_mask[v[i], u[i]] = max(depth_mask[v[i], u[i]], alpha)

            # Estatísticas
            non_ground = np.sum(heights < 15)
            above_points = np.sum(heights < 0)

            # Interface melhorada
            # Fundo semi-transparente
            overlay = fusion.copy()
            cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
            fusion = cv2.addWeighted(fusion, 0.8, overlay, 0.2, 0)

            # Informações
            cv2.putText(fusion, "LiDAR Fusion Enhanced", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(fusion, f"Total Points: {len(points):,}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(fusion, f"Projected: {len(u):,}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(fusion, f"Non-ground: {non_ground:,}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            if above_points > 0:
                cv2.putText(fusion, f"Above sensor: {above_points}", (20, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Legenda aprimorada
            legend_overlay = fusion.copy()
            cv2.rectangle(legend_overlay, (w-200, h-180), (w-10, h-10), (0, 0, 0), -1)
            fusion = cv2.addWeighted(fusion, 0.85, legend_overlay, 0.15, 0)

            cv2.putText(fusion, "Height Map:", (w-190, h-155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Itens da legenda
            legend_items = [
                (5, (255, 0, 255), "Above"),
                (3, (0, 255, 255), "Level"),
                (2, (0, 200, 255), "Below"),
                (1, (0, 200, 0), "Ground")
            ]

            for idx, (size, color, label) in enumerate(legend_items):
                y_pos = h - 120 + idx * 30
                cv2.circle(fusion, (w-170, y_pos), size, color, -1)
                cv2.putText(fusion, label, (w-150, y_pos+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            return fusion, len(u), non_ground, above_points

    return fusion, 0, 0, 0


def main():
    print("\n" + "="*70)
    print("🚀 FUSÃO FINAL OTIMIZADA")
    print("="*70)

    # Conecta
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"\n✅ Conectado! Veículos: {vehicles}")

    # Prepara todos
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

    # Posiciona em formação ideal
    print("📍 Posicionando drones...")

    # Ego como observador principal
    client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

    # Outros drones em diferentes alturas e distâncias
    positions = [
        ("Drone3", 15, -5, -15, "15m frente, 5m esquerda, altura 15m"),
        ("Drone4", 20, 5, -20, "20m frente, 5m direita, altura 20m"),
        ("Intruder1", 25, 0, -25, "25m frente, centro, altura 25m")
    ]

    for name, x, y, z, desc in positions:
        if name in vehicles:
            client.moveToPositionAsync(x, y, z, 5, vehicle_name=name).join()
            print(f"   {name}: {desc}")

    time.sleep(3)

    output_dir = Path("fusion_final_output")
    output_dir.mkdir(exist_ok=True)

    print("\n📸 CAPTURANDO FUSÃO OTIMIZADA...")

    total_projected = 0
    total_non_ground = 0
    total_above = 0

    for frame in range(5):
        print(f"\n🔄 Frame {frame+1}/5:")

        # Varia ângulo para múltiplas perspectivas
        yaw = (frame - 2) * 15  # -30, -15, 0, 15, 30
        client.rotateToYawAsync(yaw, vehicle_name="Ego").join()
        time.sleep(0.5)

        # Captura imagem
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        img_bgr = None
        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Captura LiDAR
        lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

        if lidar_data and len(lidar_data.point_cloud) > 3 and img_bgr is not None:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            print(f"   📡 Pontos capturados: {len(points):,}")

            # Cria fusão melhorada
            fusion, projected, non_ground, above = create_enhanced_fusion(img_bgr, points)

            print(f"   🎨 Projetados: {projected:,}")
            print(f"   📊 Não-chão: {non_ground:,}")
            if above > 0:
                print(f"   ⬆️ Acima do sensor: {above}")

            total_projected += projected
            total_non_ground += non_ground
            total_above += above

            # Salva resultados
            cv2.imwrite(str(output_dir / f"fusion_{frame:03d}.png"), fusion)
            cv2.imwrite(str(output_dir / f"original_{frame:03d}.png"), img_bgr)

            # Cria comparação lado a lado
            h, w = img_bgr.shape[:2]
            comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
            comparison[:, :w] = img_bgr
            comparison[:, w+20:] = fusion

            # Títulos
            cv2.putText(comparison, "Original Camera", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(comparison, "LiDAR Fusion Enhanced", (w+30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imwrite(str(output_dir / f"comparison_{frame:03d}.png"), comparison)

            # Salva nuvem de pontos
            np.save(output_dir / f"points_{frame:03d}.npy", points)

            print(f"   ✅ Frame {frame+1} salvo!")

    # Volta ao centro
    client.rotateToYawAsync(0, vehicle_name="Ego").join()

    # Estatísticas finais
    print("\n" + "="*70)
    print("📊 ESTATÍSTICAS FINAIS:")
    print("="*70)
    print(f"  • Média pontos projetados: {total_projected//5:,}")
    print(f"  • Média pontos não-chão: {total_non_ground//5:,}")
    if total_above > 0:
        print(f"  • Total pontos acima detectados: {total_above}")

    # Pousa
    print("\n🛬 Pousando...")
    for v in vehicles:
        try:
            client.landAsync(vehicle_name=v)
            client.armDisarm(False, v)
            client.enableApiControl(False, v)
        except:
            pass

    print("\n" + "="*70)
    print("✅ FUSÃO FINAL COMPLETA!")
    print("="*70)
    print(f"\n📁 Resultados salvos em: {output_dir.absolute()}")
    print("\n💡 DICAS:")
    print("  • Se DrawDebugPoints está ativo, reinicie o AirSim")
    print("  • Verifique o ambiente 3D para ver os raios do LiDAR")
    print("  • Pontos magenta = objetos detectados acima")
    print("  • Pontos amarelos = objetos ao nível do sensor")


if __name__ == "__main__":
    main()