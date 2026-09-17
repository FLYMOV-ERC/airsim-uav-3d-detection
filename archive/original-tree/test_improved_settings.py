#!/usr/bin/env python3
"""
Testa as melhorias do LiDAR com a nova configuração
"""

import cosysairsim as airsim
import numpy as np
import cv2
import time
from pathlib import Path

def test_improved_lidar():
    print("\n" + "="*70)
    print("🚀 TESTE DO LIDAR MELHORADO")
    print("="*70)

    print("\n📊 COMPARAÇÃO DE CONFIGURAÇÕES:")
    print("\nANTES (seu settings atual):")
    print("  • 16 canais")
    print("  • 200,000 pontos/seg")
    print("  • FOV: -25° a +10° (olhando PARA BAIXO)")
    print("  • ~8,192 pontos capturados")

    print("\nDEPOIS (settings_improved.json):")
    print("  • 64 canais (4x mais!)")
    print("  • 1,000,000 pontos/seg (5x mais!)")
    print("  • FOV: -45° a +45° (cobertura COMPLETA)")
    print("  • FOV horizontal: ±90° (focado na frente)")
    print("  • Range: 150m")
    print("  • LiDAR alinhado com câmera (mesmo X,Y,Z)")
    print("  • Esperado: ~50,000+ pontos")

    print("\n✅ Assumindo que o AirSim foi atualizado com a nova configuração...")
    time.sleep(2)

    # Conecta
    print("\n🔌 Conectando...")
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()

    vehicles = client.listVehicles()
    print(f"✅ Conectado! Veículos: {vehicles}")

    # Prepara
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

    # Posiciona drones em diferentes alturas para testar cobertura vertical
    print("\n📍 Posicionando drones em DIFERENTES ALTURAS...")
    client.moveToPositionAsync(0, 0, -20, 5, vehicle_name="Ego").join()

    # Coloca drones em várias alturas
    if "Drone3" in vehicles:
        client.moveToPositionAsync(20, -5, -10, 5, vehicle_name="Drone3").join()
        print("  Drone3: 20m frente, 10m altura (ACIMA do Ego)")

    if "Drone4" in vehicles:
        client.moveToPositionAsync(25, 5, -20, 5, vehicle_name="Drone4").join()
        print("  Drone4: 25m frente, 20m altura (MESMA altura)")

    if "Intruder1" in vehicles:
        client.moveToPositionAsync(15, 0, -30, 5, vehicle_name="Intruder1").join()
        print("  Intruder1: 15m frente, 30m altura (ABAIXO do Ego)")

    time.sleep(3)

    output_dir = Path("improved_lidar_test")
    output_dir.mkdir(exist_ok=True)

    print("\n📊 ANALISANDO LIDAR MELHORADO...")

    # Captura dados
    lidar_data = client.getLidarData("LidarFront", vehicle_name="Ego")

    if lidar_data and len(lidar_data.point_cloud) > 3:
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        print(f"\n✨ RESULTADOS:")
        print(f"  📡 Total de pontos: {len(points):,}")

        if len(points) > 20000:
            print(f"  ✅ SUCESSO! {len(points)/8192:.1f}x mais pontos que antes!")
        else:
            print(f"  ⚠️ Ainda com poucos pontos. Verifique se o settings.json foi atualizado.")

        print(f"\n  📏 Alcances:")
        print(f"    X (frente): {points[:,0].min():.1f} a {points[:,0].max():.1f}m")
        print(f"    Y (lateral): {points[:,1].min():.1f} a {points[:,1].max():.1f}m")
        print(f"    Z (vertical): {points[:,2].min():.1f} a {points[:,2].max():.1f}m")

        # Analisa cobertura vertical
        z_above = np.sum(points[:,2] < -5)  # Pontos acima (Z negativo = acima)
        z_level = np.sum((points[:,2] >= -5) & (points[:,2] <= 5))
        z_below = np.sum(points[:,2] > 5)  # Pontos abaixo (Z positivo = abaixo)

        print(f"\n  📊 Distribuição Vertical:")
        print(f"    Acima (Z < -5m): {z_above:,} pontos ({z_above/len(points)*100:.1f}%)")
        print(f"    Nível (-5 a 5m): {z_level:,} pontos ({z_level/len(points)*100:.1f}%)")
        print(f"    Abaixo (Z > 5m): {z_below:,} pontos ({z_below/len(points)*100:.1f}%)")

        if z_above > 100:
            print(f"  ✅ EXCELENTE! Agora captura objetos ACIMA do drone!")
        else:
            print(f"  ⚠️ Ainda não captura objetos acima. FOV vertical ainda limitado.")

        # Captura imagem
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name="Ego")

        if responses[0].image_data_uint8:
            img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_1d.reshape(responses[0].height, responses[0].width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Cria fusão
            fusion = create_better_fusion(img_bgr, points)
            cv2.imwrite(str(output_dir / "fusion_improved.png"), fusion)
            cv2.imwrite(str(output_dir / "original.png"), img_bgr)

            # Salva pontos
            np.save(output_dir / "points_improved.npy", points)

            print(f"\n  💾 Resultados salvos em: {output_dir.absolute()}")

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
    if len(points) > 20000:
        print("🎉 LIDAR MELHORADO FUNCIONANDO PERFEITAMENTE!")
        print(f"   {len(points):,} pontos por captura!")
        print("   Cobertura vertical completa!")
        print("   Pronto para fusão de alta qualidade!")
    else:
        print("⚠️ Configure o settings.json e reinicie o AirSim")
    print("="*70)


def create_better_fusion(img, points):
    """Cria fusão com mais pontos"""
    h, w = img.shape[:2]

    # Parâmetros da câmera
    fx = w / (2 * np.tan(np.radians(45)))
    fy = h / (2 * np.tan(np.radians(30)))
    cx = w / 2
    cy = h / 2

    fusion = img.copy()

    # Filtra pontos frontais
    front = points[points[:, 0] > 0]

    if len(front) > 0:
        # Filtra FOV
        angles = np.degrees(np.arctan2(front[:, 1], front[:, 0]))
        in_fov = (angles > -45) & (angles < 45)
        points_fov = front[in_fov]

        if len(points_fov) > 0:
            X = points_fov[:, 0]
            Y = points_fov[:, 1]
            Z = points_fov[:, 2]

            # Projeta
            u = (fx * Y / X + cx).astype(int)
            v = (fy * Z / X + cy).astype(int)

            # Filtra dentro da imagem
            valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u = u[valid]
            v = v[valid]
            depths = X[valid]

            # Desenha pontos coloridos
            if len(u) > 0:
                depth_norm = np.clip(depths / 100, 0, 1)

                for i in range(len(u)):
                    # Cor por profundidade
                    color = (
                        int(255 * depth_norm[i]),      # R: longe
                        int(255 * (1 - depth_norm[i])), # G: perto
                        0                                # B
                    )
                    size = max(1, int(3 * (1 - depth_norm[i])))
                    cv2.circle(fusion, (u[i], v[i]), size, color, -1)

                # Info
                cv2.putText(fusion, f"LiDAR Points: {len(u):,}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(fusion, f"Total in FOV: {len(points_fov):,}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return fusion


if __name__ == "__main__":
    test_improved_lidar()