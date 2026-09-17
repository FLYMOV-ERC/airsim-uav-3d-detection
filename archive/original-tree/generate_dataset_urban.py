#!/usr/bin/env python3
"""
URBAN DATASET GENERATOR - Urban Air Mobility (UAM) scenarios
============================================================

Generalização do generate_dataset_1000.py para 3 ambientes urbanos
(CityEnviron, AirSimNH, CityPark), com posicionamento collision-aware
e domain randomization (weather + time of day).

Mantém a mesma saída do v1: YOLO (1 classe = drone) + PointNet (point clouds + 3D boxes).
Roda 1x por ambiente — o usuário troca o executável Unreal e re-roda com --env diferente.

Uso:
    # 1. Subir CityEnviron.exe no Windows  ->  rodar:
    python generate_dataset_urban.py --env city --frames 1000 --output dataset_urban_city

    # 2. Subir AirSimNH.exe  ->  rodar:
    python generate_dataset_urban.py --env neighborhood --frames 1000 --output dataset_urban_nh

    # 3. Subir CityPark.exe  ->  rodar:
    python generate_dataset_urban.py --env citypark --frames 1000 --output dataset_urban_park

    # 4. Depois mergear os 3 outputs (script separado, ou simplesmente concatenar):
    #    cat dataset_urban_*/yolo/labels/train/*.txt   etc.

Técnicas chaves (todas validadas em scripts anteriores do projeto):
  - simGetDetections + simAddDetectionFilterMeshName  (mesma da v1, ground-truth)
  - simSetVehiclePose para teleporte rápido (mais rápido que moveToPositionAsync)
  - simGetCollisionInfo com timestamp diff para detectar colisão NOVA pós-teleport
  - simSetWeatherParameter + simSetTimeOfDay para variação ambiental
  - Sampling com viés para o FOV horizontal do Ego (drones aparecem na imagem)
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cosysairsim as airsim
from cosysairsim.types import DetectionInfo, ImageResponse
import cv2
import numpy as np

from drone_visual import DroneVisualizer


# =============================================================================
# COMPAT SHIM para binários Microsoft AirSim 1.8.x (AirSimNH/CityEnviron).
# Vários endpoints adicionaram um `external: bool` que cosysairsim 3.3.0 não
# envia (ou envia como string `annotation_name`, causando "bad cast").
# Sondamos cada método uma vez e descobrimos:
#   - simGetImages              : precisa external=False no slot 3
#   - simClearDetectionMeshNames: external=False no slot 4
#   - simSetDetectionFilterRadius: external=False no slot 5
#   - simAddDetectionFilterMeshName: external=False no slot 5
#   - simGetDetections          : external=False no slot 4
# Demais métodos (weather, pose, collision) funcionam direto pelo wrapper cosys.
# =============================================================================
def legacy_clear_detection(client, cam, img_type, veh="", external=False):
    client.client.call('simClearDetectionMeshNames', cam, img_type, veh, external)


def legacy_set_radius(client, cam, img_type, radius_cm, veh="", external=False):
    client.client.call('simSetDetectionFilterRadius', cam, img_type, radius_cm, veh, external)


def legacy_add_mesh(client, cam, img_type, mesh, veh="", external=False):
    client.client.call('simAddDetectionFilterMeshName', cam, img_type, mesh, veh, external)


def legacy_get_detections(client, cam, img_type, veh="", external=False):
    raw = client.client.call('simGetDetections', cam, img_type, veh, external)
    return [DetectionInfo.from_msgpack(r) for r in raw]


def legacy_get_images(client, requests, veh="", external=False):
    raw = client.client.call('simGetImages', requests, veh, external)
    return [ImageResponse.from_msgpack(r) for r in raw]


# =============================================================================
# PERFIS POR AMBIENTE (Urban Air Mobility)
# =============================================================================
# Bounds em coordenadas NED do AirSim (X frente, Y direita, Z BAIXO -> -Z é UP).
# Valores baseados em layouts típicos dos 3 envs oficiais; refinar se necessário.

ENV_PROFILES = {
    # NOTA: NED -> Z é BAIXO (-Z é UP). z_max (= menos negativo) define a
    # ALTITUDE MÍNIMA do drone. Mantemos z_max bem acima do skyline do env
    # pra garantir céu como background (sem oclusão por prédio/casa/árvore).
    "city": {
        "name": "CityEnviron",
        "description": "UAM city — ego 70-100m AGL (acima de prédios baixos), drones 50-110m",
        "ego_vantage_points": [
            (0.0,    0.0,   -80.0),
            (-40.0,  30.0,  -90.0),
            (50.0,  -30.0,  -75.0),
            (-30.0, -60.0, -100.0),
            (80.0,   40.0,  -85.0),
            (-70.0,  80.0,  -70.0),
        ],
        "drone_z_bounds": (-110.0, -50.0),      # 50..110m AGL — voo UAM em cidade
        "search_radius": (10.0, 65.0),
        "typical_building_height": 60.0,
        "safe_init_altitude": -120.0,
    },
    "neighborhood": {
        "name": "AirSimNH",
        "description": "Suburb delivery — voo realista 25-70m AGL (last-mile)",
        "ego_vantage_points": [
            (0.0,    0.0,   -35.0),
            (-30.0,  20.0,  -40.0),
            (40.0,  -30.0,  -45.0),
            (0.0,    60.0,  -30.0),
            (-50.0, -50.0,  -50.0),
        ],
        "drone_z_bounds": (-70.0, -25.0),       # 25..70m AGL — last-mile UAM
        "search_radius": (8.0, 60.0),
        "typical_building_height": 18.0,
        "safe_init_altitude": -80.0,
    },
    "citypark": {
        "name": "CityPark / LandscapeMountains",
        "description": "Edge-of-city — voo acima de árvores/colinas",
        "ego_vantage_points": [
            (0.0,    0.0,   -55.0),
            (-80.0,  60.0,  -60.0),
            (50.0,  -50.0,  -50.0),
            (100.0,  0.0,   -70.0),
        ],
        "drone_z_bounds": (-100.0, -45.0),      # 45..100m AGL
        "search_radius": (10.0, 80.0),
        "typical_building_height": 35.0,
        "safe_init_altitude": -120.0,
    },
    "coastline": {
        "name": "Coastline",
        "description": "Cenário litoral — voo recreativo/inspeção 20-80m AGL",
        "ego_vantage_points": [
            (0.0,    0.0,   -30.0),
            (60.0,   40.0,  -45.0),
            (-50.0,  80.0,  -25.0),
            (120.0, -30.0,  -40.0),
            (-100.0, -60.0, -55.0),
        ],
        "drone_z_bounds": (-80.0, -20.0),       # 20..80m AGL
        "search_radius": (8.0, 65.0),
        "typical_building_height": 10.0,
        "safe_init_altitude": -90.0,
    },
}

# =============================================================================
# CONSTANTES DA CÂMERA (idênticas ao v1 — mantém compat com tracker)
# =============================================================================
IMAGE_W, IMAGE_H = 1280, 720
FOV_H_DEG = 90
FOV_V_DEG = 60
FX = IMAGE_W / (2 * np.tan(np.radians(FOV_H_DEG / 2)))
# Square pixels (AirSim convention) — FY = FX, não usa FOV_V_DEG.
# Calibrado contra API relative_pose vs bbox: FY=640 dá match perfeito (FY=623 estava off).
FY = FX
CX, CY = IMAGE_W / 2, IMAGE_H / 2

CAMERA = "front_center"
IMG_TYPE = airsim.ImageType.Scene
DETECTION_RADIUS_CM = 15000  # 150m

# Câmera front_center no settings.json:
#   offset no body do Ego: (X=0.35, Y=0, Z=-0.5)
#   pitch=-15° (olhando 15° abaixo da horizontal), roll=0, yaw=0
CAM_OFFSET_BODY = np.array([0.35, 0.0, -0.5])
CAM_PITCH_RAD = math.radians(-15.0)
# R(body→cam) = R_y(-cam_pitch). Pra cam_pitch=-15°, é R_y(+15°)
_a = -CAM_PITCH_RAD
_c, _s = math.cos(_a), math.sin(_a)
R_BODY_TO_CAM = np.array([
    [_c, 0.0, _s],
    [0.0, 1.0, 0.0],
    [-_s, 0.0, _c],
])

# Dimensões aproximadas do mesh do drone (m): largura, profundidade, altura
DRONE_EXTENT = np.array([1.0, 1.0, 0.3])

# Origens (X, Y, Z em NED) dos veículos do settings.json. simSetVehiclePose
# escreve pose RELATIVA à origem; precisamos somar origin pra ter pose global.
VEHICLE_ORIGINS = {
    'Ego': (0.0, 0.0, -5.0),
    'Drone3': (15.0, -5.0, -5.0),
    'Drone4': (15.0, 5.0, -5.0),
    'Intruder1': (-15.0, 0.0, -5.0),
}

# Quadrotor1 4× extents reais (medidos via det.box3D): X=fwd=3.01, Y=right=3.93, Z=down=2.79
# Margem 1.38× pra cobrir rotores spinning excluídos do AABB AirSim.
DRONE_BBOX_HALF = np.array([3.01, 3.93, 2.79]) * 1.38 / 2

# Segmentation: assigna ID único pra "drone" classe. AirSim mapeia ID → cor.
SEG_ID_DRONE = 200
# Cor RGB descoberta na calibração (set em runtime)
DRONE_SEG_COLOR = None

# =============================================================================
# DOMAIN RANDOMIZATION
# =============================================================================
# Mantém só níveis LEVES — fog_heavy (0.7) e rain_heavy washeavam tudo,
# drones ficavam invisíveis mas labels eram salvos = lixo pro treino.
WEATHER_PRESETS = [
    "clear", "clear", "clear", "clear",
    "fog_light",
    "rain_light",
    "dust_light",
]
# Evita horários completamente escuros (drone vira pixel preto vs sky preto)
TIME_OF_DAY_HOURS = [7, 9, 11, 13, 15, 17, 18]

# Threshold de qualidade da imagem: rejeita se contraste muito baixo (fog excessiva)
MIN_IMAGE_STD = 12.0


def apply_weather(client, weather):
    """Limpa e aplica preset de clima. Retorna a string aplicada."""
    for p in (airsim.WeatherParameter.Fog,
              airsim.WeatherParameter.Rain,
              airsim.WeatherParameter.Snow,
              airsim.WeatherParameter.Dust,
              airsim.WeatherParameter.Roadwetness,
              airsim.WeatherParameter.MapleLeaf,
              airsim.WeatherParameter.RoadLeaf):
        try:
            client.simSetWeatherParameter(p, 0.0)
        except Exception:
            pass

    presets = {
        "fog_light":  {airsim.WeatherParameter.Fog: 0.15},   # leve, drone ainda visível
        "rain_light": {airsim.WeatherParameter.Rain: 0.25,
                       airsim.WeatherParameter.Roadwetness: 0.4},
        "dust_light": {airsim.WeatherParameter.Dust: 0.2},   # leve poeira
    }
    if weather in presets:
        try:
            client.simEnableWeather(True)
        except Exception:
            pass
        for p, v in presets[weather].items():
            try:
                client.simSetWeatherParameter(p, v)
            except Exception:
                pass
    return weather


def apply_time_of_day(client, hour):
    """Define horário fixo (sun position). celestial_clock_speed=0 -> sol parado."""
    try:
        client.simSetTimeOfDay(True, f"2024-06-15 {hour:02d}:00:00",
                               celestial_clock_speed=0)
    except Exception as e:
        print(f"  warning: simSetTimeOfDay falhou ({e}) — continuando sem")
    return hour


# =============================================================================
# COLLISION-AWARE POSITIONING
# =============================================================================

def yaw_quat(yaw_rad):
    """Quaternionr para rotação somente em yaw (eixo Z).
    cosysairsim usa ordem (x_val, y_val, z_val, w_val) no construtor."""
    h = float(yaw_rad) * 0.5
    return airsim.Quaternionr(0.0, 0.0, math.sin(h), math.cos(h))


def teleport(client, vehicle_name, x, y, z, yaw_rad=0.0):
    """Teleporta drone (instantâneo). ignore_collision=True para conseguir
    atravessar prédios durante a busca; a verificação real é feita depois
    via comparação de timestamp do simGetCollisionInfo."""
    pose = airsim.Pose(
        airsim.Vector3r(float(x), float(y), float(z)),
        yaw_quat(yaw_rad),
    )
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle_name)


def collided_since(client, vehicle_name, before_ts):
    """True se novo evento de colisão ocorreu após `before_ts`."""
    info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    return info.has_collided and info.time_stamp > before_ts


def find_safe_pose(client, vehicle_name, ego_pos, ego_yaw_rad,
                   search_radius, z_bounds, fov_h_deg=FOV_H_DEG,
                   max_attempts=18, settle_s=0.18):
    """
    Acha (x,y,z) collision-free para o drone, dentro do FOV do ego.

    Estratégia:
      1) Amostra (r, bearing) em coords polares relativas ao ego.
      2) Limita bearing a ~85% do FOV horizontal -> drone aparece na imagem.
      3) z amostrado em torno do z do ego, clipado em z_bounds do ambiente.
      4) Teleporta com ignore_collision=True, espera physics tick.
      5) Compara timestamp da colisão; se nova colisão -> sobe 8m e tenta de novo.
    """
    r_min, r_max = search_radius
    z_min, z_max = z_bounds
    half_fov = math.radians(fov_h_deg / 2) * 0.85

    # Snapshot do timestamp ANTES de qualquer teleport — vamos comparar contra ele
    before_info = client.simGetCollisionInfo(vehicle_name=vehicle_name)
    before_ts = before_info.time_stamp

    for _ in range(max_attempts):
        r = random.uniform(r_min, r_max)
        bearing = ego_yaw_rad + random.uniform(-half_fov, half_fov)
        x = ego_pos[0] + r * math.cos(bearing)
        y = ego_pos[1] + r * math.sin(bearing)
        # Altitude relativa: drones aparecem em quase qq lugar vertical da imagem
        z = ego_pos[2] + random.uniform(-12.0, 12.0)
        z = max(z_min, min(z_max, z))

        # Vira o nariz do drone na direção que pode ser realista (para câmera ver)
        drone_yaw = random.uniform(-math.pi, math.pi)
        teleport(client, vehicle_name, x, y, z, drone_yaw)
        time.sleep(settle_s)

        if not collided_since(client, vehicle_name, before_ts):
            return (x, y, z, drone_yaw)

        # Colidiu — tenta MAIS ALTO (acima do skyline aproximado) na próx iteração
        before_ts = client.simGetCollisionInfo(vehicle_name=vehicle_name).time_stamp

    # Última tentativa: força acima do skyline típico do ambiente
    z_safe = max(z_min, ego_pos[2] - 40.0)  # 40m acima do ego
    bearing = ego_yaw_rad + random.uniform(-half_fov, half_fov)
    r = random.uniform(r_min, (r_min + r_max) / 2)
    x = ego_pos[0] + r * math.cos(bearing)
    y = ego_pos[1] + r * math.sin(bearing)
    teleport(client, vehicle_name, x, y, z_safe, random.uniform(-math.pi, math.pi))
    time.sleep(settle_s)
    return (x, y, z_safe, 0.0)  # melhor esforço


def ego_safe(client):
    info = client.simGetCollisionInfo(vehicle_name="Ego")
    # Não temos um before_ts robusto aqui — usa só has_collided
    return not info.has_collided


# =============================================================================
# CAPTURA & PROCESSAMENTO
# =============================================================================

def depth_to_pointcloud(depth_array, max_depth=250.0):
    """Converte depth planar para point cloud cam-frame.
    max_depth: filtro pra remover sky (saturado) mas manter terreno/objetos."""
    h, w = depth_array.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth_array > 0.1) & (depth_array < max_depth)
    z = depth_array[valid]
    u_v = u[valid]
    v_v = v[valid]
    x = (u_v - CX) * z / FX
    y = (v_v - CY) * z / FY
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def capture_frame(client):
    """Pega RGB + DepthPlanar + Segmentation do Ego em uma única chamada."""
    responses = legacy_get_images(client, [
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, True, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
    ], "Ego")
    if (not responses[0].image_data_uint8) or (not responses[1].image_data_float):
        return None, None, None
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(
        responses[0].height, responses[0].width, 3)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(
        responses[1].height, responses[1].width)
    if depth.shape != (IMAGE_H, IMAGE_W):
        depth = cv2.resize(depth, (IMAGE_W, IMAGE_H))
    seg = None
    if responses[2].image_data_uint8:
        # Mantém seg na mesma ordem RGB que a calibração viu (NÃO converte BGR).
        seg = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8).reshape(
            responses[2].height, responses[2].width, 3).copy()
    return bgr, depth, seg


def calibrate_drone_seg_color(client, drone_targets, debug_dir=None):
    """
    Descobre a cor de segmentação dos drones via DIFF (mesma do v6 que funcionou):
    1) Move ego pra (0,0,-50) e TODOS drones pra (500,500,-50) (longe)
    2) Captura seg_empty (sem drones na cena)
    3) Move primeiro drone pra (12, 0, -50)
    4) Captura seg_with
    5) Pixels diferentes = drone → cor mais frequente = cor do drone
    """
    if not drone_targets:
        return None
    from collections import Counter

    orig_poses = {"Ego": client.simGetVehiclePose("Ego")}
    for d in drone_targets:
        orig_poses[d] = client.simGetVehiclePose(d)

    cal_z = -50.0
    far = airsim.Pose(airsim.Vector3r(500, 500, cal_z), yaw_quat(0))
    client.simSetVehiclePose(airsim.Pose(
        airsim.Vector3r(0, 0, cal_z), yaw_quat(0)), True, "Ego")
    for d in drone_targets:
        client.simSetVehiclePose(far, True, d)
    # Hover pra não cair / rotacionar por gravidade
    try:
        client.hoverAsync(vehicle_name="Ego")
        for d in drone_targets:
            client.hoverAsync(vehicle_name=d)
    except Exception:
        pass
    time.sleep(1.2)

    def grab_seg():
        resp = legacy_get_images(client, [
            airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
        ], "Ego")
        return np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(
            resp[0].height, resp[0].width, 3).copy()

    try:
        seg_empty = grab_seg()
        cal = drone_targets[0]
        client.simSetVehiclePose(airsim.Pose(
            airsim.Vector3r(12, 0, cal_z), yaw_quat(0)), True, cal)
        try: client.hoverAsync(vehicle_name=cal)
        except Exception: pass
        time.sleep(1.0)
        seg_with = grab_seg()
    except Exception as e:
        print(f"  calibração falhou: {e}")
        for name, pose in orig_poses.items():
            client.simSetVehiclePose(pose, True, name)
        return None

    # Após hover, ego deve estar estável. Usa pose ATUAL pra projeção precisa.
    ep2 = client.simGetVehiclePose("Ego")
    dp2 = client.simGetVehiclePose(cal)
    ego_pos = np.array([ep2.position.x_val, ep2.position.y_val, ep2.position.z_val])
    ego_R = quat_to_rot(ep2.orientation)
    drone_world = np.array([dp2.position.x_val, dp2.position.y_val, dp2.position.z_val])
    proj = world_to_image(drone_world, ego_pos, ego_R)
    color = None
    if proj is not None:
        u, v = int(proj[0]), int(proj[1])
        # Pega janela 7x7 centrada na projeção. O drone está aqui (12m de distância,
        # ~50 pixels de largura) — então pelo menos o pixel central é cor de drone.
        # Compara cada pixel com seg_empty no MESMO local: cor que mudou = drone.
        half = 3
        y1 = max(0, v - half); y2 = min(seg_with.shape[0], v + half + 1)
        x1 = max(0, u - half); x2 = min(seg_with.shape[1], u + half + 1)
        with_roi = seg_with[y1:y2, x1:x2]
        empty_roi = seg_empty[y1:y2, x1:x2]
        # Pixels que mudaram = drone (no centro deve ser quase tudo)
        changed_mask = np.any(with_roi != empty_roi, axis=2)
        if changed_mask.sum() > 0:
            drone_pixels = with_roi[changed_mask]
            cnt = Counter([tuple(int(x) for x in p) for p in drone_pixels])
            for col, n in cnt.most_common():
                if col != (0, 0, 0):
                    color = np.array(col, dtype=np.uint8)
                    break

    if debug_dir is not None:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(debug_dir) / "calib_seg_empty.png"),
                    cv2.cvtColor(seg_empty, cv2.COLOR_RGB2BGR))
        seg_with_marked = cv2.cvtColor(seg_with, cv2.COLOR_RGB2BGR).copy()
        if proj is not None:
            cv2.circle(seg_with_marked, (int(proj[0]), int(proj[1])), 12, (0, 255, 255), 2)
        cv2.imwrite(str(Path(debug_dir) / "calib_seg_with_drone.png"), seg_with_marked)
        with open(Path(debug_dir) / "calib_color.txt", "w") as f:
            f.write(f"Drone seg color (RGB): "
                    f"{tuple(color.tolist()) if color is not None else None}\n")
            if proj is not None:
                f.write(f"Projected pixel: ({int(proj[0])}, {int(proj[1])})\n")

    for name, pose in orig_poses.items():
        client.simSetVehiclePose(pose, True, name)
    time.sleep(0.3)
    return color


def visibility_via_seg(seg_img, bbox, drone_color,
                       tol=15, min_pixels=4, frac_threshold=0.05):
    """
    True se há ≥ min_pixels OU ≥ frac_threshold% do bbox com a cor do drone.
    """
    if seg_img is None or drone_color is None:
        return None  # cair pra heurística de depth
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(IMAGE_W - 1, x2); y2 = min(IMAGE_H - 1, y2)
    area = (x2 - x1) * (y2 - y1)
    if area < 6:
        return False
    roi = seg_img[y1:y2, x1:x2]
    diff = np.abs(roi.astype(int) - drone_color.astype(int)).sum(axis=2)
    drone_pixels = int((diff <= tol).sum())
    return drone_pixels >= min_pixels or drone_pixels >= frac_threshold * area


def extract_bbox(det):
    """Retorna (xmin,ymin,xmax,ymax) ou None — lida com .x_val ou .x do box2D."""
    if not hasattr(det, "box2D") or not hasattr(det.box2D, "min"):
        return None
    m, M = det.box2D.min, det.box2D.max
    try:
        if hasattr(m, "x_val"):
            return (int(m.x_val), int(m.y_val), int(M.x_val), int(M.y_val))
        return (int(m.x), int(m.y), int(M.x), int(M.y))
    except Exception:
        return None


# =============================================================================
# PROJEÇÃO 3D → 2D (para bbox sintético quando simGetDetections falha)
# =============================================================================
def quat_to_rot(q):
    """Quaternionr -> 3x3 rotation matrix (body to world)."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def world_to_image(point_world, ego_pos, ego_R):
    """Projeta ponto NED em pixel (u,v) + depth. None se atrás da câmera."""
    rel_body = ego_R.T @ (point_world - ego_pos)
    rel_cam = R_BODY_TO_CAM @ (rel_body - CAM_OFFSET_BODY)
    depth_proj = rel_cam[0]
    if depth_proj <= 0.5:
        return None
    u = rel_cam[1] / depth_proj * FX + CX
    v = rel_cam[2] / depth_proj * FY + CY
    return u, v, depth_proj


class SyntheticDet:
    """Imita o suficiente de cosysairsim DetectionInfo: .name + .box2D.min/max."""
    class _Point:
        __slots__ = ('x_val', 'y_val')
        def __init__(self, x, y):
            self.x_val = x; self.y_val = y
    class _Box:
        __slots__ = ('min', 'max')

    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
        self.box2D = SyntheticDet._Box()
        self.box2D.min = SyntheticDet._Point(x1, y1)
        self.box2D.max = SyntheticDet._Point(x2, y2)


def find_drone_clusters_in_seg(seg_img, drone_colors, exclude_bboxes=None,
                                min_area=10, max_area=4000):
    """
    Encontra clusters de pixels com EXATAMENTE as drone_colors (descobertas
    via API bbox centers). Retorna bboxes não cobertas por exclude_bboxes.
    """
    if not drone_colors:
        return []
    flat_keys = (seg_img[:, :, 0].astype(np.int32) << 16) | \
                (seg_img[:, :, 1].astype(np.int32) << 8) | \
                 seg_img[:, :, 2].astype(np.int32)
    mask = np.isin(flat_keys, list(drone_colors)).astype(np.uint8)
    if mask.sum() == 0:
        return []
    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    bboxes = []
    for lbl_id in range(1, n_labels):
        ys, xs = np.where(labels == lbl_id)
        if len(ys) < min_area or len(ys) > max_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bboxes.append((x1, y1, x2, y2))
    if exclude_bboxes:
        kept = []
        for bb in bboxes:
            bcx = (bb[0] + bb[2]) / 2; bcy = (bb[1] + bb[3]) / 2
            covered = False
            for (ax1, ay1, ax2, ay2) in exclude_bboxes:
                if ax1 - 5 <= bcx <= ax2 + 5 and ay1 - 5 <= bcy <= ay2 + 5:
                    covered = True
                    break
            if not covered:
                kept.append(bb)
        bboxes = kept
    return bboxes


def discover_drone_colors_via_depth(seg_img, depth_img, api_detections,
                                     drone_world_poses, ego_pos):
    """
    Descobre cor seg dos drones DETERMINISTICAMENTE:
    Pra cada API detection, sabemos drone_world_pose → expected_dist do ego.
    Dentro da bbox da API, pixels com depth ≈ expected_dist SÃO do drone.
    Amostra seg color desses pixels → cor exata.
    Retorna set de cor keys (int RGB packed).
    """
    from collections import Counter
    color_counts = Counter()
    for det in api_detections:
        if not hasattr(det, 'name') or det.name not in drone_world_poses:
            continue
        bb = extract_bbox(det)
        if bb is None: continue
        x1, y1, x2, y2 = bb
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(seg_img.shape[1]-1, x2); y2 = min(seg_img.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1: continue

        drone_world = drone_world_poses[det.name]
        expected_dist = float(np.linalg.norm(drone_world - ego_pos))
        tol = max(3.0, 0.20 * expected_dist)

        depth_roi = depth_img[y1:y2, x1:x2]
        seg_roi = seg_img[y1:y2, x1:x2]
        # Mask: pixels físicos na distância do drone
        drone_mask = (depth_roi > expected_dist - tol) & (depth_roi < expected_dist + tol)
        if drone_mask.sum() < 3:
            continue  # bbox sem pixels físicos no drone (drone contra céu = depth saturada)
        drone_pixels = seg_roi[drone_mask]
        for px in drone_pixels:
            key = (int(px[0]) << 16) | (int(px[1]) << 8) | int(px[2])
            if key != 0:
                color_counts[key] += 1
    # Aceita só cores que apareceram pelo menos 3 vezes (filtra noise de borda)
    return set(k for k, n in color_counts.items() if n >= 3)


def synthesize_detection(client, drone_name, ego_pos, ego_R, depth_img,
                         seg_img=None, margin_px=10, max_dist=90.0):
    """
    Se simGetDetections perdeu este drone, tenta criar bbox por projeção.
    Confirma fisicamente que drone tá visível usando segmentação se disponível
    (drone color no pixel projetado), senão fallback pra depth.
    """
    try:
        dp = client.simGetVehiclePose(vehicle_name=drone_name).position
    except Exception:
        return None
    drone_world = np.array([dp.x_val, dp.y_val, dp.z_val])
    proj = world_to_image(drone_world, ego_pos, ego_R)
    if proj is None:
        return None
    u_c, v_c, dist = proj
    if dist > max_dist:
        return None
    if not (margin_px <= u_c < IMAGE_W - margin_px and
            margin_px <= v_c < IMAGE_H - margin_px):
        return None
    u_i, v_i = int(u_c), int(v_c)
    half = 5

    # Verificação via seg (preferida) ou depth (fallback)
    if seg_img is not None and DRONE_SEG_COLOR is not None:
        win = seg_img[max(0, v_i-half):min(IMAGE_H, v_i+half+1),
                      max(0, u_i-half):min(IMAGE_W, u_i+half+1)]
        diff = np.abs(win.astype(int) - DRONE_SEG_COLOR.astype(int)).sum(axis=2)
        if (diff <= 15).sum() == 0:
            return None  # drone não está visível no pixel projetado
    else:
        win = depth_img[max(0, v_i-half):min(IMAGE_H, v_i+half+1),
                        max(0, u_i-half):min(IMAGE_W, u_i+half+1)]
        tol = max(2.5, 0.20 * dist)
        if ((win > dist - tol) & (win < dist + tol)).sum() == 0:
            return None

    # Bbox mais justo (não sobreestima — antes o height multiplier era 1.5)
    w_px = (DRONE_EXTENT[1] / dist) * FX * 0.5
    h_px = (DRONE_EXTENT[2] / dist) * FY * 1.0
    x1 = max(0, int(u_c - w_px))
    y1 = max(0, int(v_c - h_px))
    x2 = min(IMAGE_W - 1, int(u_c + w_px))
    y2 = min(IMAGE_H - 1, int(v_c + h_px))
    if (x2 - x1) * (y2 - y1) < 9:
        return None
    return SyntheticDet(drone_name, x1, y1, x2, y2)


def visibility_status(client, det, depth, ego_pos,
                      tol_frac=0.20, tol_min=2.5,
                      occluder_threshold=0.35, min_area=9):
    """
    Retorna ('ok' | 'occluded' | 'too_small' | 'no_pose' | 'no_bbox').
    Filtro mais rigoroso que o anterior:

      - Conta pixels do bbox que mostram algo MAIS PERTO que (drone_dist - tol).
        Esses pixels são "occluders" (prédio/árvore na frente).
      - Se occluder_fraction > occluder_threshold (35% por padrão) → ocluído.
      - Pixels de céu (depth >= 100m) NÃO contam como occluder (drone contra céu = OK).
      - Pixels na profundidade do drone também NÃO contam.
    """
    bbox = extract_bbox(det)
    if bbox is None:
        return ('no_bbox', None)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(IMAGE_W - 1, x2); y2 = min(IMAGE_H - 1, y2)
    area = (x2 - x1) * (y2 - y1)
    if area < min_area:
        return ('too_small', area)

    try:
        dp = client.simGetVehiclePose(vehicle_name=det.name).position
        drone_world = np.array([dp.x_val, dp.y_val, dp.z_val])
        expected_dist = float(np.linalg.norm(drone_world - ego_pos))
    except Exception:
        return ('no_pose', None)

    roi = depth[y1:y2, x1:x2]
    tol = max(tol_min, tol_frac * expected_dist)
    # Occluder = pixel com depth válida AND significativamente mais perto que o drone
    occluder = (roi > 0.1) & (roi < expected_dist - tol)
    occluder_fraction = float(occluder.sum()) / area
    if occluder_fraction > occluder_threshold:
        return ('occluded', occluder_fraction)
    return ('ok', occluder_fraction)


def is_visible(client, det, depth, ego_pos):
    """Wrapper booleano para retro-compat."""
    return visibility_status(client, det, depth, ego_pos)[0] == 'ok'


def compute_drone_bbox_manual(drone_pos_world, drone_orient_quat,
                               ego_pos, ego_R,
                               extent=(0.85, 0.85, 0.20)):
    """Calcula bbox 2D apertada projetando os 8 corners do AABB local do mesh.

    Não usa simGetDetections (que retorna AABB incluindo collision shape gorda).
    Faz a projeção manualmente usando pose do drone + intrinsics da câmera.

    Args:
        drone_pos_world: (x, y, z) NED — posição do drone
        drone_orient_quat: tuple (w, x, y, z) — orientação
        ego_pos: (x, y, z) ego no mundo
        ego_R: 3x3 rotation matrix body→world do ego
        extent: meia-largura do mesh em metros (x_half, y_half, z_half)
                Default (0.85, 0.85, 0.20) = Quadrotor1 ~4x (1.7m × 1.7m × 0.4m)

    Returns:
        (x_min, y_min, x_max, y_max) em pixels, ou None se atrás da câmera.
    """
    ex, ey, ez = extent
    # 8 corners no frame local do drone (NED conventions)
    corners_local = np.array([
        [+ex, +ey, +ez], [-ex, +ey, +ez], [+ex, -ey, +ez], [-ex, -ey, +ez],
        [+ex, +ey, -ez], [-ex, +ey, -ez], [+ex, -ey, -ez], [-ex, -ey, -ez],
    ])
    # Rotação local→mundo via quat drone
    w, x, y, z = drone_orient_quat
    R_drone = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    pixels = []
    any_in_front = False
    for c in corners_local:
        corner_world = drone_pos_world + R_drone @ c
        proj = world_to_image(corner_world, ego_pos, ego_R)
        if proj is None:
            continue
        u, v, d = proj
        if d > 0.5:
            any_in_front = True
        pixels.append((u, v))
    if not any_in_front or len(pixels) < 2:
        return None
    us = [p[0] for p in pixels]
    vs = [p[1] for p in pixels]
    return (min(us), min(vs), max(us), max(vs))


def refine_bbox_via_depth(bbox_loose, depth, drone_dist,
                           tol_pct=0.10, tol_min=1.0, expand_pct=0.5):
    """Refina bbox AABB (loose) → bbox apertado usando depth.

    AABB do simGetDetections é projeção dos vértices do AABB 3D → fica gordo,
    não-centralizado. Depth filter: pixels com depth ≈ drone_dist são DO DRONE;
    background tem depth diferente. Bbox tight = min/max desses pixels.

    Args:
        bbox_loose: (x1, y1, x2, y2) bbox solto
        depth: array (H, W) depth planar em metros
        drone_dist: distância do drone à câmera (relative_pose magnitude)
        tol_pct: tolerância % da distância
        tol_min: tolerância mínima absoluta (m)
        expand_pct: % de expansão da bbox antes de buscar (margem)

    Returns:
        (bbox_tight, n_drone_pixels) — bbox_tight no formato (x1,y1,x2,y2).
        Se não conseguir refinar (pouco pixel), retorna bbox_loose.
    """
    H, W = depth.shape
    x1, y1, x2, y2 = [int(v) for v in bbox_loose]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return bbox_loose, 0
    mx = max(8, int((x2 - x1) * expand_pct))
    my = max(8, int((y2 - y1) * expand_pct))
    xs1 = max(0, x1 - mx); ys1 = max(0, y1 - my)
    xs2 = min(W, x2 + mx); ys2 = min(H, y2 + my)
    sub = depth[ys1:ys2, xs1:xs2]
    tol = max(tol_min, drone_dist * tol_pct)
    mask = (sub > drone_dist - tol) & (sub < drone_dist + tol)
    n = int(mask.sum())
    if n < 4:
        return bbox_loose, n
    ys, xs = np.where(mask)
    return (xs1 + int(xs.min()), ys1 + int(ys.min()),
            xs1 + int(xs.max()), ys1 + int(ys.max())), n


def build_labels(detections, depth, client=None, ego_pos=None, ego_R=None,
                 drone_world_poses=None, drone_world_quats=None,
                 stats=None, seg=None,
                 refine_bbox=False):  # default off — bbox manual já é apertado
    """
    Gera labels YOLO + 3D para um frame. Filtros:
      1) Cross-check: projeção 3D do drone deve bater com centro do bbox da API
         (rejeita se API retornou bbox stale/de pose desatualizada). Tol: 25px.
      2) Visibility: pixel no centro do bbox tem depth ≈ distância real do drone.
         Se MUITO menor (algo na frente) → ocluído.
      3) Se refine_bbox=True, refina o bbox AABB (gordo) → bbox apertado via depth
         filter (pixels com depth ≈ distância do drone).
    Usa poses do snapshot pausado (drone_world_poses) — sem race conditions.
    """
    yolo_lines = []
    labels_3d = []
    viz_boxes = []

    # Filtro: se ALGUM VisQuad_* está nas detecções, mantém só as visuais
    # (o multirotor original também é detectado pelo mesh "Quadrotor1" e duplica).
    has_visual = any(getattr(d, 'name', '').startswith("VisQuad_") for d in detections)
    if has_visual:
        detections = [d for d in detections if getattr(d, 'name', '').startswith("VisQuad_")]

    for det in detections:
        name = det.name if hasattr(det, 'name') else "drone"
        # Mapeia VisQuad_DroneX → DroneX (objeto visual sincronizado com o multirotor)
        if name.startswith("VisQuad_"):
            name = name[len("VisQuad_"):]

        # ====== BBox 2D = CENTRO do AABB AirSim + tamanho calculado por distância ======
        # O AABB do AirSim é inflado pela collision shape, MAS seu CENTRO está correto.
        # Então: usa o centro, mas substitui tamanho pelo projetado do mesh visual.
        bbox = extract_bbox(det)
        if bbox is None:
            continue
        x1a, y1a, x2a, y2a = bbox
        cx = (x1a + x2a) / 2.0
        cy = (y1a + y2a) / 2.0

        # Filtro mínimo: rejeita só se drone está atrás da câmera (fwd<0).
        # det já filtrou drones fora da imagem (bbox 2D válida).
        if drone_world_poses and name in drone_world_poses and ego_pos is not None and ego_R is not None:
            d_global = drone_world_poses[name]
            rel_world = d_global - ego_pos
            rel_body = ego_R.T @ rel_world
            rel_cam_body = rel_body - CAM_OFFSET_BODY
            ang = -CAM_PITCH_RAD
            rc, rs = math.cos(ang), math.sin(ang)
            x_fwd = rc*rel_cam_body[0] + rs*rel_cam_body[2]
            if x_fwd < 0.5:
                if stats is not None: stats['behind_cam'] = stats.get('behind_cam', 0) + 1
                continue
        api_dist_for_size = None
        try:
            rp = det.relative_pose.position
            api_dist_for_size = float(np.sqrt(rp.x_val**2 + rp.y_val**2 + rp.z_val**2))
        except Exception:
            pass
        if api_dist_for_size is not None and api_dist_for_size > 0.5:
            # Tamanho físico do drone projetado: w_px = (DRONE_W / dist) * FX
            # Medidos via simSpawnObject('Quadrotor1', scale=4x) + simGetDetections.box3D:
            #   extent X=0.48m, Y=5.45m, Z=1.49m (com hélices incluídas)
            # W = max horizontal projetado para qualquer rotação yaw = sqrt(X² + Y²) ≈ 5.5m
            # H = Z + folga p/ tilt do drone em manobras
            DRONE_PHYS_W = 5.50
            DRONE_PHYS_H = 2.00
            w_px = DRONE_PHYS_W / api_dist_for_size * FX
            h_px = DRONE_PHYS_H / api_dist_for_size * FY
            bbox = (cx - w_px / 2.0, cy - h_px / 2.0,
                    cx + w_px / 2.0, cy + h_px / 2.0)

        x1, y1, x2, y2 = bbox
        x1 = int(max(0, x1)); y1 = int(max(0, y1))
        x2 = int(min(IMAGE_W - 1, x2)); y2 = int(min(IMAGE_H - 1, y2))
        bbox = (x1, y1, x2, y2)
        area = (x2 - x1) * (y2 - y1)
        if area < 25:
            if stats is not None: stats['too_small'] = stats.get('too_small', 0) + 1
            continue

        # ====== Distância vem direto da API (relative_pose) — não depende de depth ======
        api_dist = None
        try:
            rp = det.relative_pose.position
            api_dist = float(np.sqrt(rp.x_val**2 + rp.y_val**2 + rp.z_val**2))
        except Exception:
            pass

        # ====== Filtro MINIMO: rejeita só oclusão FORTE (>=60% pixels mais próximos) ======
        if api_dist is not None:
            depth_roi = depth[y1:y2, x1:x2]
            tol = max(3.0, 0.25 * api_dist)
            occluder_mask = (depth_roi > 0.1) & (depth_roi < api_dist - tol)
            occluder_frac = float(occluder_mask.sum()) / max(1, area)
            if occluder_frac > 0.60:
                if stats is not None: stats['occluded'] = stats.get('occluded', 0) + 1
                continue
        if stats is not None: stats['ok'] = stats.get('ok', 0) + 1

        # ====== Refinamento via depth: bbox AABB (gordo) → bbox apertado ======
        if refine_bbox and api_dist is not None:
            bbox_tight, n_drone_px = refine_bbox_via_depth(bbox, depth, api_dist)
            if n_drone_px >= 4:
                bbox = bbox_tight
                if stats is not None:
                    stats['refined'] = stats.get('refined', 0) + 1

        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(IMAGE_W - 1, x_max); y_max = min(IMAGE_H - 1, y_max)
        if x_max - x_min < 3 or y_max - y_min < 3:
            continue

        # YOLO (normalizado)
        xc = (x_min + x_max) / 2 / IMAGE_W
        yc = (y_min + y_max) / 2 / IMAGE_H
        bw = (x_max - x_min) / IMAGE_W
        bh = (y_max - y_min) / IMAGE_H
        if not (0 < bw < 1 and 0 < bh < 1):
            continue
        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # 3D label MANUAL (não usa det.relative_pose / det.box3D — AirSim tem bug
        # de pitch que offseta ~5m em down e box3D tem tamanho subestimado).
        # Frame: CV camera (x=right, y=down, z=fwd) — mesmo do PC raw.
        depth_str = "?"
        if (drone_world_poses and name in drone_world_poses
                and ego_pos is not None and ego_R is not None):
            d_global = drone_world_poses[name]
            d_quat = drone_world_quats.get(name, (1.0, 0.0, 0.0, 0.0))
            # quat → yaw (drone hovering com roll=pitch=0)
            w, x, y, z = d_quat
            d_yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

            # Centro: world → ego body → cam (pitch + offset) → reorder p CV
            rel_world = d_global - ego_pos
            rel_body = ego_R.T @ rel_world
            rel_cam_body = rel_body - CAM_OFFSET_BODY
            ang = -CAM_PITCH_RAD
            rc, rs = math.cos(ang), math.sin(ang)
            c_fwd = rc*rel_cam_body[0] + rs*rel_cam_body[2]
            c_right = rel_cam_body[1]
            c_down = -rs*rel_cam_body[0] + rc*rel_cam_body[2]
            center_cv = [float(c_right), float(c_down), float(c_fwd)]
            dist_cv = float(np.sqrt(c_right*c_right + c_down*c_down + c_fwd*c_fwd))

            # 8 corners do AABB body (com yaw drone) → world → CV
            ex, ey, ez = DRONE_BBOX_HALF
            local = np.array([
                [+ex, +ey, +ez], [-ex, +ey, +ez], [-ex, -ey, +ez], [+ex, -ey, +ez],
                [+ex, +ey, -ez], [-ex, +ey, -ez], [-ex, -ey, -ez], [+ex, -ey, -ez],
            ])
            cd, sd = math.cos(d_yaw), math.sin(d_yaw)
            R_d = np.array([[cd, -sd, 0], [sd, cd, 0], [0, 0, 1]])
            corners_world = (R_d @ local.T).T + d_global
            corners_cv = []
            for cw in corners_world:
                rb = ego_R.T @ (cw - ego_pos) - CAM_OFFSET_BODY
                cf = rc*rb[0] + rs*rb[2]
                cr = rb[1]
                cd2 = -rs*rb[0] + rc*rb[2]
                corners_cv.append([float(cr), float(cd2), float(cf)])
            corners_arr = np.array(corners_cv)
            # Filtro: rejeita se algum corner está atrás da câmera (fwd <= 0)
            # — projeção fica degenerada e bbox parece "2 pontos" no PLY.
            if (corners_arr[:, 2] < 0.5).any():
                if stats is not None:
                    stats['corner_behind'] = stats.get('corner_behind', 0) + 1
                continue
            bmin_cv = corners_arr.min(axis=0).tolist()
            bmax_cv = corners_arr.max(axis=0).tolist()

            label_3d = {
                "class": "drone",
                "name": name,
                "bbox_2d": [x_min, y_min, x_max, y_max],
                "center": center_cv,
                "distance_m": dist_cv,
                "box3D_min": [float(v) for v in bmin_cv],
                "box3D_max": [float(v) for v in bmax_cv],
                "corners": [[float(v) for v in c] for c in corners_cv],
                "drone_yaw_rad": float(d_yaw),
            }
            labels_3d.append(label_3d)
            depth_str = f"{dist_cv:.1f}m"
        viz_boxes.append((x_min, y_min, x_max, y_max, depth_str))

    return yolo_lines, labels_3d, viz_boxes


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Dataset urban UAM — generaliza v1 para 3 envs")
    ap.add_argument("--env", required=True, choices=list(ENV_PROFILES.keys()),
                    help="Ambiente atualmente rodando no AirSim")
    ap.add_argument("--frames", type=int, default=1000, help="Frames-alvo COM detecções")
    ap.add_argument("--output", type=str, default=None,
                    help="Dir de saída (default: dataset_urban_<env>)")
    ap.add_argument("--ip", type=str, default="172.19.80.1", help="IP do AirSim")
    ap.add_argument("--port", type=int, default=41451)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--yaws_per_scenario", type=int, default=4,
                    help="Quantos yaw offsets do ego por placement (mais yaw = mais frames variados)")
    ap.add_argument("--scenario_change_every", type=int, default=4,
                    help="Re-posiciona ego/drones a cada N frames")
    ap.add_argument("--weather_change_every", type=int, default=40,
                    help="Muda weather/time-of-day a cada N frames")
    ap.add_argument("--visual_scale", type=float, default=4.0,
                    help="Escala do Quadrotor1 visual anexado a cada drone (default 4x). "
                         "Use 1.0 pra desativar (drone original visível).")
    ap.add_argument("--start_frame", type=int, default=0,
                    help="Offset inicial do contador de frames (pra retomar após crash). "
                         "Gera de start_frame até start_frame+frames.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    profile = ENV_PROFILES[args.env]
    output_dir = Path(args.output or f"dataset_urban_{args.env}")

    print("=" * 72)
    print(f"  URBAN DATASET GENERATOR | env={profile['name']}")
    print(f"  {profile['description']}")
    print("=" * 72)

    # --- Conexão ---
    try:
        client = airsim.MultirotorClient(ip=args.ip, port=args.port)
        client.confirmConnection()
        vehicles = client.listVehicles()
        print(f"Conectado. Veículos: {vehicles}")
    except Exception as e:
        print(f"ERRO: não consegui conectar em {args.ip}:{args.port} -> {e}")
        sys.exit(1)

    if "Ego" not in vehicles:
        print("ERRO: settings.json precisa ter um veículo 'Ego' com câmera 'front_center'.")
        sys.exit(1)

    drone_targets = [v for v in vehicles if v != "Ego"]
    if not drone_targets:
        print("ERRO: nenhum drone-alvo encontrado (preciso de Drone3/Drone4/Intruder1 etc).")
        sys.exit(1)
    print(f"Drones-alvo: {drone_targets}")

    # --- Arm + control ---
    for v in vehicles:
        try:
            client.enableApiControl(True, v)
            client.armDisarm(True, v)
        except Exception:
            pass

    # --- Spawn inicial seguro (acima do skyline do ambiente) ---
    print(f"Movendo todos os drones para altitude segura inicial "
          f"(z={profile['safe_init_altitude']}m)...")
    init_z = profile["safe_init_altitude"]
    for i, v in enumerate(vehicles):
        # Espalha horizontalmente em volta da origem
        ang = 2 * math.pi * i / len(vehicles)
        teleport(client, v, 30.0 * math.cos(ang), 30.0 * math.sin(ang), init_z)
    time.sleep(1.5)

    # --- Estrutura de diretórios (idêntica ao v1) ---
    output_dir.mkdir(exist_ok=True)
    yolo_dir = output_dir / "yolo"
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    pn_dir = output_dir / "pointnet"
    (pn_dir / "point_clouds").mkdir(parents=True, exist_ok=True)
    (pn_dir / "point_clouds_lidar").mkdir(parents=True, exist_ok=True)
    (pn_dir / "labels_3d").mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"; vis_dir.mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    # --- Drone visual maior (Quadrotor1 4x atached a cada multirotor) ---
    visualizer = None
    if args.visual_scale and args.visual_scale != 1.0:
        print(f"\n[Visual] Anexando Quadrotor1 {args.visual_scale}x aos drones...")
        visualizer = DroneVisualizer(client, drone_targets,
                                     scale=args.visual_scale, sync_hz=30)
        visualizer.attach()
        visualizer.start_sync()
        time.sleep(0.5)

    # --- Setup detection ---
    legacy_clear_detection(client, CAMERA, IMG_TYPE, "Ego")
    legacy_set_radius(client, CAMERA, IMG_TYPE, DETECTION_RADIUS_CM, "Ego")
    if visualizer is not None:
        # Detecta os Quadrotor1 spawnados (mesh = "Quadrotor1")
        legacy_add_mesh(client, CAMERA, IMG_TYPE, "Quadrotor1", "Ego")
        print("[Detection] mesh filter: Quadrotor1 (visual)")
    else:
        for d in drone_targets:
            legacy_add_mesh(client, CAMERA, IMG_TYPE, d, "Ego")

    # Filtro de oclusão: depth heurístico (provado em v4/v6).
    # Seg-based foi removido por instabilidade (diff captura noise de anti-aliasing).
    global DRONE_SEG_COLOR
    DRONE_SEG_COLOR = None
    print("Filtro de oclusão: depth-based (visibility_status)")

    # --- Loop principal ---
    target_frames = args.start_frame + args.frames
    total = 0
    with_dets = args.start_frame
    without_dets = 0
    safe_failures = 0
    filter_stats = {}  # contagem global de motivos de descarte
    current_weather = "clear"
    current_hour = 12
    apply_weather(client, current_weather)
    apply_time_of_day(client, current_hour)

    print(f"\nMeta: {target_frames} frames COM detecções (frames sem det são pulados)")
    print(f"Saída: {output_dir.absolute()}\n")

    t0 = time.time()
    iter_count = 0
    while with_dets < target_frames:
        iter_count += 1
        # Hard timeout: 2x o esperado (~0.5s/frame médio) — evita loop infinito
        if time.time() - t0 > args.frames * 2.0 + 300:
            print("\nTimeout — possivelmente o env não está produzindo detecções. Saindo.")
            break

        # ---- 1) Trocar weather/time-of-day periodicamente ----
        if iter_count % args.weather_change_every == 1:
            current_weather = random.choice(WEATHER_PRESETS)
            current_hour = random.choice(TIME_OF_DAY_HOURS)
            apply_weather(client, current_weather)
            apply_time_of_day(client, current_hour)
            print(f"  [env-randomization] weather={current_weather}  hour={current_hour:02d}h")

        # ---- 2) Reposicionar ego ----
        ego_xyz = random.choice(profile["ego_vantage_points"])
        ego_yaw_rad = random.uniform(-math.pi, math.pi)
        teleport(client, "Ego", *ego_xyz, ego_yaw_rad)
        time.sleep(0.3)
        if not ego_safe(client):
            # Sobe 20m e tenta de novo
            teleport(client, "Ego", ego_xyz[0], ego_xyz[1], ego_xyz[2] - 20, ego_yaw_rad)
            time.sleep(0.2)

        # ---- 3) Reposicionar drones de forma collision-aware ----
        placed = []
        ego_pos = np.array(ego_xyz, dtype=float)
        for d in drone_targets:
            pose = find_safe_pose(client, d, ego_pos, ego_yaw_rad,
                                  profile["search_radius"], profile["drone_z_bounds"])
            if pose is None:
                safe_failures += 1
            else:
                placed.append((d, pose))
                # hoverAsync mantém o drone na posição (evita drift por gravidade)
                try:
                    client.hoverAsync(vehicle_name=d)
                except Exception:
                    pass

        if not placed:
            continue

        # Também trava o ego
        try:
            client.hoverAsync(vehicle_name="Ego")
        except Exception:
            pass
        time.sleep(0.4)  # estabiliza física + rendering

        # ---- 4) Capturar várias variações de yaw do ego ----
        yaw_offsets = random.sample(range(-25, 26, 5), k=min(args.yaws_per_scenario, 11))
        for off_deg in yaw_offsets:
            if with_dets >= target_frames:
                break

            # RE-TELEPORTA tudo a cada frame pra eliminar drift acumulado
            # (hoverAsync não segura drones de forma confiável).
            for dn, drone_pose in placed:
                teleport(client, dn, drone_pose[0], drone_pose[1], drone_pose[2], drone_pose[3])
            new_yaw = ego_yaw_rad + math.radians(off_deg)
            teleport(client, "Ego", ego_xyz[0], ego_xyz[1], ego_xyz[2], new_yaw)
            time.sleep(0.18)

            # ====== CRITICAL: pausa cena pra image/detections/poses serem CONSISTENTES ======
            # Sem pausa, drones driftam ~1m em 0.5s entre as queries, causando bboxes
            # deslocadas dos drones renderizados.
            try:
                client.simPause(True)
            except Exception:
                pass

            try:
                bgr, depth, seg = capture_frame(client)
                if bgr is None or depth is None:
                    continue

                # Sanity check: imagem washed
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if float(gray.std()) < MIN_IMAGE_STD:
                    without_dets += 1
                    total += 1
                    filter_stats['low_contrast'] = filter_stats.get('low_contrast', 0) + 1
                    continue

                # Detections E poses na MESMA cena pausada
                detections = legacy_get_detections(client, CAMERA, IMG_TYPE, "Ego")

                # IMPORTANTE: simGetVehiclePose retorna pose RELATIVA ao origin
                # declarado em settings.json. Precisamos somar origin pra ter pose GLOBAL,
                # que é o que aparece na cena renderizada (com DroneVisualizer fixado).
                ego_pose_now = client.simGetVehiclePose(vehicle_name="Ego")
                ego_origin_arr = np.array(VEHICLE_ORIGINS.get("Ego", (0.0, 0.0, 0.0)))
                ego_pos_now = np.array([ego_pose_now.position.x_val,
                                         ego_pose_now.position.y_val,
                                         ego_pose_now.position.z_val]) + ego_origin_arr
                ego_R_now = quat_to_rot(ego_pose_now.orientation)

                # Cache pose GLOBAL de cada drone (snapshot consistente — pos + quat orient)
                drone_world_poses = {}
                drone_world_quats = {}
                for dn in drone_targets:
                    try:
                        dp = client.simGetVehiclePose(vehicle_name=dn)
                        dn_origin = np.array(VEHICLE_ORIGINS.get(dn, (0.0, 0.0, 0.0)))
                        drone_world_poses[dn] = np.array([dp.position.x_val,
                                                          dp.position.y_val,
                                                          dp.position.z_val]) + dn_origin
                        drone_world_quats[dn] = (dp.orientation.w_val,
                                                  dp.orientation.x_val,
                                                  dp.orientation.y_val,
                                                  dp.orientation.z_val)
                    except Exception:
                        pass

                # LIDAR scan (SensorLocalFrame = mesmo frame da câmera após alinhamento)
                lidar_pts = None
                try:
                    ld = client.getLidarData(lidar_name="LidarFront", vehicle_name="Ego")
                    if ld and ld.point_cloud and len(ld.point_cloud) >= 3:
                        lidar_pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
                except Exception:
                    pass
            finally:
                try:
                    client.simPause(False)
                except Exception:
                    pass
            # ====== fim do bloco pausado ======

            # Sem synthesis. Re-teleport por frame deve permitir API detectar todos.

            if not detections:
                without_dets += 1
                total += 1
                continue

            yolo_lines, labels_3d, viz_boxes = build_labels(
                detections, depth, client=client,
                ego_pos=ego_pos_now, ego_R=ego_R_now,
                drone_world_poses=drone_world_poses,
                drone_world_quats=drone_world_quats,
                stats=filter_stats, seg=seg)
            if not yolo_lines:
                without_dets += 1
                total += 1
                continue

            # === Salva ===
            frame_name = f"frame_{with_dets:06d}"
            is_train = with_dets % 5 != 0
            split = "train" if is_train else "val"

            cv2.imwrite(str(yolo_dir / "images" / split / f"{frame_name}.jpg"), bgr)
            with open(yolo_dir / "labels" / split / f"{frame_name}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

            pc = depth_to_pointcloud(depth)
            np.save(pn_dir / "point_clouds" / f"{frame_name}.npy", pc)
            # LIDAR PC (mesmo frame da câmera com novo alinhamento)
            if lidar_pts is not None and len(lidar_pts) > 0:
                np.save(pn_dir / "point_clouds_lidar" / f"{frame_name}.npy", lidar_pts)

            # Filtro: cada bbox precisa ter >= 5 pontos do PC dentro do AABB
            # (rejeita bboxes degeneradas que aparecem como "2 pts" no PLY).
            if labels_3d and len(pc) > 0:
                filtered = []
                for l in labels_3d:
                    bmin = np.array(l['box3D_min']); bmax = np.array(l['box3D_max'])
                    inside = ((pc[:, 0] >= bmin[0]) & (pc[:, 0] <= bmax[0]) &
                              (pc[:, 1] >= bmin[1]) & (pc[:, 1] <= bmax[1]) &
                              (pc[:, 2] >= bmin[2]) & (pc[:, 2] <= bmax[2]))
                    n_in = int(inside.sum())
                    if n_in >= 5:
                        l['n_pts_inside'] = n_in
                        filtered.append(l)
                    else:
                        filter_stats['pts_in_bbox_lt5'] = filter_stats.get('pts_in_bbox_lt5', 0) + 1
                labels_3d = filtered

            if labels_3d:
                with open(pn_dir / "labels_3d" / f"{frame_name}.json", "w") as f:
                    json.dump(labels_3d, f, indent=2)

            # Viz a cada 50 frames
            if with_dets % 50 == 0:
                viz = bgr.copy()
                for (x1, y1, x2, y2, ds) in viz_boxes:
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(viz, f"D:{ds}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(viz, f"env={args.env}  w={current_weather}  h={current_hour:02d}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(str(vis_dir / f"{frame_name}.jpg"), viz)

            # Metadata POR FRAME (poses GLOBAIS reais, snapshot consistente)
            meta = {
                "frame_id": int(with_dets),
                "env": args.env,
                "ego_xyz_vehicle": list(map(float, ego_xyz)),  # intent vehicle-pose
                "ego_xyz_global": [float(v) for v in ego_pos_now.tolist()],
                "ego_yaw_rad": float(new_yaw),
                "weather": current_weather,
                "hour": int(current_hour),
                "split": split,
                "n_detections": len(detections),
                "n_labels": len(yolo_lines),
                "drone_positions_vehicle": [
                    {"name": n, "xyz_yaw": list(map(float, p))} for n, p in placed
                ],
                "drone_positions_global": [
                    {"name": n,
                     "xyz": [float(v) for v in drone_world_poses.get(n, [0,0,0])],
                     "quat_wxyz": list(drone_world_quats.get(n, (1,0,0,0)))}
                    for n in drone_world_poses
                ],
            }
            with open(output_dir / "metadata" / f"{frame_name}.json", "w") as f:
                json.dump(meta, f, indent=2)

            with_dets += 1
            total += 1

            if with_dets % 50 == 0:
                rate = with_dets / max(1, total) * 100
                elapsed = time.time() - t0
                fps = with_dets / max(1, elapsed)
                print(f"  [{with_dets:5d}/{target_frames}]  total={total}  "
                      f"hit_rate={rate:5.1f}%  fps={fps:.2f}  "
                      f"safe_failures={safe_failures}")

    # --- dataset.yaml para YOLO ---
    dataset_yaml = (
        f"path: {yolo_dir.absolute()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        "names: ['drone']\n"
    )
    with open(yolo_dir / "dataset.yaml", "w") as f:
        f.write(dataset_yaml)

    n_train = len(list((yolo_dir / "labels" / "train").glob("*.txt")))
    n_val = len(list((yolo_dir / "labels" / "val").glob("*.txt")))

    if visualizer is not None:
        print("\n[Visual] Desanexando Quadrotor1 visuais...")
        visualizer.stop_sync()
        visualizer.detach()

    print("\n" + "=" * 72)
    print(f"  DONE  ({profile['name']})")
    print("=" * 72)
    print(f"  Total iterações:     {total}")
    print(f"  Frames com labels:   {with_dets}  (train={n_train}  val={n_val})")
    print(f"  Frames sem detect.:  {without_dets}")
    print(f"  Falhas de safe-pos:  {safe_failures}")
    if filter_stats:
        print(f"  Filtro de oclusão:")
        for k, v in sorted(filter_stats.items(), key=lambda kv: -kv[1]):
            print(f"    {k:>12}: {v}")
    print(f"  Saída:               {output_dir.absolute()}")
    print()
    print("  Para mergear com outros envs depois:")
    print(f"    rsync -a {output_dir}/yolo/  dataset_urban_merged/yolo/")
    print(f"    rsync -a {output_dir}/pointnet/  dataset_urban_merged/pointnet/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
