# 3D Drone Detection System in Simulated UAM Environments

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** This documents **Phase 1** of the project. Several scripts it
> presents as current have since been superseded and now live under `archive/`
> (the plain-PointNet training, the early dataset builders, the painted
> detectors). It is kept because it is one of the few records of the AirSim
> sensor configuration — Section 4.1 is the source of `configs/settings.json` —
> and of the early detector work whose numbers the dissertation reports.


**Documentação Técnica Completa**
**Data:** 2026-05-20
**Versão:** 1.0

---

## Contents

1. [Resumo Executivo](#1-resumo-executivo)
2. [Contexto e Objetivos](#2-contexto-e-objetivos)
3. [Arquitetura do Sistema](#3-arquitetura-do-sistema)
4. [Pipeline de Coleta de Dados](#4-pipeline-de-coleta-de-dados)
5. [Detecção 2D: YOLO11s](#5-detecção-2d-yolo11s)
6. [Detecção 3D: Abordagens com PointNet](#6-detecção-3d-abordagens-com-pointnet)
7. [Experimentos e Resultados Comparativos](#7-experimentos-e-resultados-comparativos)
8. [Análise Técnica e Insights](#8-análise-técnica-e-insights)
9. [Limitações Conhecidas](#9-limitações-conhecidas)
10. [Estrutura de Arquivos e Scripts](#10-estrutura-de-arquivos-e-scripts)
11. [Referências](#11-referências)

---

## 1. Resumo Executivo

Este trabalho desenvolveu um sistema completo de detecção e localização 3D de drones (Urban Air Mobility - UAM) usando o simulador AirSim com múltiplos sensores (RGB camera, depth camera, LIDAR). O sistema combina:

- **YOLO11s** treinado para detecção 2D de drones com **mAP@0.5 = 0.73**
- **PointNet variants** para localização 3D, incluindo abordagens inovadoras:
  - **Frustum-PointNet** tradicional (baseline)
  - **PointPainting-style** com canal de probabilidade derivado do YOLO
  - **PointNet++** com Focal Loss em dataset balanceado

**Resultado principal**: Pipeline end-to-end onde o PointNet++ Painted atinge MSE de centro 3D = 0.003 (1200× melhor que baseline) com F1 = 0.68 em classificação binária drone/não-drone.

### Datasets coletados

| Ambiente | Frames | Modalidades | Tamanho |
|---|---|---|---|
| AirSimNH (suburbano) | 1000 | RGB + Depth (1280×720) + LIDAR (64ch) + 3D labels | 7.0 GB |
| CityEnviron (cidade) | 1000 | idem | 7.7 GB |
| Coastline (litoral) | 1000 | idem | 7.4 GB |
| **Total** | **3000** | - | ~22 GB |

---

## 2. Contexto e Objetivos

### 2.1 Urban Air Mobility (UAM)

UAM refere-se à operação de veículos aéreos (drones, eVTOL) em ambientes urbanos, tipicamente em altitudes entre 30-120m AGL (Above Ground Level), conforme regulações de espaço aéreo Classe G. Aplicações incluem:

- Entrega de última milha (last-mile delivery)
- Inspeção de infraestrutura urbana
- Sky taxi (eVTOL)
- Surveillance e monitoramento

### 2.2 Problema

Sistemas autônomos UAM precisam **detectar e localizar drones próximos no espaço 3D** para:
- **Sense-and-avoid** (evitar colisão com tráfego aéreo)
- **Tracking** de múltiplos targets
- **Geometric reasoning** para planejamento de trajetória

Os desafios técnicos centrais:
1. Drones são objetos pequenos (~1m wingspan) em distâncias de 20-100m → poucos pixels no sensor
2. Background dinâmico (prédios, vegetação, céu)
3. Câmera monocular não fornece profundidade direta
4. LIDAR esparso em alturas elevadas (sem return em céu)

### 2.3 Objetivos

1. **Coletar dataset multi-modal** sintético de drones em 3 cenários UAM diversos
2. **Treinar detector 2D** (YOLO) como front-end
3. **Treinar detector 3D** (PointNet variants) para localização 3D precisa
4. **Investigar abordagens fusion** (RGB+depth+lidar) e técnicas SOTA

---

## 3. System architecture

### 3.1 Overview

```
                    ┌─────────────────────────────────┐
                    │ AirSim Simulator (Windows)      │
                    │  - Multi-rotor physics          │
                    │  - 3D environments              │
                    │  - Sensor simulation            │
                    └────────────┬────────────────────┘
                                 │ TCP/RPC (msgpack)
                                 │ <AIRSIM_HOST>:41451
                    ┌────────────▼────────────────────┐
                    │ Data Collection (WSL/Linux)     │
                    │  generate_dataset_urban.py      │
                    └────────────┬────────────────────┘
                                 │
                ┌────────────────┼──────────────────────┐
                │                │                      │
        ┌───────▼──────┐  ┌─────▼──────┐  ┌────────────▼───────┐
        │ RGB Camera   │  │ Depth+Seg  │  │ LIDAR (64 channels)│
        │ 1280×720,    │  │ 1280×720,  │  │ 1M pts/s, 150m     │
        │ FOV 90°      │  │ planar     │  │ Pitch=-15° aligned │
        └───────┬──────┘  └─────┬──────┘  └────────────┬───────┘
                │                │                      │
                └────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │ Processing Pipeline             │
                    │  - Frustum extraction           │
                    │  - PointPainting projection     │
                    │  - PC normalization             │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┼────────────────────┐
              │                  │                    │
        ┌─────▼─────┐    ┌──────▼──────┐    ┌────────▼─────┐
        │  YOLO11s  │    │  PointNet   │    │ PointNet++   │
        │ (2D det)  │    │ (frustum)   │    │ Painted v4   │
        │ mAP 0.73  │    │ acc 0.967   │    │ F1 0.68      │
        └───────────┘    └─────────────┘    └──────────────┘
```

### 3.2 Technology stack

| Componente | Versão | Justificativa |
|---|---|---|
| AirSim (binários) | Microsoft v1.8.1 | Pré-compilado, estável, multi-environment |
| cosysairsim (Python client) | 3.3.0 | Suporte API moderno, compatibilidade com servidores antigos via shims |
| Python | 3.10 | Compat com PyTorch + ultralytics |
| PyTorch | 2.9.1+cu128 | Treinamento GPU |
| Ultralytics | YOLO11s | Estado-da-arte 2024, mesma arquitetura usada em projetos paralelos do laboratório |
| OpenCV | 4.13 | Image processing |
| NumPy/SciPy | latest | PC manipulation |

### 3.3 Data flow

```
Frame N:
  1. simSetVehiclePose(ego_xyz, yaw)         # teleport ego
  2. simSetVehiclePose(drone_i, ...)          # teleport drones
  3. simPause(True)                           # freeze scene
     a. simGetImages([RGB, DepthPlanar, Seg]) # captura sync
     b. simGetDetections()                    # bbox API
     c. simGetVehiclePose(all)                # poses
     d. getLidarData()                        # LIDAR scan
  4. simPause(False)
  5. depth_to_pointcloud(depth)              # PC em CV frame
  6. Salvar: .jpg, .npy, .json (labels 2D + 3D)
```

A **pausa de simulação** (`simPause`) é crítica: sem ela, drones movem-se ~1m em 0.5s entre queries, causando misalignment entre imagem renderizada e poses retornadas.

---

## 4. Data-collection pipeline

### 4.1 Setup AirSim

**Hardware**: AirSim roda no Windows; scripts Python no WSL/Linux. Comunicação via TCP (IP <AIRSIM_HOST> = WSL gateway).

**Configuração** (`settings.json`):

```json
{
  "SettingsVersion": 2.0,
  "SimMode": "Multirotor",
  "LocalHostIp": "0.0.0.0",
  "ApiServerPort": 41451,
  "SegmentationSettings": {
    "InitMethod": "CommonObjectsRandomIDs"
  },
  "Vehicles": {
    "Ego": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": -5,
      "Cameras": {
        "front_center": {
          "X": 0.35, "Y": 0.0, "Z": -0.5,
          "Pitch": -15, "Roll": 0, "Yaw": 0,
          "CaptureSettings": [
            { "ImageType": 0, "Width": 1280, "Height": 720, "FOV_Degrees": 90 },
            { "ImageType": 1, "Width": 1280, "Height": 720, "FOV_Degrees": 90 },
            { "ImageType": 2, "Width": 1280, "Height": 720, "FOV_Degrees": 90 },
            { "ImageType": 5, "Width": 1280, "Height": 720, "FOV_Degrees": 90 }
          ]
        }
      },
      "Sensors": {
        "LidarFront": {
          "SensorType": 6,
          "NumberOfChannels": 64,
          "PointsPerSecond": 1000000,
          "RotationsPerSecond": 20,
          "X": 0.35, "Y": 0.0, "Z": -0.5,
          "Pitch": -15,
          "VerticalFOVUpper": 30, "VerticalFOVLower": -30,
          "HorizontalFOVStart": -45, "HorizontalFOVEnd": 45,
          "Range": 150,
          "DataFrame": "SensorLocalFrame"
        }
      }
    },
    "Drone3": { "X": 15, "Y": -5, "Z": -5 },
    "Drone4": { "X": 15, "Y": 5, "Z": -5 },
    "Intruder1": { "X": -15, "Y": 0, "Z": -5, "Yaw": 180 }
  }
}
```

**Decisão técnica**: alinhar LIDAR à câmera (mesma posição body offset, mesmo pitch=-15°, FOV horizontal/vertical igual à câmera). Resultado: LIDAR e câmera compartilham mesmo frame, simplificando projeção 3D→2D.

### 4.2 Camera intrinsics

```
IMAGE_W = 1280, IMAGE_H = 720
FOV_H = 90°
FX = 1280 / (2 * tan(45°)) = 640.0
FY = FX = 640.0  (square pixels - convenção AirSim)
CX = 640.0, CY = 360.0
```

**Nota técnica**: Inicialmente assumi FY = IMAGE_H / (2·tan(FOV_V/2)) com FOV_V = 60°, dando FY = 623. Calibrei empiricamente contra `relative_pose` da API e descobri que AirSim usa **square pixels** (FY = FX = 640). Pequena correção mas essencial para alinhamento PC ↔ bbox 3D.

### 4.3 Compatibility shims (cosys-airsim ↔ Microsoft AirSim 1.8)

O cliente Python `cosysairsim 3.3.0` foi projetado para o fork Cosys-AirSim, mas operamos com binários Microsoft AirSim 1.8.x. As assinaturas de várias APIs RPC divergem:

| API | cosys-airsim 3.3 | Microsoft AirSim 1.8 |
|---|---|---|
| `simClearDetectionMeshNames` | `(cam, type, vehicle, annotation_name)` | `(cam, type, vehicle, external: bool)` |
| `simGetDetections` | idem | idem |
| `simGetImages` | `(requests, vehicle)` | `(requests, vehicle, external: bool)` |
| `simAddDetectionFilterMeshName` | `(cam, type, mesh, vehicle, annotation_name)` | `(cam, type, mesh, vehicle, external: bool)` |

Causava `RPCError: bad cast` ou contagens erradas. **Solução**: shims que chamam RPC diretamente com assinatura legada:

```python
def legacy_get_images(client, requests, veh="", external=False):
    raw = client.client.call('simGetImages', requests, veh, external)
    return [ImageResponse.from_msgpack(r) for r in raw]

def legacy_get_detections(client, cam, img_type, veh="", external=False):
    raw = client.client.call('simGetDetections', cam, img_type, veh, external)
    return [DetectionInfo.from_msgpack(r) for r in raw]
```

(Ver `generate_dataset_urban.py:39-66` para implementação completa)

### 4.4 Ambientes UAM

Selecionei 3 ambientes Microsoft AirSim que cobrem o espectro UAM:

#### 4.4.1 AirSimNH (suburbano)
- **Cenário**: bairro suburbano com casas, ruas, árvores
- **Use case real**: entrega de última milha
- **Altitude ego**: 30-50m AGL
- **Altitude drones**: 25-70m AGL

#### 4.4.2 CityEnviron (cidade densa)
- **Cenário**: downtown com arranha-céus, ruas, prédios
- **Use case real**: corredores aéreos UAM entre edificações
- **Altitude ego**: 70-100m AGL
- **Altitude drones**: 50-110m AGL

#### 4.4.3 Coastline (litoral)
- **Cenário**: terreno costeiro com vegetação tropical
- **Use case real**: inspeção/recreativo
- **Altitude ego**: 25-55m AGL
- **Altitude drones**: 20-80m AGL

Ver `ENV_PROFILES` em `generate_dataset_urban.py:88-152` para perfis completos.

### 4.5 Placement configuration

#### Ego vantage points
Posições pré-definidas estratégicas (centro, cantos, alturas variadas) escolhidas para diversificar perspectivas:

```python
"city": {
    "ego_vantage_points": [
        (0.0,    0.0,   -80.0),
        (-40.0,  30.0,  -90.0),
        (50.0,  -30.0,  -75.0),
        (-30.0, -60.0, -100.0),
        (80.0,   40.0,  -85.0),
        (-70.0,  80.0,  -70.0),
    ],
    ...
}
```

#### Drone placement (collision-aware)
Função `find_safe_pose()` em `generate_dataset_urban.py:206-237`:

1. **Sample radial polar**: distância `r ∈ [r_min, r_max]`, bearing `θ ∈ [-FOV/2 · 0.85, +FOV/2 · 0.85]` em torno do yaw do ego (vies para FOV)
2. **Altitude**: `z ∈ [ego_z - 12, ego_z + 12]`, clipado em `drone_z_bounds`
3. **Teleporte** via `simSetVehiclePose(ignore_collision=True)`
4. **Validação**: `simGetCollisionInfo()` — compara timestamp para detectar nova colisão
5. **Retry**: se colidiu, sobe 8m e tenta novamente
6. **Fallback**: força altitude alta (ego_z - 40m) após 18 tentativas

### 4.6 Resolving race conditions

**Problema descoberto**: entre `simGetImages()` e `simGetVehiclePose()`, drones movem-se devido a gravidade. Medição empírica: drift de 1.18m em 0.5s sem pause.

**Solução**: `simPause(True)` antes do bloco de queries, `simPause(False)` depois:

```python
client.simPause(True)
try:
    bgr, depth, seg = capture_frame(client)
    detections = legacy_get_detections(client, ...)
    ego_pose = client.simGetVehiclePose("Ego")
    for d in drone_targets:
        drone_world_poses[d] = client.simGetVehiclePose(d)
    lidar_pts = client.getLidarData("LidarFront", "Ego").point_cloud
finally:
    client.simPause(False)
```

(Ver `generate_dataset_urban.py:997-1040`)

### 4.7 Per-frame re-teleport to eliminate accumulated drift

Mesmo com `hoverAsync`, drones derivam ~3.7m em 4s. Para múltiplos yaw frames por scenario (default 4 frames × 0.18s settle = 0.7s), drift pode acumular significativamente.

**Solução**: re-teleportar drones e ego para posições planejadas no início de cada yaw frame:

```python
for off_deg in yaw_offsets:
    for dn, drone_pose in placed:
        teleport(client, dn, *drone_pose)
    new_yaw = ego_yaw_rad + math.radians(off_deg)
    teleport(client, "Ego", *ego_xyz, new_yaw)
    time.sleep(0.18)
    # ... capture
```

### 4.8 Depth-based occlusion filter

Após teleport, drones podem ficar atrás de prédios mesmo com `simGetDetections` retornando bbox (API usa frustum check, não occlusion check). Implementei filtro de visibilidade em `visibility_status()`:

```python
def visibility_status(client, det, depth, ego_pos, ...):
    bbox = extract_bbox(det)
    drone_pose = client.simGetVehiclePose(det.name)
    expected_dist = ||drone_world - ego_pos||

    roi = depth[bbox]
    occluder_mask = (roi > 0.1) & (roi < expected_dist - tol)
    occluder_frac = occluder_mask.sum() / area

    if occluder_frac > 0.35:
        return 'occluded'
    return 'ok'
```

Princípio: se mais de 35% dos pixels da bbox mostram depth significativamente menor que distância esperada, há objeto à frente → drone ocluído.

### 4.9 Handling API misses (detection API ≠ render)

**Observação experimental**: `simGetDetections` mete ~36% dos drones que estão fisicamente em FOV (medido em teste com 15 cenários). API filtragem interna é mais conservativa que renderização real.

**Estratégias tentadas**:
1. **Synthesis via projeção 3D** (rejeitada): tentei projetar `simGetVehiclePose(drone)` para image space, mas convenção de coordenadas AirSim para `relative_pose` não corresponde à pinhole padrão
2. **Synthesis via seg cluster** (rejeitada): identificar drones via segmentação por cor — gerou muitos false positives em cenas com cores rara similares
3. **Aceitar limitação** (escolhida): trabalhar com o que a API retorna. Recall do sistema fica limitado a ~70% do recall máximo possível

### 4.10 Statistics of the collected dataset

#### General parameters
- **Frames por env**: 1000 (800 train, 200 val)
- **Drones por frame**: 1-3 (média ~2.0)
- **Distância drone-ego**: 8-65m (média 35m)
- **Yaw offsets per scenario**: 4 (drone fica estático, ego rotaciona)
- **Weather presets**: clear (57%), fog_light Fog=0.15 (14%), rain_light Rain=0.25 (14%), dust_light Dust=0.2 (15%)
- **Time of day**: 07h, 09h, 11h, 13h, 15h, 17h, 18h (uniforme)

#### Point-cloud density (after the 250 m filter)
- **Depth-derived PC**: 100k-330k pontos/frame (média 200k)
- **LIDAR PC**: 9k-12k pontos/frame em NH/City, mais esparso em Coast (cenas abertas)
- **Frustums (após crop por bbox 2D)**: 50-4000 pontos/frustum (média 1200)

#### Filters applied during collection
- `depth > 0.1m` e `depth < 250m` (remove sky e horizonte distante)
- `area_bbox > 25 pixels`
- Image contrast `std(gray) > 12` (rejeita frames washed pela fog)

---

## 5. 2D detection: YOLO11s

### 5.1 Arquitetura

**YOLO11s** (scale=s): YOLOv11 small. ~10M parâmetros. Escolhido por:
1. Estado-da-arte 2024 (Ultralytics)
2. Compatibilidade com tracker do laboratório (`<workspace>/ubaswarm_ws/src/tracker/`)
3. Trade-off velocidade/precisão adequado para inferência em tempo real

### 5.2 Dataset YOLO

Formato YOLO padrão (txt per image):
```
0 0.5234 0.4012 0.0234 0.0156   # class_id cx cy w h (normalizado)
```

Estrutura merged:
```
dataset_urban_merged/yolo/
├── images/
│   ├── train/  (2400 .jpg)
│   └── val/    (600 .jpg)
├── labels/
│   ├── train/  (2400 .txt)
│   └── val/    (600 .txt)
└── dataset.yaml
```

`dataset.yaml`:
```yaml
path: <DATA_ROOT>/dataset_urban_merged/yolo
train: images/train
val: images/val
nc: 1
names: ['drone']
```

### 5.3 Training hyperparameters

```python
model.train(
    data='dataset_urban_merged/yolo/dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='0',  # GPU 0
    workers=4,
    patience=30,
    name='drone_urban_v1',
)
```

Otimizador padrão Ultralytics (SGD com Nesterov momentum, weight decay, cosine LR scheduler).

### 5.4 Resultados YOLO11s

| Métrica | Valor |
|---|---|
| **mAP@0.5** | 0.731 |
| mAP@0.5:0.95 | 0.403 |
| Precision | 0.91 |
| Recall | 0.64 |
| Inference time | ~5ms/frame (RTX-class) |

**Análise**:
- **Precision alta** (91%): quando detecta, está certo. Poucos false positives.
- **Recall moderado** (64%): perde ~36% dos drones reais. Bottleneck do sistema.
- **mAP@0.5:0.95 baixo** (40%): bbox tight em drones tiny é difícil. Pixel-level accuracy limitada.

---

## 6. 3D detection: PointNet approaches

### 6.1 Frustum-PointNet baseline (Experimento 1)

#### 6.1.1 Conceito
Recortar pontos do PC que projetam dentro da bbox 2D (frustum view-cone). PointNet processa apenas esses pontos.

#### 6.1.2 Dataset
Construído por `build_pointnet_dataset.py`:
- **Positivos**: frustums extraídos pelas bboxes GT (matched via IoU)
- **Negativos**: random crops onde não há drone

| Origem | POS | NEG | Total |
|---|---|---|---|
| NH | 2015 | 1000 | 3015 |
| City | 2043 | 1000 | 3043 |
| Coast | 2078 | 1000 | 3078 |
| **Total** | **6136** (67%) | **3000** (33%) | **9136** |

Cada sample:
- `point_cloud.npy`: (N=512, 3) — XYZ normalizado para esfera unitária
- `label.json`: `is_drone`, `center_rel_normalized`, `size_normalized`, `distance_m`

#### 6.1.3 Arquitetura

```python
class PointNetDetector(nn.Module):
    def __init__(self):
        # Shared MLPs (Conv1D point-wise)
        self.mlp1 = Sequential(
            Conv1d(3, 64, 1), BN, ReLU,
            Conv1d(64, 64, 1), BN, ReLU,
        )
        self.mlp2 = Sequential(
            Conv1d(64, 128, 1), BN, ReLU,
            Conv1d(128, 256, 1), BN, ReLU,
            Conv1d(256, 512, 1), BN, ReLU,
        )
        # Global max pool + FC
        self.fc = Sequential(
            Linear(512, 256), BN, ReLU, Dropout(0.3),
            Linear(256, 128), BN, ReLU,
        )
        self.cls_head = Linear(128, 1)
        self.center_head = Linear(128, 3)
        self.size_head = Linear(128, 3)
```

Parâmetros: ~0.35M.

#### 6.1.4 Loss

```
L = w_cls · BCE(cls_logit, is_drone)
  + w_center · SmoothL1(center_pred, center_gt) · 1[is_drone=1]
  + w_size · SmoothL1(size_pred, size_gt) · 1[is_drone=1]

w_cls = 1.0, w_center = 2.0, w_size = 1.0
```

#### 6.1.5 Resultados

| Métrica | Valor |
|---|---|
| **Val accuracy** | 0.967 |
| Train accuracy | 0.97 |
| Center MSE (normalized) | 3.6 |
| Size MSE (normalized) | 0.23 |

**Análise**: Classificação excelente (96.7%) porque a tarefa "frustum tem drone?" é fácil quando comparamos com random crops. **Mas regression ruim** (MSE 3.6 em espaço normalizado [-1, 1]).

### 6.2 PointPainting Approach (Experimento 2)

#### 6.2.1 Motivation

Vora et al. (CVPR 2020) — "PointPainting: Sequential Fusion for 3D Object Detection" — propõem aumentar cada ponto LIDAR com scores de segmentação semântica da rede 2D. **Adaptação proposta**: usar a confidence do bbox YOLO como canal extra.

#### 6.2.2 Pipeline

```
1. YOLO inferência → list[(bbox_2d, confidence)] por frame
2. Para cada ponto do PC:
   a. Projeta para image space: (u, v) = (X·FX/Z + CX, Y·FY/Z + CY)
   b. Se (u, v) ∈ bbox YOLO: prob = confidence
   c. Senão: prob = 0
3. Input PointNet: (N, 4) = [xyz, prob]
4. Output: cls + center + size (mesma cabeça)
```

#### 6.2.3 Per-bbox processing

Multi-instance handling: cada bbox YOLO vira um sample independente. Para 3 bboxes em uma imagem, 3 samples diferentes do PointNet (cada com prob channel específico de uma bbox).

#### 6.2.4 Painted dataset (initial 88/12 version)

| Tipo | Quantidade | % |
|---|---|---|
| True Positives (YOLO det matched GT IoU≥0.3) | 3883 | 88.0% |
| False Positives YOLO (sem GT match) | 299 | 6.8% |
| Negatives (frames sem detecção YOLO) | 229 | 5.2% |
| **Total** | **4411** | - |

Severamente desbalanceado para "drone vs não-drone" (88% positives).

#### 6.2.5 Arquitetura modificada

```python
class PointNetPainted(nn.Module):
    def __init__(self, in_channels=4):
        # Diferença: Conv1d(in_channels, 64) em vez de Conv1d(3, 64)
        self.mlp1 = Sequential(
            Conv1d(in_channels, 64, 1), BN, ReLU,
            Conv1d(64, 64, 1), BN, ReLU,
        )
        self.mlp2 = Sequential(
            Conv1d(64, 128, 1), BN, ReLU,
            Conv1d(128, 256, 1), BN, ReLU,
            Conv1d(256, 1024, 1), BN, ReLU,  # maior que baseline
        )
        # FC mais profunda
        self.fc = Sequential(
            Linear(1024, 512), BN, ReLU, Dropout(0.3),
            Linear(512, 256), BN, ReLU, Dropout(0.2),
            Linear(256, 128), BN, ReLU,
        )
```

Parâmetros: ~1.0M.

#### 6.2.6 Resultados Painted v1 (sampler + pos_weight)

| Métrica | Valor |
|---|---|
| Val accuracy | 0.499 |
| Val precision | 0.99 |
| Val recall | 0.35 |
| Center MSE | **0.003** |

**Center MSE de 0.003 é 1200× melhor que baseline (3.6)** — a inovação PointPainting **funciona pra regression**.

**Mas classification ruim**: identificamos bug do duplo balanceamento (WeightedRandomSampler + BCE pos_weight ambos tentando corrigir imbalance).

#### 6.2.7 Painted v2 results (double-balance fix, sampler only)

| Métrica | Valor |
|---|---|
| Val accuracy | 0.651 |
| Val precision | 0.96 |
| Val recall | 0.62 |
| Center MSE | 0.003 |

Melhora classificação mas ainda longe do baseline. Análise: o dataset 88/12 leva o modelo a soluções degeneradas.

### 6.3 PointNet++ with focal loss (experiment A)

#### 6.3.1 Conceito

PointNet++ (Qi et al. 2017) introduz **set abstraction (SA)** hierárquica: FPS sampling + ball query + MLP. Captura features locais (geometria) ao invés de apenas features globais (PointNet).

**Focal Loss** (Lin et al. 2017): `FL(p_t) = -α_t(1-p_t)^γ log(p_t)`. Down-weights easy examples, focus em hard examples.

#### 6.3.2 From-scratch implementation (plain PyTorch)

Implementei todas as ops sem dependências externas (`pointnet2_ops`, `pytorch3d` não instaladas):

```python
def farthest_point_sample(xyz, npoint):
    """O(N²) FPS — viable for N=4096 com batch <= 16"""
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,))
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query: pra cada new_xyz, idx dos nsample pts no raio"""
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def forward(self, xyz, points):
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)  # normalize
        grouped_points = torch.cat([grouped_xyz, index_points(points, idx)], dim=-1)
        # Apply MLP point-wise then max pool
        new_points = max_pool(MLP(grouped_points))
        return new_xyz, new_points
```

#### 6.3.3 Arquitetura PointNet++ Painted

```
Input: (B, N=4096, 4) = xyz + prob

SA1: 4096 → 512 points, radius=0.2, nsample=32
     MLP: [64, 64, 128]
SA2: 512 → 128 points, radius=0.4, nsample=64
     MLP: [128, 128, 256]
SA3 (global): all → 1 point
     MLP: [256, 512, 1024]

FC: [1024 → 512 → 256 → 128]
Heads: cls(1), center(3), size(3)
```

Parâmetros: 1.5M.

#### 6.3.4 Hyperparameter iterations

| Versão | Sampler | Focal α | Focal γ | Dataset | Resultado |
|---|---|---|---|---|---|
| v1 | 50/50 | 0.25 | 2.0 | 88/12 | F1=0.45, rec=0.32 (bias negativo) |
| v2 | nenhum | 0.75 | 2.0 | 88/12 | F1=0.93 trivial (rec=1.0, prec=ratio) |
| v3 | nenhum | 0.5 | 2.0 | 88/12 | F1=0.93 trivial (mesmo padrão) |
| **v4** | **50/50** | **0.5** | **2.0** | **26/74** | **F1=0.68, rec=0.93, prec=0.58 (REAL)** |

#### 6.3.5 Solution: aggressive re-balancing of the dataset

Reescrevemos `build_painted_dataset.py` para gerar **2 negativos aleatórios por positivo YOLO**:

```python
# Para cada YOLO detection (positivo)
# → adiciona 2 random crops sem GT overlap (negativos)
n_negatives_to_add = max(3, 2 * len(yolo_preds))
for neg_i in range(n_negatives_to_add):
    neg_bbox = random_bbox(w=60-300, h=30-150)
    if IoU(neg_bbox, qualquer_gt) > 0.1:
        continue  # rejeita overlap com drone real
    fake_conf = uniform(0.3, 0.9)  # mimica confidence falsa
    painted = paint_pc(pc, neg_bbox, fake_conf)
    # ... save como negativo
```

Resultado: dataset **dataset_painted v2** com 16568 samples (25.6% pos / 74.4% neg).

### 6.4 Comparison against YOLO

**Pergunta crítica analisada**: o detector 3D pode superar a precisão do YOLO?

**Resposta detalhada**:

| Métrica | Recall | Precision | 3D IoU |
|---|---|---|---|
| YOLO sozinho | 73% (mAP) | 91% | N/A |
| Sistema YOLO + PointNet++ Painted | **bounded por YOLO** | **pode aumentar** | **pode aumentar** |

- **Recall**: bounded por YOLO. Se YOLO não detecta, PointNet não tem prob channel para trabalhar.
- **Precision**: PointNet++ pode filtrar false positives YOLO (rejeitando bboxes onde geometria 3D não bate).
- **Localização 3D**: PointNet pode refinar drasticamente (MSE 0.003 << YOLO bbox 2D).

---

## 7. Experimentos e Resultados Comparativos

### 7.1 Complete table of every experiment

| # | Modelo | Dataset | Train Task | Val Acc | Val F1 | Prec | Rec | C MSE | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | PointNet baseline | `dataset_pointnet` (67/33) | drone vs random | **0.967** | - | - | - | 3.6 | ok |
| 2 | PointNet Painted v1 | `dataset_painted` (88/12) | drone vs YOLO FP | 0.499 | - | 0.99 | 0.35 | 0.003 | bias |
| 3 | PointNet Painted v2 | `dataset_painted` (88/12) | drone vs YOLO FP | 0.651 | - | 0.96 | 0.62 | **0.003** | ok |
| 4 | PointNet++ v1 (α=0.25) | `dataset_painted` (88/12) | drone vs YOLO FP | 0.405 | 0.45 | 0.91 | 0.32 | 0.004 | bias |
| 5 | PointNet++ v2 (α=0.75) | `dataset_painted` (88/12) | drone vs YOLO FP | 0.87* | 0.93* | 0.87 | 1.0 | 0.012 | trivial |
| 6 | PointNet++ v3 (α=0.5) | `dataset_painted` (88/12) | drone vs YOLO FP | 0.87* | 0.93* | 0.87 | 1.0 | 0.013 | trivial |
| 7 | **PointNet++ v4** | `dataset_painted_v2` (26/74) | drone vs YOLO FP + random | **0.775** | **0.68** | 0.58 | **0.93** | **0.003** | ok |

\* = solução trivial (predição constante)

### 7.2 YOLO11s

| Métrica | Valor |
|---|---|
| mAP@0.5 | 0.731 |
| mAP@0.5:0.95 | 0.403 |
| Precision | 0.91 |
| Recall | 0.64 |

### 7.3 Summary of the final production models

| Componente | Modelo | Métrica chave | Path |
|---|---|---|---|
| **Detector 2D** | YOLO11s | mAP 0.73 | `runs/drone/drone_urban_v1/weights/best.pt` |
| **Detector 3D classify** | PointNet baseline | acc 0.97 | `runs/pointnet/drone_detector/best.pt` |
| **Detector 3D regress** | PointNet++ Painted v4 | MSE 0.003, F1 0.68 | `runs/pointnet2_painted/v4_balanced/best.pt` |

---

## 8. Technical analysis and insights

### 8.1 Why is the PointNet baseline more accurate than PointNet++ Painted?

**Aparente paradoxo**: arquitetura mais simples (PointNet) tem maior accuracy que PointNet++ painted. Resolução:

1. **Tarefa diferente**:
   - Baseline classifica "frustum tem drone vs random crop sem drone" — **fácil** (random crops têm geometria diferente de drones)
   - Painted classifica "drone vs YOLO false positive" — **difícil** (FP têm geometria parecida porque YOLO detectou algo)

2. **Distribuição diferente**:
   - Baseline tem negatives "fáceis" (random)
   - Painted tem negatives "hard" (YOLO FP — alguma estrutura)

3. **Padronização não comparável**: comparar 0.967 vs 0.78 não diz que arquitetura é pior. Painted resolve problema mais útil em deployment.

### 8.2 Why does PointPainting work for regression?

PointNet baseline regression tem MSE 3.6 (normalized). PointPainting tem MSE 0.003 (1200× melhor). Por quê?

**Hipótese**: o canal `prob` atua como **attention soft** — concentra o modelo na região de interesse, mas mantém **contexto espacial** dos pontos vizinhos. Contexto ajuda regression (modelo sabe escala absoluta da cena) mas não tanto classification (que só precisa "tem drone aqui?").

### 8.3 Class-balancing iterations

Lições aprendidas durante 4 iterações de PointNet++ Painted:

1. **WeightedRandomSampler + BCE pos_weight**: dupla-correção causa bias forte para classe minoritária após sampler
2. **Sem sampler + Focal Loss α=0.5/0.75**: focal loss sozinho insuficiente para imbalance > 80/20, modelo defaulta para majoritária
3. **Solução real**: rebalancear o **dataset** (não a loss). Sintetizar negativos via random crops para atingir ~25/75 ratio
4. **Métrica robusta**: F1 (não accuracy) — accuracy é enganosa com classes desbalanceadas

### 8.4 Fundamental limitations identified

#### 8.4.1 API Detection miss rate (~36%)

`simGetDetections` filtra drones que estão tecnicamente em FOV mas marginalmente cobertos pela renderização. Isso é **server-side**, não corrigível pelo nosso código sem inventar bboxes (que causa ghosts).

#### 8.4.2 Drone drift after a teleport

Mesmo com `hoverAsync`, drones derivam 3-4m em 4s. **Solução**: re-teleport por frame.

#### 8.4.3 Sparse LiDAR at high altitudes

Em cenários abertos (Coast, City alta altitude), maioria dos rays LIDAR vai para sky (sem return). Em NH (suburbano com casas próximas), LIDAR é denso (~10k pts/scan).

#### 8.4.4 AirSim `relative_pose` coordinate convention

API `simGetDetections` retorna `relative_pose.position` em frame ambíguo (não corresponde a CV camera frame nem NED body frame standard). Empiricamente não consegui decodificar. **Workaround**: ignorar `relative_pose`, usar `simGetVehiclePose(drone)` em coordenadas world.

---

## 9. Known limitations

### 9.1 Dataset

- **Single class**: apenas "drone". Sistema não distingue Drone3/Drone4/Intruder1 individualmente.
- **3 ambientes**: cobertura UAM limitada. Não inclui ambientes industriais, áreas rurais, ambientes noturnos extremos.
- **Single ego**: apenas câmera frontal do Ego. Não testa multi-view.
- **3 drones por scenario**: limita densidade. Em cenários UAM reais pode haver dezenas de drones.

### 9.2 YOLO

- **Recall 64%**: bottleneck do sistema. 36% dos drones reais não são detectados.
- **mAP@0.5:0.95 = 0.40**: bbox accuracy moderada. Drones tiny < 10×10 pixels têm IoU degraded.
- **Não testado em vídeo real**: trained on synthetic, sim-to-real gap não avaliado.

### 9.3 PointNet variants

- **Classification ainda imperfeita**: F1 máximo 0.68 no painted setting
- **3D bbox size regression menos robusta**: size MSE 0.23 (normalized) — drones quase sempre tamanho similar (1m), pouca variância pra aprender
- **No multi-instance native**: cada YOLO bbox processada independentemente. Não modela interações entre drones.

### 9.4 Sistema end-to-end

- **Não foi avaliado de ponta a ponta**: cada componente foi treinado/avaliado separadamente.
- **Latência não medida**: PointNet++ com N=4096 pode ser lento (FPS é O(N²)). Não rodamos benchmarks.
- **Tracking não integrado**: SORT 3D existe no projeto separado mas não testado com nossos modelos.

---

## 10. File and script layout

### 10.1 Project directory

```
<DATA_ROOT>/
│
├── DOCUMENTACAO_TECNICA.md          ← Este arquivo
├── settings_urban_uam.json          ← AirSim settings (deploy em Documents/AirSim/)
│
├── generate_dataset_urban.py        ← Coleta principal (1099 linhas)
├── build_pointnet_dataset.py        ← Build frustum dataset
├── build_painted_dataset.py         ← Build painted dataset
├── run_yolo_inference.py            ← YOLO inference sobre frames
├── render_all_visualizations.py     ← Visualizações com bboxes
├── convert_pointclouds_to_ply.py    ← Convert PC para CloudCompare
├── convert_frustum_to_ply.py        ← Convert frustums para CloudCompare
│
├── train_yolo11s_drone.py           ← Treino YOLO11s
├── train_pointnet.py                ← Treino PointNet baseline
├── train_pointnet_painted.py        ← Treino PointNet Painted
├── train_pointnet2_painted.py       ← Treino PointNet++ Painted
│
├── dataset_nh_v2/                   ← 1000 frames AirSimNH
├── dataset_city_v3/                 ← 1000 frames CityEnviron
├── dataset_coast_v2/                ← 1000 frames Coastline
├── dataset_urban_merged/            ← Merged YOLO 3000 frames
├── dataset_pointnet/                ← Frustum dataset (9136 samples)
├── dataset_painted/                 ← Painted dataset (16568 samples)
│
├── runs/drone/drone_urban_v1/       ← YOLO11s weights + logs
├── runs/pointnet/drone_detector/    ← PointNet baseline
├── runs/pointnet_painted/.../       ← PointNet Painted v1, v2
└── runs/pointnet2_painted/.../      ← PointNet++ v1-v4
```

### 10.2 File formats

#### Image (.jpg)
1280×720 RGB, padrão JPEG.

#### YOLO label (.txt)
```
class_id center_x center_y width height
0 0.5234 0.4012 0.0234 0.0156
```
Coordenadas normalizadas [0, 1].

#### Point Cloud depth-derived (.npy)
NumPy float32 array, shape `(N, 3)`.
**Frame**: CV camera (X=right, Y=down, Z=forward, origem na câmera).

#### Point Cloud LIDAR (.npy)
NumPy float32 array, shape `(N, 3)`.
**Frame**: SensorLocalFrame (alinhado com câmera após config).

#### Label 3D (.json)
```json
[{
  "class": "drone",
  "name": "Drone4",
  "bbox_2d": [427, 369, 453, 379],
  "center": [37.23, -13.24, 0.43],
  "distance_m": 39.52,
  "box3D_min": [...],
  "box3D_max": [...]
}]
```

#### Painted PC (.npy)
NumPy float32 array, shape `(N=4096, 4)`.
4 channels: `[x, y, z, prob_yolo]`.

### 10.3 Scripts principais (detalhes)

#### `generate_dataset_urban.py`
- **CLI**: `--env {neighborhood, city, coastline} --frames N --output DIR`
- **Output**: yolo/, pointnet/, visualizations/, metadata/
- **Características**:
  - `simPause` para sync de queries
  - Re-teleport per-frame
  - Domain randomization (weather, time of day)
  - Multi-env profiles
  - LIDAR + depth + RGB + segmentation collection

#### `build_painted_dataset.py`
- **Inputs**: `dataset_nh_v2`, `dataset_city_v3`, `dataset_coast_v2` + YOLO predictions
- **Output**: `dataset_painted/`
- **Per-sample**: full PC subsampled para N=4096 + prob channel painted via YOLO bbox

#### `train_pointnet2_painted.py`
- PointNet++ from-scratch (FPS, ball query, SA)
- Focal Loss
- WeightedRandomSampler
- Métricas: F1, precision, recall, center MSE
- Salva best por F1 (não acc)

---

## 11. References

### 11.1 Bibliografia principal

1. **Qi, C. R., et al.** (2017). *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR.
2. **Qi, C. R., et al.** (2017). *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.* NeurIPS.
3. **Qi, C. R., et al.** (2018). *Frustum PointNets for 3D Object Detection from RGB-D Data.* CVPR.
4. **Vora, S., et al.** (2020). *PointPainting: Sequential Fusion for 3D Object Detection.* CVPR.
5. **Lin, T. Y., et al.** (2017). *Focal Loss for Dense Object Detection.* ICCV.
6. **Ultralytics** (2024). *YOLO11 Documentation.* https://docs.ultralytics.com/

### 11.2 Software

- **AirSim**: Microsoft Open Source. https://github.com/Microsoft/AirSim
- **Cosys-AirSim**: KU Leuven fork. https://github.com/Cosys-Lab/Cosys-AirSim
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **PyTorch**: https://pytorch.org/

### 11.3 Reference datasets

- **KITTI**: Geiger et al. (2012) — baseline para outdoor 3D detection
- **SemanticKITTI**: Behley et al. (2019) — semantic segmentation in LIDAR
- **nuScenes**: Caesar et al. (2020) — multi-modal autonomous driving

---

## Appendix A: commands to reproduce

### A.1 Data collection (run once per environment)

```bash
# Pré-requisitos: copiar settings_urban_uam.json para C:\Users\<user>\Documents\AirSim\settings.json
# Abrir o .exe do ambiente AirSim no Windows

# WSL:
cd <DATA_ROOT>

# AirSimNH
python3 generate_dataset_urban.py --env neighborhood --frames 1000 --output dataset_nh_v2

# CityEnviron
python3 generate_dataset_urban.py --env city --frames 1000 --output dataset_city_v3

# Coastline
python3 generate_dataset_urban.py --env coastline --frames 1000 --output dataset_coast_v2
```

### A.2 Build datasets derivados

```bash
# Frustum dataset (para PointNet baseline)
python3 build_pointnet_dataset.py

# Painted dataset (para PointPainting)
python3 run_yolo_inference.py  # YOLO inference em todos os frames
python3 build_painted_dataset.py
```

### A.3 Treinamento

```bash
# YOLO11s
python3 train_yolo11s_drone.py --epochs 100 --batch 16

# PointNet baseline
python3 train_pointnet.py --epochs 50 --batch 64

# PointNet Painted
python3 train_pointnet_painted.py --epochs 40 --batch 16

# PointNet++ Painted (v4, com dataset balanceado)
python3 train_pointnet2_painted.py --epochs 30 --batch 8 --focal_alpha 0.5
```

### A.4 Conversion for CloudCompare

```bash
python3 convert_pointclouds_to_ply.py dataset_nh_v2 --limit 50
python3 convert_frustum_to_ply.py --root dataset_pointnet --limit 30
```

---

## Appendix B: metrics of each run

### B.1 YOLO11s training (`runs/drone/drone_urban_v1/results.png`)
- 100 epochs completos
- Val mAP@0.5 estabilizou em ~0.73 após epoch 60
- Sem early stopping triggered

### B.2 PointNet baseline (`runs/pointnet/drone_detector/training_log.json`)
- 50 epochs
- Val accuracy 0.967 em epoch 36

### B.3 PointNet++ v4 (`runs/pointnet2_painted/v4_balanced/training_log.json`)
- 30 epochs
- Best val F1 0.682 em epoch 25
- Center MSE estabilizou em 0.003 desde epoch 15

---

**Fim do documento**

*Para questões técnicas ou esclarecimentos, ver inline comments nos scripts referenciados.*
