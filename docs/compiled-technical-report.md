# Complete Technical Compilation — 3D Drone Detection and Localization/Tracking in Simulated UAM Environments (AirSim)

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** This is the project's master technical record and the authoritative
> provenance document: Section 13.3 is a hand-written manifest of the canonical
> code, Section 13.4 lists every key constant verbatim, Section 4.3 documents the
> three automatic-labeling routes that failed, and Section 9.3 states the
> in-sample caveat for the voxel results. Its own result tables **predate** the
> dissertation's final numbers and disagree with them; why has not been traced —
> see "Known discrepancies" in the top-level README.md. Section 13.3 also lists
> `record_campaign.sh`, which is **not present** in this repository.


> Documento-mestre de referência para a dissertação de mestrado. Reúne, de forma exaustiva e
> cronológica, **todo o roadmap do projeto**: contexto, infraestrutura de simulação, coleta de
> dados (duas gerações de datasets), correções geométricas críticas, **todas as arquiteturas
> treinadas** (com hiperparâmetros exatos), metodologia de avaliação 3D-MOT, resultados
> consolidados e achados científicos. Numeração de seções pensada para mapear capítulos da
> dissertação.

---

## 0. Executive summary

Desenvolveu-se um sistema completo de **detecção 2D + localização e rastreamento 3D de drones**
para o cenário de *Urban Air Mobility* (UAM), usando o simulador **AirSim** (binários Microsoft
1.8.x, cliente `cosysairsim 3.3.0`) sobre WSL/Linux. O pipeline final é de **dois estágios** —
detector 2D **YOLOv11s** → recorte de **frustum** na nuvem de pontos (depth) → rede 3D
(**PointNet++ / segmentação T-Net**) → **fusão YOLO×PointNet** → **tracker SORT 2D** — comparado
contra um paradigma alternativo de **um estágio** baseado em **voxel/BEV** (PointPillars e
variantes). A avaliação rigorosa em **60 sequências de vídeo** (3 ambientes, observador em
movimento) com **métricas padrão de 3D-MOT** (AMOTA/AMOTP estilo nuScenes, CLEAR-MOT, detecção e
erro 3D) estabeleceu:

- **Localização 3D de ~0,78–1,2 m de RMSE** com alta precisão de detecção (0,89–1,00);
- O **gargalo de recall** é o detector 2D (YOLO) em alvos distantes/baixa visibilidade;
- O paradigma **voxel com features de pillar aprendidas (PFN)** alcança a **melhor localização de
  todo o estudo (0,78 m)** mantendo recall alto (0,80), sem o gargalo do YOLO;
- A heurística *depth-band* (corte de fundo) pode ser substituída por **segmentação aprendida
  (T-Net)**, que iguala/supera a localização sem corte manual.

---

## 1. Contexto, problema e objetivos

### 1.1 Urban Air Mobility (UAM)
Operação de veículos aéreos (drones, eVTOL) em ambiente urbano, tipicamente 30–120 m AGL (espaço
aéreo Classe G). Aplicações: entrega de última milha, inspeção de infraestrutura, sky-taxi,
vigilância. Sistemas autônomos UAM exigem **sense-and-avoid**: detectar e localizar no espaço 3D
o tráfego aéreo próximo, rastrear múltiplos alvos e raciocinar geometricamente para planejamento.

### 1.2 Central technical challenges
1. Drones são **objetos pequenos** (~1 m de envergadura) a 20–100 m → poucos pixels e poucos
   retornos de profundidade (esparsidade).
2. **Fundo dinâmico e variado** (prédios, vegetação, céu, água).
3. Câmera monocular **não fornece profundidade direta** (resolvido via depth camera/LIDAR sim).
4. **LIDAR esparso em altitude** (sem retorno no céu).

### 1.3 Objetivos
(i) Coletar datasets multimodais sintéticos em ambientes UAM diversos; (ii) treinar um detector 2D
robusto; (iii) treinar detectores 3D para localização precisa; (iv) comparar paradigmas
(frustum 2-estágios vs voxel 1-estágio), técnicas de fusão e a substituição de heurísticas por
componentes aprendidos; (v) avaliar end-to-end com métricas reconhecidas de 3D-MOT.

---

## 2. Project roadmap (chronological view)

| Fase | Foco | Entregas principais |
|------|------|---------------------|
| **F1 — Fundação** | Coleta multimodal real no AirSim; primeiros detectores 3D | 3×1000 frames (RGB+depth+LIDAR+seg); YOLO urbano (mAP50 0,73); PointNet baseline; PointPainting; PointNet++ painted v1–v7; descoberta de bugs de simulação/geometria |
| **F2 — Pipeline 2-estágios** | Falha da rotulagem automática → **dataset YOLO sintético**; correção da bbox 3D; frustum PN2; depth-band; fusão; SORT | 3 vias automáticas falham (seg/API/projeção) → composição por sprites; YOLOv11s sintético (mAP50 0,994); VisQuad 4×; bbox 3D manual; frustum PN2 (val-F1 0,73); depth-band (RMSE 3,0→1,2 m); fusão YOLO×PN; SORT 2D + coast |
| **F3 — Avaliação científica** | Campanha de vídeos; métricas 3D-MOT | 60 sequências (ego móvel); AMOTA/AMOTP, CLEAR-MOT, det P/R/F1, erro 3D; protocolo de *dump* |
| **F4 — Alternativas à heurística** | Substituir o depth-band; paradigma voxel | Normalização robusta p95 (negativo); Frustum-ConvNet; Segmentação T-Net; PointPillars-lite; fusão tardia |
| **F5 — Experimentos voxel** | O que move o ponteiro no voxel | PFN (features aprendidas); voxel 3D denso; resolução fina (negativo); temporal multi-frame (negativo); fusão T-Net⊕PFN |

---

## 3. Simulation infrastructure and geometry

### 3.1 Stack
| Componente | Versão | Papel |
|---|---|---|
| AirSim (binários) | Microsoft 1.8.x | Simulação física + ambientes + sensores |
| cosysairsim (cliente Py) | 3.3.0 | Cliente RPC (com *shims* de compatibilidade) |
| Python | 3.10 | — |
| PyTorch | 2.9.1+cu128 | Treino GPU |
| Ultralytics | YOLO11s | Detector 2D |
| OpenCV / NumPy / SciPy / motmetrics | — | Visão, álgebra, métricas MOT |

Topologia: AirSim roda no **Windows**; os scripts Python no **WSL**, comunicando por TCP/RPC
(msgpack) com o gateway WSL (ex.: `<AIRSIM_HOST>:41451`).

### 3.2 `settings.json` (essencial)
- **Ego** `SimpleFlight` em `(0,0,−5)`; câmera `front_center` em **offset body `(0.35, 0.0, −0.5)`
  m**, **pitch −15°**, roll/yaw 0.
- **Capturas** (1280×720, FOV 90°): ImageType 0 (RGB), 1 (DepthPlanar), 2 (Seg), 5 (depth aux).
- **LIDAR** `LidarFront` (SensorType 6): 64 canais, 1 Mpt/s, 20 rps, **mesmo offset/pitch da
  câmera**, FOV V ±30°, FOV H ±45°, alcance 150 m, frame `SensorLocalFrame`. *Decisão de projeto:*
  alinhar LIDAR à câmera simplifica a projeção 3D↔2D (frame compartilhado).
- Veículos extra: `Drone3 (15,−5,−5)`, `Drone4 (15,5,−5)`, `Intruder1 (−15,0,−5, yaw 180)`.

### 3.3 *Compatibility shims* (cosys-airsim 3.3 ↔ Microsoft 1.8)
*Porquê:* o cliente `cosysairsim 3.3.0` foi feito para o fork Cosys-AirSim, mas operávamos com
binários **Microsoft 1.8.x**, cujas assinaturas RPC divergem — vários *endpoints* do Microsoft
exigem um argumento posicional **`external: bool`** que o cosys não envia (ou envia como
`annotation_name` string), causando `RPCError: bad cast`/contagens erradas. Sondou-se cada método
e mapeou-se o **slot** do `external=False`:

| API | slot de `external` |
|---|---|
| `simGetImages` | 3 |
| `simGetDetections` | 4 |
| `simClearDetectionMeshNames` | 4 |
| `simSetDetectionFilterRadius` | 5 |
| `simAddDetectionFilterMeshName` | 5 |

*Solução:* funções `legacy_*` que chamam `client.client.call(<api>, …, external)` diretamente e
desserializam com `from_msgpack` (`generate_dataset_urban.py:39–81`). Métodos de weather/pose/
collision funcionam direto pelo wrapper cosys.

### 3.4 Camera model (intrinsics) — exact values
```
IMAGE_W, IMAGE_H = 1280, 720
FOV_H = 90°
FX = IMAGE_W / (2·tan(FOV_H/2)) = 1280 / (2·tan 45°) = 640.0
FY = FX = 640.0            # pixels quadrados (convenção AirSim, calibrado empiricamente)
CX, CY = 640.0, 360.0
Faixa de profundidade válida: 0.1 m ≤ z < 250 m
```
*Nota de calibração:* inicialmente assumiu-se `FY = H/(2·tan(FOV_V/2))` com FOV_V=60° → FY≈623;
a calibração contra `relative_pose` revelou **pixels quadrados** (FY=FX=640). Correção pequena,
porém essencial ao alinhamento nuvem↔bbox 3D.

### 3.5 Coordinate frames
- **NED** (mundo): X=norte, Y=leste, Z=baixo.
- **FRD** (corpo): X=frente, Y=direita, Z=baixo.
- **CV** (câmera): X=direita, Y=baixo, Z=frente.

**Matriz de rotação** (ZYX Tait–Bryan, mundo→câmera), `ekf_3d_global.rotation_matrix(φ,θ,ψ)`:
```
R = R_yaw(ψ) · R_pitch(θ) · R_roll(φ)
R_roll  = [[1,0,0],[0,cφ,−sφ],[0,sφ,cφ]]
R_pitch = [[cθ,0,sθ],[0,1,0],[−sθ,0,cθ]]
R_yaw   = [[cψ,−sψ,0],[sψ,cψ,0],[0,0,1]]
```
> **Correção crítica de bug:** mundo→câmera usa **Rᵀ**, não R. Tanto a projeção
> (`project_global_to_pixel`) quanto a transformação de medida (`measurement_to_global`) foram
> corrigidas para `p_frd = Rᵀ·(p_global − x_plat[:3])`. O bug (uso de R) **invertia o eixo
> vertical** da projeção (drones apareciam no fundo da imagem). Detectado por diagnóstico de
> dupla-projeção contra o ground-truth.

### 3.6 Depth → point cloud (CV frame)
```python
# inference_pipeline.depth_to_pointcloud_cv(depth, max_depth=250)
x = (u − CX)·z/FX     # CV X = direita
y = (v − CY)·z/FY     # CV Y = baixo
z =  depth            # CV Z = frente
```

### 3.7 Body → camera transform (lever arm + pitch)
`p_cam = R_pitch(+15°)·(p_body − [0.35,0,−0.5])`, com
`R_pitch(+15°) = [[cos15°,0,sin15°],[0,1,0],[−sin15°,0,cos15°]]` ≈ `[[0.966,0,0.259],[0,1,0],[−0.259,0,0.966]]`.

### 3.8 Cloud source: depth (primary) vs. LiDAR (secondary) — why
A nuvem de pontos primária vem da **DepthPlanar** (depth monocular), **não** do LIDAR. *Porquê:*
em altitudes UAM (30–120 m AGL) o drone voa **contra o céu**, região onde o **LIDAR não tem
retorno** (sem superfície para refletir) → nuvem LIDAR **esparsa/ausente** justamente sobre o alvo.
Tentou-se mitigar **expandindo o FOV vertical** do LIDAR ("olhar mais para cima": `VerticalFOVUpper`
45→70°, `Lower` −45→−20°, `LIDAR_CONFIG_GUIDE.md`), mas a *depth* densa (100k–330k pts/frame) provou
ser a fonte confiável para drones aéreos. O LIDAR (64 canais, 1 Mpt/s, alcance 150 m) é **alinhado à
câmera** (mesmo offset/pitch, `DataFrame=SensorLocalFrame`) — *porquê:* frame compartilhado
simplifica a projeção 3D↔2D e a fusão; fica como sensor secundário/validação.

### 3.9 Synchronization and drift (`simPause` + re-teleport) — why
Entre `simGetImages` e `simGetVehiclePose`, os drones se movem por gravidade/física → **drift de
~1 m em 0,5 s** medido (`generate_dataset_urban.py:1229`), desalinhando a bbox renderizada da pose
retornada. *Solução em duas camadas:* (i) **`simPause(True)`** congela a cena durante **todo** o
bloco de *queries* (captura atômica); (ii) **re-teleporte por frame** de ego e drones para as poses
planejadas + `sleep(0.18 s)` para assentar a física *antes* de pausar — *porquê:* `hoverAsync` não
segura os drones de forma confiável (drift de 3–4 m em 4 s ao acumular vários *yaw frames*).

---

## 4. Data collection

### 4.1 Phase 1 — Multimodal collected dataset (AirSim collection)
**Três ambientes** que cobrem o espectro UAM: **AirSimNH** (suburbano, entrega de última milha,
ego 30–50 m AGL), **CityEnviron** (cidade densa, corredores aéreos, ego 70–100 m AGL),
**Coastline** (litoral/vegetação, ego 25–55 m AGL).

| Ambiente | Frames | Modalidades | Tamanho |
|---|---|---|---|
| AirSimNH | 1000 (800/200) | RGB + Depth (1280×720) + LIDAR 64ch + Seg + labels 3D | 7,0 GB |
| CityEnviron | 1000 | idem | 7,7 GB |
| Coastline | 1000 | idem | 7,4 GB |
| **Total** | **3000** | — | ~22 GB |

**Parâmetros de coleta:** 1–3 drones/frame (média ~2,0); distância 8–65 m (média ~35 m); 4 *yaw
offsets* por cenário (drone estático, ego rotaciona); *domain randomization* de weather
(clear 57%, fog_light Fog=0.15 14%, rain_light Rain=0.25 14%, dust_light Dust=0.2 15%) e hora do
dia (07/09/11/13/15/17/18 h); filtro de contraste `std(cinza) > 12` (rejeita frames lavados por
fog). Densidade de nuvem: depth 100k–330k pts/frame (média ~200k); LIDAR ~9k–12k pts (NH/City),
mais esparso em Coast.

#### 4.1.1 Per-frame protocol (synchronization)
```
1. simSetVehiclePose(ego, drones)         # teleporte
2. simPause(True)                          # congela a cena
   a. simGetImages([RGB, DepthPlanar, Seg])
   b. simGetDetections()                   # bbox API
   c. simGetVehiclePose(todos)             # poses
   d. getLidarData()
3. simPause(False)
4. depth_to_pointcloud (frame CV) → salva .jpg/.npy/.json
```
A **pausa de simulação** é crítica: sem ela, drones movem ~1,18 m em 0,5 s entre *queries* →
*misalignment* imagem↔pose. Mesmo com `hoverAsync`, há **drift de ~3–4 m em 4 s** → solução:
**re-teleportar** ego e drones no início de cada *yaw frame*.

#### 4.1.2 Posicionamento *collision-aware* (`find_safe_pose`)
Amostragem polar (raio `r∈[r_min,r_max]`, *bearing* `θ∈[−0.85·FOV/2, +0.85·FOV/2]` em torno do
yaw do ego — viés para o FOV), altitude `z∈[ego_z±12]` clipada; teleporte com
`ignore_collision=True`; validação por `simGetCollisionInfo` (comparação de timestamp); *retry*
subindo 8 m; *fallback* a `ego_z−40 m` após 18 tentativas.

#### 4.1.3 Depth-based occlusion filters (two thresholds)
`simGetDetections` faz *frustum check*, não *occlusion check* (retorna bbox de drone fisicamente
atrás de um prédio). Implementaram-se **dois** filtros por *depth*, com limiares distintos por
finalidade:
- **`visibility_status` (rigoroso, 35%):** conta os pixels da bbox que mostram *depth*
  significativamente **menor** que a distância esperada do drone (`occluder = depth>0.1 ∧
  depth < dist_esp − tol`, `tol = max(2.5; 0.20·dist)`). Pixels de **céu** (`depth ≥ 100 m`) e na
  profundidade do drone **não** contam (drone contra o céu = OK). Se a fração de *occluders* >
  **0,35** → `occluded`.
- **Filtro mínimo do loop principal (60%):** versão permissiva (`tol = max(3; 0.25·dist)`,
  rejeita só `occluder_frac > 0.60`) — descarta apenas oclusão **forte**, mantendo oclusão parcial.
> *Porquê de dois limiares:* o 35% serve à **curadoria de qualidade** (rótulos limpos para
> avaliação); o 60% serve à **coleta em massa**, onde oclusão parcial ainda é amostra válida e
> descartar demais reduziria o recall do dataset.

#### 4.1.4 The API limitation that forced the change of strategy
`simGetDetections` perde ~36% dos drones tecnicamente em FOV (filtragem *server-side* mais
conservadora que a renderização). Mais grave que o recall: **não havia caminho confiável de
rotulagem automática** no AirSim 1.8 para o nosso mesh, e cada *query* da API era cara/instável.
Isso **motivou a Fase 2** — abandonar a rotulagem automática e construir um **dataset sintético**
com rótulos perfeitos. A árvore de decisão completa (três tentativas falhas) está na §4.3.

### 4.2 The enlarged visual mesh (VisQuad 4×) — why and how
**Problema:** o multirotor nativo do AirSim é **pequeno demais** (<1 m de envelope) para gerar
pixels suficientes ao YOLO e retornos de *depth* densos a 20–100 m. **Decisão:** anexar a cada
drone um mesh visual `Quadrotor1` em **escala 4×** (`DroneVisualizer`/VisQuad). *Porquê 4×:* foi a
escala que tornou o drone consistentemente detectável pelo YOLO e amostrável pela nuvem nas
distâncias UAM **sem** distorcer demais a geometria; **o YOLO foi treinado exatamente nesse mesh
4×** (`generate_dataset_urban.py:600`: *"sem isso não detecta"*), portanto o VisQuad é obrigatório
também na inferência.
- **Spawn e sincronização:** `simSpawnObject(VisQuad_<drone>, "Quadrotor1", pose_global, scale=4,
  physics_enabled=False)`; uma **thread daemon a 30 Hz** reespelha a pose real de cada drone no
  mesh visual (`simSetObjectPose`), garantindo que o que o YOLO vê coincide com o ground-truth.
- **Bug de frame de origem (corrigido):** `simGetVehiclePose` retorna pose **relativa à origem** do
  veículo (declarada no `settings.json`), mas `simSetObjectPose` espera **global**. Sem somar a
  origem (`VEHICLE_ORIGINS = {Ego:(0,0,−5), Drone3:(15,−5,−5), Drone4:(15,5,−5),
  Intruder1:(−15,0,−5)}`) o mesh visual fica **deslocado** do drone real. Corrigido em
  `vehicle_to_global()`.
- **Extents medidos** (empiricamente, via `simSpawnObject`+`box3D`, em `measure_quadrotor.py`):
  **3,01 × 3,93 × 2,79 m** (fwd×right×down); meias-extensões usadas `[3.01,3.93,2.79]·1.38/2 =
  [2.077, 2.713, 1.925] m`. *Porquê do fator 1,38:* o AABB do AirSim **exclui as hélices girando**;
  a margem cobre o disco dos rotores.

### 4.2.1 Critical correction of the 3D bounding box (the central finding)
A bbox 3D **não** é obtida de `det.relative_pose`/`det.box3D`. *Porquê:* o AirSim 1.8 tem **dois
bugs** (comentário verbatim em `generate_dataset_urban.py:960`: *"AirSim tem bug de pitch que
offseta ~5 m em down e box3D tem tamanho subestimado"*):
1. **Offset de pitch no `relative_pose`** desloca o centro do drone em **~5 m** no eixo *down*.
2. **`box3D` subdimensionado** — não contém o mesh de forma confiável (pior sob rotação).

Por isso a bbox 3D é **calculada manualmente** a partir da pose global real:
- **Centro:** mundo→corpo do ego (Rᵀ) → subtrai offset da câmera → aplica pitch +15° → reordena
  para CV.
- **Caixa:** 8 cantos do AABB (com a **rotação de yaw** do drone, assumindo roll=pitch≈0 em *hover*)
  → mundo → CV; AABB no frame CV. **Rejeição** se qualquer canto cai atrás da câmera (`fwd < 0,5 m`)
  — senão a projeção degenera (a bbox "vira 2 pontos" no PLY).
- **Filtros de qualidade:** ≥5 pontos de nuvem dentro da bbox; `area_bbox > 25 px`.

### 4.3 Phase 2 — The automatic-labeling crisis and the turn to the synthetic dataset
Para treinar o YOLO precisávamos de **bounding boxes 2D** confiáveis em escala. Tentaram-se **três**
vias automáticas no AirSim 1.8; **todas falharam** — daí a decisão pelo dataset sintético.
Documenta-se cada tentativa e o motivo exato do fracasso (é o ponto-chave do roadmap).

#### 4.3.1 Attempt 1 — Semantic segmentation (`ImageType.Segmentation`) — **failed**
- **Ideia:** o AirSim mapeia *Object IDs* → cores RGB (`ID = R + G·256 + B·256²`); bastaria isolar
  a cor do drone e extrair a bbox por componentes conexas. Config `SegmentationSettings.InitMethod =
  "CommonObjectsRandomIDs"`.
- **Bug do AirSim 1.8 (causa-raiz):** os meshes de drone (blueprint `BP_FlyingPawn`) **não recebem
  um *Stencil ID* único** — todas as instâncias caem na **mesma cor** ou se confundem com o fundo.
  Atribuir IDs únicos exigiria configurar *Custom Depth-Stencil* no Unreal Engine e **recompilar**
  os binários do ambiente — **inviável** (usávamos binários pré-compilados Microsoft).
- **Heurística de contorno (também rejeitada):** detectar "objetos escuros e pequenos"
  (`soma_RGB < 100`, área 0,01–2% da imagem) na máscara — gerou **muitos falsos positivos**
  (sombras, artefatos) e falsos negativos (drone claro sob certa luz), **sem validação possível**.

#### 4.3.2 Attempt 2 — Detection API (`simGetDetections`) — **failed**
- **Ideia:** usar a API oficial com filtros (`simSetDetectionFilterRadius`,
  `simAddDetectionFilterMeshName`).
- **Falha:** testaram-se **>20 padrões de nome de mesh** (`"*"`, `"BP_FlyingPawn*"`, `"*FlyingPawn*"`,
  `"Drone*"`, `"Intruder*"`, `"SimpleFlight*"`, `"*Pawn*"`, `"BP_*"`, `"*_C"`, …) e **todos
  retornaram 0 detecções** — a API exige anotações pré-configuradas no Unreal que **não existiam**
  para o blueprint. Quando funcionava parcialmente, ainda **perdia ~36%** dos drones em FOV.

#### 4.3.3 Attempt 3 — 3D→2D projection of the known pose — **failed**
- **Ideia:** projetar `simGetVehiclePose(drone)` para pixel com o modelo pinhole (rótulo "de graça").
- **Falha:** **incompatibilidade de convenção** NED (X=frente, Y=direita, Z=baixo) ↔ CV
  (X=direita, Y=baixo, Z=frente). As primeiras versões ignoravam a **orientação** da câmera
  (usavam K sem `R|t`), gerando **erros sistemáticos de 5–30 px**; e **não havia ground-truth
  independente** para validar a bbox projetada (o alinhamento depth↔RGB tinha *shift* sub-pixel).
  > Observação: essa convenção foi **depois** corrigida (a transformação `Rᵀ` da §3.5/§4.2.1) e é o
  > que torna a bbox **3D** confiável — mas para gerar **milhares** de rótulos 2D de treino no
  > momento certo, a via de projeção foi abandonada em favor do sintético.

#### 4.3.4 Decision: synthetic composition (perfect labels, decoupled from the API)
Conclusão: **nenhuma via automática do AirSim 1.8 fornecia rótulos 2D confiáveis em escala**.
*Porquê o sintético resolve:* (i) **rótulos pixel-perfeitos** — sabemos exatamente onde cada sprite
foi colado (erro <1 px); (ii) **rótulo 3D de distância** derivado da escala do sprite; (iii)
**independência total** da API bugada; (iv) **volume ilimitado** (milhares de frames de ~200
sprites + centenas de fundos, sem *crash*/timeout do simulador); (v) **distribuição controlada**
(nº de drones, escalas, oclusão, fundo).

### 4.4 Sprite extraction (`capture_sprites.py` → dataset_generation/synthetic/capture_sprites.py)
- **Método — limiar de brilho:** o `Quadrotor1` visual é **muito escuro** (estrutura preta) e o céu
  é **claro** → em escala de cinza, `mask = (gray < threshold)` (≈170–180) separa o drone do céu;
  morfologia *open/close* (kernel 3×3) limpa ruído; **componentes conexas** isolam o maior *blob*;
  a bbox são os limites do *blob*. Saída **RGBA** (RGB do drone + canal **alpha** = máscara binária).
- *Porquê brilho e não outras vias:* **(a) vs RGB-diff** — *diff* exigiria duas capturas
  (com/sem drone) sincronizadas pixel-a-pixel e é sensível a ruído/variação de sol; **(b) vs
  segmentação** — já provada inviável (§4.3.1). O contraste drone-escuro/céu-claro é robusto a
  variação de iluminação (<20 níveis de cinza no mesmo ambiente).
- **Biblioteca de sprites:** matriz de captura **12 yaws × 3 pitches × 3 rolls × 5 distâncias**
  (10/15/25/40/60 m) → **~196 sprites** salvos em PNG RGBA + metadados (`distance_m`, `yaw/pitch/
  roll_deg`, `sprite_size` em px). *Porquê salvar distância e tamanho:* permite derivar a distância
  do alvo na composição pela razão de escala (abaixo).

### 4.5 Synthetic composition (`compose_synthetic_dataset.py` → dataset_generation/synthetic/compose.py)
- **Fundos:** capturados **sem drones** (drones enterrados em `(−5000,−5000,2000)` p/ sumirem),
  por ambiente (NH/City/Coast/blocks), com variação de weather/hora — **~600 fundos** (≈200/ambiente).
- **Colagem:** redimensiona o sprite por `scale ∈ [0,4; 1,5]`, *alpha-blending*
  `out = α·sprite + (1−α)·fundo`, posição aleatória; **bbox = exatamente a região colada** (rótulo
  perfeito). Em formato YOLO normalizado (1 classe: `drone`).
- **Distância 3D (rótulo):** `dist = dist_original · (largura_original_px / largura_composta_px)`
  — fisicamente fundamentada no tamanho angular do drone (sprite capturado a distância conhecida).
- **Restrições (realismo):** oclusão entre drones **≤ 50%**; sprites pequenos (`largura < 80 px`)
  **restritos à metade superior** da imagem (céu) — *porquê:* drone distante/pequeno aparece alto no
  quadro, não rente ao chão. 1–5 drones/frame.
- **Volume final:** ~2500 frames (≈2400/600 treino/val em uma das gerações; o gerador aceita
  `--n_train/--n_val`). Saída: `yolo/{images,labels}/{train,val}` + `labels_3d/*.json` +
  `visualizations/`.

### 4.6 Phase 3 — Evaluation campaign (60 video sequences)
Conjunto de teste end-to-end, **gravado** (offline) para avaliação reprodutível:
- **60 sequências** (20 por ambiente: NH/City/Coast), ~75–110 frames, ~5 fps.
- Variação por sequência: **regime de distância** (near 18–32 m / mid 32–50 m / far 50–78 m),
  **iluminação** (h7–h18), **weather** (clear/fog_light/rain_light — **sem dust**, *porquê:* reduz
  demais a visibilidade e polui a avaliação), 2–3 drones, **seeds** distintos; *azimute* ±32°
  (manter no FOV).
- **Observador (ego) em MOVIMENTO**: oscilação 3D + yaw (`FlightThread`, raio ~4 m, ω≈0,3 rad/s,
  altitude ~−20 m NED) — *porquê:* valida o tracking em **plataforma móvel** (caso UAM real), não
  só com câmera estática.
- **Ground-truth:** pose global real de cada drone por frame (`VisQuad_*`), salva em `meta.json`
  junto com a pose do ego (`x_plat`).
- **Robustez operacional:** WSL caía ~de hora em hora; toda a campanha (`record_campaign.sh`,
  20 configs) é **resumível** e valida contra corrupção (conta apenas `.jpg` não-vazios, ≥45
  frames; re-grava corrompidas). Cache de nuvens CV in-range subamostradas (`build_pc_cache.py`)
  acelera o treino dos voxel de >5 min/época para ~10 s/época.

---

## 5. 2D detection (YOLO)

### 5.1 Arquitetura e treino
**YOLOv11s** (~10 M parâmetros). Treino Ultralytics (SGD + Nesterov, weight decay, cosine LR).
- **Fase 1 (urbano, dados reais):** `data=dataset_urban_merged`, epochs 100, batch 16, **imgsz 640**,
  patience 30 → **mAP50 0,731**, mAP50-95 0,403, **P 0,91, R 0,64** (recall = gargalo).
- **Fase 2 (sintético):** dataset composto por sprites, **imgsz 1280**, aug forte (mosaic, HSV,
  scale) → **mAP50 0,994**. Modelo de produção do pipeline final.

### 5.2 Analysis
Precisão alta (quando detecta, acerta) e **recall o fator limitante** do sistema em alvos
pequenos/distantes — confirmado em todas as fases seguintes como o gargalo de recall do pipeline
de 2 estágios.

---

## 6. 3D detection and localization — evolution of the architectures

> Convenção: todas as redes de nuvem normalizam por **subtração do centróide** e divisão por uma
> **escala** (modo `max` = distância máxima; `p95` = percentil 95; `fixed` = 8,0 m). Otimizador
> **Adam (lr 1e−3, weight decay 1e−4)** + **CosineAnnealingLR**. Aumentação: rotação em torno do
> eixo Y (vertical) em ±180° e ruído gaussiano σ=0,01 m. Balanceamento por **WeightedRandomSampler
> 50/50** na cabeça de classificação. Cabeças: `cls` (1, *is_drone*), `center` (3), `size` (3),
> e — quando aplicável — `seg` (por-ponto).

### 6.1 PointNet baseline (F1) — frustum classification
- **Conceito:** recorta pontos que projetam dentro da bbox 2D (cone/*frustum*); PointNet processa
  só esses pontos.
- **Dataset:** 9 136 amostras (6 136 pos / 3 000 neg; pos = frustums de GT casados por IoU,
  neg = *crops* aleatórios). (N=512 pts, XYZ normalizado).
- **Arquitetura (~0,35 M):** MLP1 `Conv1d(3→64→64)`; MLP2 `Conv1d(64→128→256→512)`; *global
  max-pool* → FC `512→256→128` (Dropout 0,3); cabeças cls/center/size.
- **Loss:** `BCE + 2,0·SmoothL1(center) + 1,0·SmoothL1(size)` (regressão só em positivos).
- **Resultado:** **acc 0,967** (classificar "tem drone" vs *crop* aleatório é fácil), **mas
  regressão ruim (center MSE 3,6** normalizado) — motiva PointPainting.

### 6.2 PointPainting (F1) — YOLO probability channel
- **Inspiração:** Vora et al. (CVPR 2020). Cada ponto recebe um **4º canal** = confiança do YOLO
  se projeta dentro da bbox (senão 0). Entrada `(N,4)=[x,y,z,prob]`. Multi-instância: 1 amostra por
  bbox.
- **Arquitetura (~1,0 M):** como o baseline mas `in=4` e MLP2 até 1024; FC mais profunda.
- **Achado-chave:** **center MSE 0,003** (1200× melhor que o baseline 3,6) — o canal `prob` age
  como *atenção suave* preservando contexto espacial (ajuda regressão). Porém **classificação
  fraca** (dataset 88/12 desbalanceado, *double-balancing* `sampler`+`pos_weight` → viés).

### 6.3 PointNet++ painted (v1–v7) — set abstraction + Focal Loss
- **Arquitetura (set abstraction hierárquica, FPS+ball-query implementados em PyTorch puro):**
  - **SA1:** 4096→512 pts, raio 0,2, nsample 32, MLP `[64,64,128]`;
  - **SA2:** 512→128 pts, raio 0,4, nsample 64, MLP `[128,128,256]`;
  - **SA3 (global):** MLP `[256,512,1024]`;
  - FC `1024→512→256→128` (Dropout 0,4/0,3); cabeças cls/center/size.
- **Loss:** Focal (`α`, `γ=2`) + `5,0·SmoothL1(center)` + `1,0·SmoothL1(size)`. epochs 40, batch 8.
- **Iterações de balanceamento (lição central):**

| Versão | Sampler | Focal α | Dataset | Resultado |
|---|---|---|---|---|
| v1 | 50/50 | 0,25 | 88/12 | F1 0,45 (viés negativo) |
| v2 | nenhum | 0,75 | 88/12 | F1 0,93 **trivial** (rec 1,0) |
| v3 | nenhum | 0,5 | 88/12 | F1 0,93 **trivial** |
| **v4** | 50/50 | 0,5 | **26/74** (rebalanceado) | **F1 0,68, rec 0,93, prec 0,58, MSE 0,003** |

> **Lição:** rebalancear a **loss** não basta para *imbalance* >80/20; é preciso **rebalancear o
> dataset** (sintetizar 2 negativos aleatórios por positivo → `dataset_painted_v2`, 16 568
> amostras, 26/74). Métrica robusta = **F1**, não accuracy.

### 6.4 Frustum PointNet++ (F2) — the consolidated two-stage pipeline
Mesma SA do §6.3, mas **`in=3`** (xyz, sem canal pintado). Para cada caixa 2D do YOLO recorta-se o
frustum (+25%), amostra-se **N=4096** (`MIN_FRUSTUM_PTS=8`), normaliza-se (modo `max` por padrão).
- **Loss/treino:** Focal (α=0,5, γ=2) + 5,0·SmoothL1(center) + SmoothL1(size); epochs 30, batch 8.
- **Painted vs Frustum:** *Painted* (nuvem inteira + prob) val-F1 0,50; **Frustum** (recorte por
  caixa) val-F1 **0,73 → escolhido**.

### 6.5 Depth band — heuristic foreground extraction (F2)
- **Problema diagnosticado:** a normalização usa a **distância máxima**; **poucos pontos de fundo
  distante esticam a escala** e **comprimem o drone a ~11%** do espaço normalizado → classificador
  tímido + centro 3D impreciso (**RMSE ~3 m**).
- **Solução heurística:** ancorar na profundidade do objeto mais próximo (`z0 = percentil 5 de z`)
  e descartar fundo além de uma banda (`z ≤ z0 + BAND_M`).
- **Efeito (ablação, 60 vídeos):** RMSE **3,0→1,2 m**; precisão **0,25→0,91**; **AMOTA 0,05→0,82**.
- **Limitação reconhecida:** frágil a oclusor frontal (arbusto/árvore na frente do drone) →
  motivou as alternativas *principled* da §6.7–6.8.

### 6.6 Robust p95 normalization (F4) — a negative result
Trocar `max` por **percentil 95** na escala de normalização **falha** (AMOTA 0,03; RMSE 2,88 m).
**Conclusão:** *clipar* a escala não basta — o **centróide** continua puxado pelo fundo. É preciso
**remover** o fundo, não reescalá-lo.

### 6.7 Frustum-ConvNet (F4) — depth slices
- **Arquitetura (~0,47 M):** mini-PointNet por ponto `Conv1d(3→64→128→128)`; **16 fatias de
  profundidade** (z dividido em bins), *pooling* por fatia via `scatter_reduce(amax)`; conv 1D ao
  longo das fatias `Conv1d(128→128→256→256)`; cabeça `256→256→128` → cls/center/size.
- **Resultado:** **iguala o depth-band** (AMOTA 0,81 vs 0,82; RMSE 1,15 vs 1,20 m) **sem
  heurística** — as fatias tratam o fundo nativamente.

### 6.8 T-Net / F-PointNet segmentation (F4) — the principled replacement for the depth band
- **Arquitetura (FrustumSegNet):** MLP1 `Conv1d(3→64→64)`; MLP2 `Conv1d(64→128→1024)`; **cabeça de
  segmentação por-ponto** `Conv1d(1088→256→128→1)`; *pooling* de foreground ponderado pela máscara
  → cabeça de caixa `1024→512→256` → cls/center/size.
- **Rótulo de foreground on-the-fly:** ponto é *fg* se cai **dentro da caixa GT** normalizada
  (`|p−c_gt| ≤ size_gt/2`). **Loss** = Focal(cls) + SmoothL1(center,size) + **BCE(seg, fg)**.
  epochs 20, batch 32, workers 12.
- **Resultado:** **SUPERA o depth-band** — RMSE **1,03 m** (vs 1,20), AMOTA **0,83**, prec 0,91. A
  rede **aprende** quais pontos são o drone (robusto a oclusor), sem corte manual → substituto
  ideal da "gambiarra".

### 6.9 PointPillars-lite (F4) — the single-stage voxel/BEV paradigm
- **Representação:** *pillariza* a nuvem em grade BEV (CV: X=direita [−40,40], Z=frente [2,90],
  célula **1,25 m** → grade **64×70**). Features hand-crafted por célula (4 canais): `log1p(count)`,
  altura média (down), `max(z)`. **Não usa YOLO** — detecta na cena inteira.
- **Arquitetura (PPNet):** CNN 2D `Conv2d(4→64→128→128)`; cabeças **heatmap** (centro-baseada,
  *focal heatmap loss* estilo CenterNet, alvo gaussiano raio 2) + **regressão** (4: dx,dy,z,logsize,
  SmoothL1 mascarada). epochs 15, batch 8. Split 80/20 por sessão.
- **Resultado:** perfil **oposto** ao frustum — **recall alto (0,77)** sem o gargalo do YOLO, mas
  **ruidoso** (muitos ID-switches, AMOTA 0,64). RMSE 1,12 m.

### 6.10 PointPillars-PFN (F5) — learned pillar features — **best localization**
- **Mudança:** substitui as features hand-crafted por um **mini-PointNet por pillar**. Por ponto:
  6 features aumentadas `[x, down, z, x−x_c, z−z_c, down]` → MLP `Linear(6→64→64)` → **scatter
  `amax`** por célula → pseudo-imagem BEV (64 canais) → CNN `Conv2d(64→128→128→128)` → heatmap+reg.
  `MAXP=12000` pts/frame. ~0,38 M parâmetros.
- **Resultado (média 3 envs):** **RMSE 0,78 m** (melhor de todo o estudo, supera o T-Net 1,03 m),
  **AMOTA 0,69**, **det-F1 0,84**, **det-P 0,92**, **det-R 0,80**, ID-switches ~666→**429**. O
  mini-PointNet extrai a **geometria fina do drone** que as estatísticas manuais descartam.

### 6.11 Dense 3D voxel (F5, SECOND-style) — ties with the PFN
- **Representação:** grade 3D (X[−40,40]×Y_down[−24,24]×Z[2,90], célula **2,0 m** → **40×24×44**),
  ocupância `log1p(count)`.
- **Arquitetura (Vox3DNet, ~0,47 M):** backbone **Conv3D** `1→16→32→32→64` com *stride (2,1,1)*
  (reduz só a altura), **colapsa a altura** (reshape) → CNN 2D `192→128→128` → heatmap+reg(3).
  epochs 15, batch 4.
- **Resultado:** **≈ PFN** (AMOTA 0,69; RMSE 0,85 m; F1 0,84) **mesmo com grade grossa (2,0 m)** — a
  cabeça de regressão recupera precisão **sub-voxel**. Confirma que o ganho vem de **representar
  melhor a estrutura local**, não da resolução. (Overfit cedo: melhor época = 2.)

### 6.12 Voxel variants with a **negative** result (F5)
- **Resolução fina (célula 0,7 m → 125×114):** **NÃO ajuda** (AMOTA 0,64→0,60; RMSE 1,12 m
  inalterado). O gargalo é a **esparsidade de pontos no drone**, não a quantização — mais células
  só geram mais picos espúrios (FP/ID-switches).
- **Temporal multi-frame (3 frames, compensação só do ego):** **PIORA** (AMOTA 0,64→0,52; RMSE
  1,12→1,58 m; AMOTP dobra). A compensação alinha o **fundo estático** mas **borra o drone em
  movimento** (não há compensação por-objeto — exigiria já ter o tracking, circular). Densificar
  só ajuda alvos estáticos.

---

## 7. Fusion

### 7.1 YOLO × PointNet fusion (score level, F2)
`score = w·conf_YOLO + (1−w)·prob_PN`, **w=0,5**. O YOLO confiante "puxa" o classificador 3D tímido
(positivos ~0,60 → decisão confiável). Resolve a baixa confiança nativa do classificador de nuvem.

### 7.2 Late frustum ⊕ voxel fusion (decision level, F4–F5)
União das detecções dos dois paradigmas + **casamento por proximidade 3D** (gate `GATE3D=3,0 m`) +
*score* por concordância (bônus `+0,25` se ambos concordam; desconto `×0,65` para detecção só-voxel)
+ **NMS 3D** (`2,0 m`). A ancoragem define o regime:

| Fusão | AMOTA | det-P | det-R | det-F1 | RMSE | Leitura |
|---|---|---|---|---|---|---|
| T-Net ⊕ PP base | 0,75 | 0,89 | 0,68 | 0,75 | 1,09 m | versão inicial |
| **T-Net ⊕ PFN (ancora T-Net)** | **0,76** | 0,91 | 0,67 | 0,74 | 1,03 m | melhor AMOTA com recall ≥0,65; eleva recall do T-Net 0,49→0,67 |
| **T-Net ⊕ PFN (ancora PFN)** | 0,71 | 0,91 | **0,81** | **0,84** | 0,79 m | preserva recall+localização do PFN; T-Net confirma |

> A fusão **não supera o melhor standalone em AMOTA** (T-Net puro 0,83, mas recall 0,49). Seu valor
> é **equilibrar** precisão, recall e localização numa única configuração — útil quando se quer
> recall aceitável **sem** sacrificar precisão.

---

## 8. Rastreamento (tracking)

### 8.1 Tracker final — SORT 2D (`sort_tracker.py`)
Associação por **IoU de bbox 2D** (do YOLO, limpa) — não pelo 3D ruidoso. Parâmetros:
`iou_thr=0,3`, `nms_iou=0,4` (NMS prévio colapsa caixas no mesmo objeto), `min_hits=3`
(confirmação), `max_age=8`, **`coast=4`** (track persiste por predição em falhas de 2–4 frames).
Modelo de velocidade por suavização exponencial (`v ← 0,5·v + 0,5·Δbbox`). Posição 3D global do
track: `ego + R(rpy)·FRD(center_cv)`.
> **Bug corrigido:** o `confirmed()` antes só retornava `time_since_update==0` (max_age não tinha
> efeito); com o *coast*, a cobertura track-on-GT subiu de 68%→81%.

### 8.2 Global 3D EKF tracker (F1, historical)
Filtro de Kalman estendido por alvo, estado `x=[x,y,z,vx,vy,vz,r]` (NED). Modelo de processo de
*random walk* em aceleração (`σ_a=0,5 m/s²`); **medida esférica** `[d,φ,θ,r]` com
`measurement_to_global` usando **Rᵀ**. Ruído de medida: `σ_φ=σ_θ=0,005 rad`, distância
`σ_d=max(0,5; 0,05·d)`. Associação por **distância de Mahalanobis** com **gate χ²=13,28** (99%,
4 GL) + Hungarian; ciclo de vida `min_hits=3`, `max_age=5`. Usado na fase inicial e como baseline
de ablação (frustum v4 + EKF) na §10.

---

## 9. Evaluation methodology

### 9.1 Metrics (the standard of the literature)
- **3D-MOT (nuScenes/AB3DMOT):** **AMOTA/AMOTP** — integral sobre **19 pontos de recall**;
  **independem do detector** (varrem o threshold) → medem a força do *tracker*. CLEAR-MOT:
  **MOTA@0,5**, **IDF1**, **ID-switches**, MT/ML (via `py-motmetrics`).
- **Detecção:** Precision, Recall, F1.
- **Localização 3D:** **RMSE/MAE do centróide** (por faixa de distância 0–30/30–50/50–90 m).
  Casamento GT↔track por **distância 3D**, gate ~3–4 m.
- *Nota:* MOTP/AMOTP via `motmetrics` retorna `nan` em sequências com matches esparsos →
  reporta-se o **erro 3D RMSE/MAE** (sempre definido) como métrica de localização.

### 9.2 Dump protocol (efficiency + reproducibility)
**1 passada pesada** (YOLO + detector 3D) por vídeo grava um *dump* de detecções
(`bbox, yolo, pn, gpos, d` + GT por frame); as **métricas são recalculadas barato** sobre o dump
(varrendo o threshold para AMOTA, rodando o SORT + `motmetrics`). **Cache de caixas YOLO**
reutilizado entre experimentos de frustum (~10× mais rápido). Teste **offline** (grava cru →
processa sem pressão de tempo).

### 9.3 Methodological caveat on rigour (to be addressed in the dissertation)
Os detectores **voxel** (PointPillars/PFN/Vox3D/temporal/fine) são treinados sobre as próprias
sessões da campanha com *split* **80/20 por sessão** (`val_s = sessões[::5]`), mas o *dump* de
avaliação roda sobre **todas as 60** → as métricas voxel incluem sessões de treino (**otimista**).
Os detectores **frustum** treinam num *dataset* de amostras de frustum dedicado. Para uma
comparação final rigorosa, recomenda-se reportar os voxel **apenas no split de validação** (ou
re-treinar com *hold-out* por ambiente). Os números aqui servem de comparação relativa entre
variantes voxel (mesmo protocolo), mas essa assimetria deve ser declarada.

---

## 10. Resultados consolidados

### 10.1 Reference configuration of the two-stage pipeline (best, 60 sequences)
| Métrica | NH | City | Coast | Global |
|---|---|---|---|---|
| AMOTA (↑) | 0,78±0,20 | 0,80±0,34 | **0,88±0,10** | — |
| MOTA@0,5 | 0,28 | 0,57 | 0,51 | — |
| IDF1 | 0,27 | 0,59 | 0,46 | — |
| det-Precision | 0,89 | 0,84 | **1,00** | — |
| det-Recall | 0,32 | 0,59 | 0,53 | — |
| det-F1 | 0,44 | 0,67 | 0,66 | — |
| **Erro 3D RMSE** (↓) | 1,45 m | 1,15 m | **1,04 m** | **1,20 m** (MAE 1,07; n=5455) |

Estável por distância: 0–30 m=1,22 m · 30–50 m=1,18 m · 50–90 m=1,26 m.

### 10.2 Ablation — depth band + SORT vs. baseline (frustum v4 max-norm + EKF)
| Métrica | Baseline | **Proposto** | Ganho |
|---|---|---|---|
| AMOTA | 0,05 | **0,82** | ~16× |
| MOTA@0,5 | −0,23 | **0,45** | +0,68 |
| IDF1 | 0,11 | **0,44** | 4× |
| det-F1 | 0,15 | **0,59** | 4× |
| det-Precision | 0,25 | **0,91** | 3,6× |
| Erro 3D RMSE | 2,99 m | **1,20 m** | −60% |

### 10.3 Master table — every method (60 sequences, mean over 3 environments)
| # | Método | Paradigma | AMOTA↑ | MOTA@0,5↑ | det-P↑ | det-R↑ | det-F1↑ | RMSE 3D↓ | ID-sw↓ |
|---|--------|-----------|--------|-----------|--------|--------|---------|----------|--------|
| 1 | Baseline (frustum, max-norm) | frustum 2-est. | 0,05 | −0,23 | 0,25 | baixo | 0,15 | 2,99 m | ~13 |
| 2 | Norm. robusta (p95) | frustum 2-est. | 0,03 | — | — | — | 0,08 | 2,88 m | — |
| 3 | Depth-band (heurística) | frustum 2-est. | 0,82 | 0,45 | 0,91 | 0,48 | 0,59 | 1,20 m | ~50 |
| 4 | Frustum-ConvNet | frustum 2-est. | 0,81 | 0,41 | 0,84 | 0,44 | 0,55 | 1,15 m | ~48 |
| 5 | **Segmentação T-Net (F-PointNet)** | frustum 2-est. | **0,83** | 0,46 | 0,91 | 0,49 | 0,60 | 1,03 m | ~62 |
| 6 | PointPillars base (hand-crafted) | voxel 1-est. | 0,64 | 0,59 | 0,87 | 0,77 | 0,80 | 1,12 m | ~666 |
| 7 | PointPillars resolução fina (0,7 m) | voxel 1-est. | 0,60 | 0,55 | 0,85 | 0,75 | 0,77 | 1,12 m | 701 |
| 8 | PointPillars temporal (3 frames) | voxel 1-est. | 0,52 | 0,47 | 0,82 | 0,72 | 0,74 | 1,58 m | 881 |
| 9 | Voxel 3D denso (Conv3D) | voxel 1-est. | 0,69 | 0,69 | 0,92 | 0,81 | **0,84** | 0,85 m | 469 |
| 10 | **PointPillars-PFN (aprendidas)** | voxel 1-est. | 0,69 | 0,69 | **0,92** | 0,80 | **0,84** | **0,78 m** | **429** |
| 11 | Fusão tardia (T-Net ⊕ PP base) | híbrido | 0,75 | — | 0,89 | 0,68 | 0,75 | 1,09 m | — |
| 12 | Fusão T-Net ⊕ PFN (ancora T-Net) | híbrido | 0,76 | 0,57 | 0,91 | 0,67 | 0,74 | 1,03 m | 443 |
| 13 | Fusão T-Net ⊕ PFN (ancora PFN) | híbrido | 0,71 | 0,65 | 0,91 | **0,81** | **0,84** | 0,79 m | ~480 |

### 10.4 Leaders per metric
- **Localização 3D (RMSE):** PFN 0,78 m › Voxel-3D 0,85 › T-Net 1,03 › Fusão 1,09 › PP base 1,12.
- **Recall:** voxel (0,77–0,81) » todo frustum ≤0,49 (teto do YOLO).
- **F1 de detecção:** PFN/Voxel-3D 0,84 › PP base 0,80 › Fusão 0,75 › T-Net 0,60.
- **AMOTA:** T-Net 0,83 › depth-band 0,82 › F-ConvNet 0,81 › Fusão 0,76 › PFN/Voxel-3D 0,69.
- **ID-switches:** frustum « voxel (frustum ~50; PFN 429, já bem melhor que PP base ~666).

---

## 11. Scientific findings and lessons

1. **A normalização por distância máxima é o vilão da localização frustum:** poucos pontos de
   fundo comprimem o drone no espaço normalizado. Reescalar (p95) **não resolve** — é preciso
   **remover** o fundo (depth-band) ou **segmentá-lo** (T-Net).
2. **Heurística → componente aprendido:** a segmentação T-Net **substitui** o depth-band e dá a
   melhor localização do paradigma frustum (1,03 m), robusta a oclusor.
3. **No voxel, melhor representação > mais resolução > mais frames:** features de pillar
   **aprendidas** (PFN) ou conv 3D (Vox3D) levam o RMSE a ~0,8 m e AMOTA a 0,69; resolução fina e
   temporal ingênuo **pioram** (esparsidade e *smearing* de alvo móvel).
4. **Trade-off de paradigmas:** frustum (2-est.) = **precisão e AMOTA altas**, recall preso pelo
   YOLO (~0,49); voxel (1-est.) = **recall alto (0,80)** sem o gargalo, historicamente mais
   ruidoso — mas o **PFN fechou** esse gap de precisão (0,92) e localização (0,78 m).
5. **Gargalo do sistema = recall 2D do YOLO** em alvos distantes/fog/ocluídos — a alavanca de maior
   impacto é uma campanha de dados do detector com fundo difícil; alternativamente, o paradigma
   voxel contorna o gargalo (detecta na cena inteira).
6. **AMOTA alto (0,78–0,88) vs MOTA@0,5 modesto:** coerente — o MOTA num ponto é puxado pelo recall
   do detector; o AMOTA integra sobre recall e mostra um **tracker forte** (competitivo com carros
   no nuScenes ~0,6–0,7).
7. **Engenharia de simulação é parte do método:** `simPause`, re-teleporte por frame, *shims* de
   RPC, filtro de oclusão por depth, validação anti-corrupção e *caching* foram **pré-condições**
   para dados e avaliação confiáveis.

---

## 12. Known limitations
- **Recall 2D do YOLO** (~0,49 em média na campanha) limita o pipeline frustum.
- **Caveat de avaliação dos voxel** (treino/teste sobreposto por sessão — §9.3) a ser endereçado
  com *hold-out* rigoroso.
- **Classe única** (`drone`); não distingue identidades nativamente.
- **Sim-to-real gap** não avaliado (treino sintético).
- Regressão de **tamanho 3D** pouco informativa (drones têm tamanho quase constante).
- WSL instável (quedas ~horárias) — mitigado por resumibilidade/cache, mas custou tempo.

---

## 13. Artefatos e reprodutibilidade

### 13.1 Modelos treinados (`runs/`)
- **YOLO:** `runs/drone/drone_synth_v1` (sintético, mAP50 0,994), `runs/drone/drone_urban_v1`
  (urbano, mAP50 0,73).
- **Frustum/painted PointNet++:** `runs/pointnet2_frustum/{v1..v5, v5band, p95, convnet, segnet}`,
  `runs/pointnet2_painted/{v1..v7, v4_balanced}`, `runs/pointnet/drone_detector` (baseline),
  `runs/pointnet_painted/...`.
- **Voxel:** `runs/{pointpillars, pointpillars_pfn, voxel3d, pointpillars_fine,
  pointpillars_temporal}/best.pt`.

### 13.2 Detection dumps (cheap recomputation of the metrics)
`det_dumps_{base, best, p95, convnet, segnet, pp, ppfine, pptemp, pfn, vox3d, painted, fusion,
fusion_pfn, fusion_pfnA}`.

### 13.3 Core code
- **Coleta/geometria:** `generate_dataset_urban.py`, `record_session.py`, `record_campaign.sh`,
  `inference_pipeline.py`, `ekf_3d_global.py`, `build_pc_cache.py`.
- **Datasets derivados:** `build_frustum_pn2_dataset.py` (depth-band, NORM_MODE),
  `build_painted_dataset.py`, `compose_synthetic_dataset.py`, `capture_sprites.py`.
- **Treino:** `train_yolo11s_drone.py`, `train_pointnet2_frustum.py`, `train_pointnet2_painted.py`,
  `train_segnet.py`; modelos voxel `pointpillars_lite.py` (flags `VOXEL_CELL`/`VOXEL_TK`),
  `pointpillars_pfn.py`, `voxel3d.py`; arquiteturas `frustum_segnet.py`, `frustum_convnet.py`.
- **Inferência/avaliação:** `detector_frustum_pn2.py` (NORM_MODE, band_m, arch),
  `dump_detections.py`, `pp_dump.py`, `metrics_from_dump.py`, `fuse_dumps.py`,
  `sort_tracker.py`, `render_results.py`, `render_bev_*`.
- **Documentos:** `DOCUMENTACAO_TECNICA.md` (Fase 1), `METODOLOGIA.md`, `RESULTS_artigo.md`,
  `RESULTS_scientific.md`, `RESULTS_abordagens.md`, `RESULTS_voxel.md`, `RESULTS_master.md`,
  e **este** `COMPILADO_DISSERTACAO.md`.

### 13.4 Constantes-chave (verbatim)
```
Câmera: 1280×720, FOV 90°, FX=FY=640, CX=640, CY=360, depth∈[0.1,250] m
Câmera offset body=(0.35,0,−0.5) m, pitch −15°
Drone (VisQuad 4×): extents 3.01×3.93×2.79 m; meias-extensões [2.077,2.713,1.925] m (×1.38/2)
Voxel BEV: X∈[−40,40], Z∈[2,90], célula 1.25 m → 64×70 (fine 0.7 m → 125×114)
Voxel 3D: +Y_down∈[−24,24], célula 2.0 m → 40×24×44
Frustum: N=4096 pts, MIN_FRUSTUM_PTS=8, expand 25%, IoU GT 0.3
Fusão score w=0.5; fusão tardia GATE3D=3.0, NMS3D=2.0, bônus +0.25, desconto ×0.65
SORT: iou_thr=0.3, nms_iou=0.4, min_hits=3, max_age=8, coast=4
EKF: estado 7D, χ²-gate=13.28 (4 GL), σ_a=0.5, σ_φ=σ_θ=0.005, σ_d=max(0.5,0.05·d)
Treino nuvem: Adam(1e−3, wd 1e−4) + CosineAnnealingLR; aug rot-Y ±180° + ruído σ0.01; sampler 50/50
```

---

## 14. References
1. Qi et al. (2017) *PointNet*. CVPR.
2. Qi et al. (2017) *PointNet++*. NeurIPS.
3. Qi et al. (2018) *Frustum PointNets*. CVPR.
4. Vora et al. (2020) *PointPainting*. CVPR.
5. Lin et al. (2017) *Focal Loss*. ICCV.
6. Lang et al. (2019) *PointPillars*. CVPR.
7. Yan et al. (2018) *SECOND: Sparsely Embedded Convolutional Detection*. Sensors.
8. Zhou & Tuzel (2018) *VoxelNet*. CVPR.
9. Zhou et al. (2019) *CenterNet / Objects as Points*. (heatmap center-based)
10. Bewley et al. (2016) *SORT*. ICIP.
11. Weng et al. (2020) *AB3DMOT* (AMOTA/AMOTP). IROS.
12. Caesar et al. (2020) *nuScenes*. CVPR.
13. Ultralytics (2024) *YOLO11*.
14. Microsoft *AirSim*; Cosys-Lab *Cosys-AirSim*.

---

*Fim do compilado. Para detalhes de implementação, ver os comentários inline nos scripts
referenciados na §13.3 e os documentos de resultados na §13.3.*
