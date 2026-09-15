# Methodology — 3D drone detection and tracking for UAM (AirSim)

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** The most compact accurate narrative of the method. Its results
> table is **stale** relative to the dissertation (it gives AMOTA 0.83/0.82 and
> RMSE 1.03/0.78 m, where Chapter 7 gives 0.81 and 1.05/0.87 m). This file
> predates the dissertation's final numbers; where the two disagree, the
> dissertation is the reported result. **Why they differ has not been traced,**
> and it is not the held-out recomputation of Section 7.4.3 — that concerns only
> the voxel rows, and the frustum rows differ here too.

Documento de referência para o artigo. Reúne todas as técnicas implementadas e testadas.

---
## 1. Data generation

### 1.1 2D detector (YOLO) — synthetic dataset
- **Composição sintética**: 600 backgrounds (200/ambiente) + 196 sprites de drone extraídos,
  compostos em 2500 frames. Extração de sprite por **limiar de brilho** (drone escuro vs céu claro)
  após tentativas falhas com diff RGB (variação de sol) e segmentação (bug do MS AirSim 1.8).
- **YOLOv11s @ imgsz=1280**, treino 80 épocas, aug forte (mosaic, hsv, scale). **mAP50 = 0.994**.

### 1.2 3D dataset (PointNet) — AirSim recording
- **Mesh visual ampliado (VisQuad)**: cada multirotor recebe um `Quadrotor1` em **escala 4×**
  (extents reais medidos: 3.01×3.93×2.79 m) anexado e sincronizado a 30 Hz (`DroneVisualizer`),
  pois o drone nativo é pequeno demais para o YOLO/depth.
- **3 ambientes**: AirSimNH, City, Coastline (trocados via reabertura do mapa).
- Captura pausada (`simPause`) para imagem+depth+poses consistentes; LiDAR + DepthPlanar.
- Nuvem de pontos via `depth_to_pointcloud` no **frame CV da câmera** (x=right, y=down, z=fwd).

### 1.3 Critical 3D bounding-box correction (the central finding)
- **Bug do AirSim**: `det.relative_pose` tem offset de pitch (~5 m em "down") e `det.box3D`
  subestima o tamanho. **Solução**: calcular a bbox 3D **manualmente** a partir da pose global
  real (`simGetObjectPose` + origins do `settings.json`) + extents fixos do Quadrotor 4× +
  transformação body→câmera (offset (0.35,0,−0.5) + pitch −15°).
- **Bug de origem**: `simGetVehiclePose`/`simSetObjectPose` usam frames diferentes
  (relativo-à-origem vs global) → corrigido no `DroneVisualizer`.
- **Bug de projeção**: `project_global_to_pixel` usava `R` em vez de `Rᵀ` (mundo→câmera) →
  invertia o eixo vertical. Corrigido.
- Filtros de qualidade: ≥5 pontos dentro da bbox; rejeitar corners atrás da câmera.

---
## 2. 3D detection/tracking pipeline (two stages)

### 2.1 YOLO → Frustum
Para cada caixa 2D do YOLO, recorta-se o **frustum** (pontos da nuvem dentro do cone 2D, +25%).

### 2.2 PointNet++ on the frustum (Frustum-PointNet)
- Arquitetura SSG (SA `[64,64,128]`/`[128,128,256]`/`[256,512,1024]`), cabeças: cls (is_drone) +
  center(3) + size(3). Treino com **Focal Loss** + Cosine LR + weight decay.
- **Painted vs Frustum** (duas abordagens de fusão YOLO↔PointNet):
  - *Painted*: pinta a nuvem inteira com a prob do YOLO (4º canal). val_f1 = 0.50.
  - *Frustum*: recorta por caixa (3 canais xyz). val_f1 = 0.73 → **escolhido**.

### 2.3 Depth band (heuristic foreground extraction)
- Problema: a normalização do PointNet usa a **distância máxima**; pouquíssimos pontos de fundo
  distante esticam a escala e **comprimem o drone** (~11% do espaço normalizado) → classificador
  tímido e centro 3D impreciso (RMSE ~3 m).
- Solução (heurística): ancorar na profundidade do objeto mais próximo e descartar o fundo além
  de uma banda. **Efeito grande**: RMSE 3.0→1.2 m, precisão 0.25→0.91, AMOTA 0.05→0.82.
- *Limitação reconhecida*: frágil a oclusor frontal (motivou as alternativas principled da Seção 4).

### 2.4 YOLO × PointNet fusion
`score = w·conf_YOLO + (1−w)·prob_PN` (w=0.5). O YOLO confiante "puxa" o PN tímido → robusto.
Resolve a baixa confiança do classificador 3D (positivos ~0.60 → decisão confiável).

### 2.5 Tracking — SORT 2D
- Associação por **IoU de bbox 2D** (limpa, do YOLO) — não pelo 3D ruidoso.
- **NMS** prévio (colapsa caixas no mesmo objeto), `min_hits` (confirmação), `max_age` + **coast**
  (track persiste por predição em falhas de 2-3 frames → +13 pts de cobertura).
- Posição 3D global do track: `ego + R(rpy)·FRD(center_cv)`.

---
## 3. Evaluation

### 3.1 Campanha experimental
- **60 sequências** (20/ambiente × NH/City/Coast), ~75-110 frames, ~5 fps.
- Variação: regime de distância (near/mid/far), iluminação (h7–h18), weather (clear/fog/rain).
- **Ego (observador) em MOVIMENTO** (oscilação 3D + yaw) — valida tracking em plataforma móvel.
- Posições aleatórias por seed; validação anti-corrupção (rejeita frames vazios do WSL).
- Ground-truth: pose global real de cada drone por frame.

### 3.2 Metrics (the standard of the literature)
- **3D-MOT (nuScenes/AB3DMOT)**: AMOTA/AMOTP (integral sobre recall, independe do detector),
  MOTA, MOTP, IDF1, ID-switches, MT/ML.
- **Detecção**: Precision, Recall, F1.
- **Localização 3D**: RMSE/MAE do centróide, por faixa de distância.
- Pipeline de dump: 1 passada pesada (YOLO+detector) por vídeo → métricas recalculadas barato
  (varrendo o threshold para AMOTA). Cache de caixas YOLO reutilizado entre experimentos.

### 3.3 Inference optimizations
- Teste **offline** (grava cru → processa sem pressão de tempo → vídeo em tempo real).
- Cache de caixas YOLO (idênticas entre variantes do frustum) → dumps subsequentes ~10× mais rápidos.

---
## 4. Approaches compared (replacing the depth-band heuristic)
Resultados em `RESULTS_abordagens.md` (média 3 ambientes, 60 vídeos):

| Método | Tipo | AMOTA | det-P | det-R | det-F1 | RMSE 3D |
|--------|------|-------|-------|-------|--------|---------|
| Baseline frustum (max-norm) | 2-est. | 0.05 | 0.25 | — | 0.15 | 2.99 m |
| Normalização robusta (p95) | 2-est. | 0.03 | — | — | 0.08 | 2.88 m |
| Depth-band (heurística) | 2-est. | 0.82 | 0.91 | 0.48 | 0.59 | 1.20 m |
| **Frustum-ConvNet** | 2-est. | 0.81 | 0.84 | 0.44 | 0.55 | 1.15 m |
| **Segmentação T-Net (F-PointNet)** | 2-est. | **0.83** | 0.91 | 0.49 | 0.60 | **1.03 m** |
| **PointPillars (voxel)** | 1-est. cena | 0.64 | 0.87 | **0.77** | **0.80** | 1.12 m |
| **Fusão tardia (T-Net ⊕ PointPillars)** | híbrido | 0.75 | **0.89** | **0.68** | 0.75 | 1.09 m |

**Conclusões:** (a) normalização robusta não basta — é preciso *remover* o fundo; (b) segmentação
aprendida (T-Net) substitui a gambiarra e dá a melhor localização (1.03 m); (c) voxel (PointPillars)
tem perfil oposto (recall alto, sem gargalo do YOLO, porém ruidoso); (d) **fusão tardia** combina
precisão (0.89) + recall (0.68).

---
## 5. Modelos e artefatos
- `runs/drone/drone_synth_v1` (YOLO), `runs/pointnet2_frustum/{v4,v5band,convnet,segnet,p95}`,
  `runs/pointnet2_painted/{v6,v7}`, `runs/pointpillars`.
- Dumps de detecção: `det_dumps_{base,best,p95,convnet,segnet,pp,fusion}` (recálculo barato de métricas).
- Vídeos de resultado: 1ª pessoa + BEV ego-cêntrico + BEV global (ego móvel).

---
## 6. Additional voxel experiments — what moves the needle in the single-stage paradigm
Quatro modificações sobre o PointPillars-lite base (resultados completos em `RESULTS_voxel.md`,
60 vídeos). Cache de nuvens CV in-range subamostradas (`build_pc_cache.py`) acelera o treino.

| Variação (voxel) | AMOTA | det-F1 | RMSE 3D | vs base |
|------------------|-------|--------|---------|---------|
| Base PointPillars (features hand-crafted) | 0.64 | 0.80 | 1.12 m | — |
| **V1 — PFN (features de pillar aprendidas)** | **0.69** | **0.84** | **0.78 m** | **melhor** |
| **V4 — voxel 3D denso (Conv3D, SECOND-style)** | 0.69 | 0.84 | 0.85 m | ≈V1 |
| V3 — resolução fina (célula 0.7 m) | 0.60 | 0.77 | 1.12 m | pior |
| V2 — temporal multi-frame (3 frames, ego-comp.) | 0.52 | 0.74 | 1.58 m | pior |

**Conclusões:** (a) **PFN (features aprendidas)** é o maior ganho — mini-PointNet por pillar leva o
RMSE a **0.78 m** (melhor localizador de todo o estudo, supera o frustum T-Net) mantendo recall 0.80;
(b) **voxel 3D** empata com o PFN mesmo com grade grossa (regressão recupera sub-voxel) → o ganho
vem de *representar melhor a estrutura local*, não da resolução; (c) **resolução fina não ajuda** —
gargalo é a esparsidade de pontos, não a quantização; (d) **temporal ingênuo piora** — compensação
só-do-ego *borra o alvo em movimento*. Recomendado p/ o artigo: **PointPillars-PFN** como detector
voxel forte; reportar V2/V3 como resultados negativos que delimitam o espaço de design.

### Artifacts of this section
`runs/{pointpillars_pfn,voxel3d,pointpillars_fine,pointpillars_temporal}/best.pt`;
dumps `det_dumps_{pfn,vox3d,ppfine,pptemp}`; código `pointpillars_pfn.py`, `voxel3d.py`,
`build_pc_cache.py` + flags `VOXEL_CELL`/`VOXEL_TK` em `pointpillars_lite.py`.
