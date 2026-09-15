# Results — 3D drone detection and tracking (AirSim, 60 sequences)

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status: superseded.** A paper-oriented cut of the results, overlapping
> docs/results/master.md, and carrying the same numbers — which do not match
> Chapter 7's. See the status note of docs/results/master.md: the difference has
> not been traced.


## Setup experimental
- **60 sequências** (vídeos), 20 por ambiente: **AirSimNH, City, Coastline**.
- Variação por sequência: regime de distância (near 18-32m / mid 32-50m / far 50-78m),
  iluminação (h7–h18), weather (clear / fog_light / rain_light), 2–3 drones, seeds distintos.
- **Ego (observador) em MOVIMENTO** (oscilação 3D + yaw) — valida o tracking em plataforma móvel.
- Ground-truth: pose global real de cada drone por frame (~75–110 frames/vídeo, ~5 fps).
- Pipeline: YOLOv11 (synth, mAP50=0.99) → Frustum PointNet++ (depth-band) → fusão YOLO×PN → SORT (IoU 2D).

## Metrics (3D-MOT literature standard: nuScenes/AB3DMOT + detection + localization)
Média ± desvio entre as 20 sequências de cada ambiente.

| Métrica | NH | City | Coast |
|---|---|---|---|
| **AMOTA** (↑, integral sobre recall) | 0.78 ± 0.20 | 0.80 ± 0.34 | **0.88 ± 0.10** |
| MOTA@0.5 (ponto de operação) | 0.28 | 0.57 | 0.51 |
| IDF1 (↑) | 0.27 | 0.59 | 0.46 |
| Detecção — Precision (↑) | 0.89 | 0.84 | **1.00** |
| Detecção — Recall (↑) | 0.32 | 0.59 | 0.53 |
| Detecção — F1 | 0.44 | 0.67 | 0.66 |
| ID-switches (total) | 68 | 30 | 66 |
| **Erro 3D RMSE** (↓) | 1.45 m | 1.15 m | **1.04 m** |

**Localização 3D global (3 envs):** RMSE = **1.20 m**, MAE = 1.07 m (n=5455 matches).
Estável por faixa de distância: 0-30m=1.22m · 30-50m=1.18m · 50-90m=1.26m.

## Interpretation — do the results make sense?
1. **AMOTA alto (0.78–0.88)** vs **MOTA@0.5 modesto (0.28–0.57)**: coerente e é o ponto-chave.
   O MOTA num único ponto é puxado pelo recall do detector; o AMOTA integra sobre recall
   (independe do detector, padrão nuScenes) e mostra que **o tracker é forte** quando a
   detecção entrega o alvo. AMOTA 0.8+ é competitivo (carros no nuScenes ~0.6–0.7).
2. **Precision altíssima (0.84–1.00; Coast=1.00)**: o frustum+fusão+depth-band **quase não gera
   falso positivo** — o PointNet confirma só drone real. Faz sentido (Coast tem fundo de água/céu).
3. **Recall moderado (0.32–0.59)**: o gargalo é o **YOLO 2D** em alvos distantes/fog/ocluídos
   (confirmado em toda a investigação). NH pior (0.32) por fundo de árvores denso; Coast/City
   melhores (céu/estruturas). O caso `nh_far_fog` zera (fog mata alvo distante) — honesto.
4. **Erro 3D ~1.2m estável com a distância**: o depth-band entrega localização precisa mesmo a
   50-90m (vs ~3.2m do baseline sem depth-band). É o resultado mais sólido p/ UAM.

## Ablation — baseline (frustum v4, no depth band, EKF) vs. proposed (depth band + SORT)
Média entre os 3 ambientes (60 vídeos).

| Métrica | Baseline | **Proposto** | Ganho |
|---|---|---|---|
| AMOTA | 0.05 | **0.82** | ~16× |
| MOTA@0.5 | −0.23 | **0.45** | +0.68 |
| IDF1 | 0.11 | **0.44** | 4× |
| Detecção F1 | 0.15 | **0.59** | 4× |
| Detecção Precision | 0.25 | **0.91** | 3.6× |
| Erro 3D RMSE (global) | 2.99 m | **1.20 m** | −60% |

O **depth-band** corta o fundo distante do frustum → (a) localização 3D de 3.0m→1.2m,
(b) PointNet confiante → precision 0.25→0.91. O **SORT (IoU 2D) + coast** dá tracks estáveis.
O salto no AMOTA (0.05→0.82) resume o efeito combinado.

## Conclusion
Sistema **localiza drones com ~1.2m de erro 3D e altíssima precisão**, com tracking forte
(AMOTA 0.78–0.88) sob observador móvel e 3 ambientes. **Limite atual = recall 2D do YOLO**
em alvos distantes/baixa-visibilidade — a alavanca de maior impacto é uma campanha de dados
do detector com fundo difícil.

## Notas
- MOTP/AMOTP via py-motmetrics deu nan em ambientes com matches esparsos; reportamos o **erro 3D
  RMSE/MAE** (sempre definido) como métrica de localização.
- Métricas computadas via dump de detecções (1 passada YOLO+frustum) + SORT, varrendo o threshold
  para AMOTA (19 pontos de recall).
