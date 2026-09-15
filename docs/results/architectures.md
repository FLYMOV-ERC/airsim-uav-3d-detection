# Comparison of approaches — replacing the depth-band heuristic

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status: superseded, and the difference is untraced.** Same caveat as
> docs/results/master.md: these figures do not match Chapter 7's Table 7.4 (T-Net
> RMSE 1.03 m here against 1.05 m there, AMOTA 0.83 against 0.81), the reason has
> not been established, and it is not the held-out recomputation of Section 7.4.3
> — that concerns only the voxel rows, and the frustum rows differ here too.

60 vídeos (NH+City+Coast, 20 cada), ego em movimento, ground-truth 3D. Métricas média entre os 3 ambientes.

| Método | Tipo | AMOTA↑ | MOTA@0.5↑ | det-P↑ | det-R↑ | det-F1↑ | RMSE 3D↓ | ID-sw↓ |
|--------|------|--------|-----------|--------|--------|---------|----------|--------|
| Baseline (frustum max-norm) | frustum 2-estágios | 0.05 | −0.23 | 0.25 | baixo | 0.15 | 2.99 m | ~13 |
| **#3a** Norm. robusta (p95) | frustum 2-est. | 0.03 | — | — | — | 0.08 | 2.88 m | — |
| **depth-band** (gambiarra) | frustum 2-est. | 0.82 | 0.45 | 0.91 | 0.48 | 0.59 | 1.20 m | ~50 |
| **#4** Frustum-ConvNet | frustum 2-est. | 0.81 | 0.41 | 0.84 | 0.44 | 0.55 | 1.15 m | ~48 |
| **#1** Segmentação T-Net | frustum 2-est. | **0.83** | 0.46 | 0.91 | 0.49 | 0.60 | **1.03 m** | ~62 |
| **#2** PointPillars (voxel) | cena inteira 1-estágio | 0.64 | **0.59** | 0.87 | **0.77** | **0.80** | 1.12 m | alto (~222/env) |

## Findings
**Sobre a crítica à gambiarra (depth-band "pega o 1º objeto"):**
- **#3 Normalização robusta (p95) FALHA** (AMOTA 0.03, RMSE 2.9m). Clipar a escala não basta:
  o **centróide** ainda é puxado pelo fundo → centro 3D errado. Tem que **remover** o fundo, não re-escalar.
- **#4 Frustum-ConvNet IGUALA o depth-band** (AMOTA 0.81 vs 0.82, RMSE 1.15 vs 1.20m) **sem heurística** —
  as fatias de profundidade tratam o fundo nativamente.
- **#1 Segmentação T-Net SUPERA** o depth-band (RMSE **1.03m** vs 1.20m, AMOTA 0.83) — é o substituto
  principled ideal: a rede **aprende** quais pontos são o drone (robusto a oclusor), sem corte manual.

→ **Conclusão:** a gambiarra pode (e deve, p/ o artigo) ser trocada. Recomendado: **segmentação T-Net**
(melhor) ou **Frustum-ConvNet** (iguala, mais simples). Normalização robusta não resolve.

**Sobre o paradigma alternativo (#2 PointPillars):**
- Perfil **oposto**: **recall alto (0.77 vs 0.49)** e melhor MOTA@0.5 (0.59) porque **NÃO depende do YOLO**
  (detecta na cena inteira) → remove o gargalo de recall 2D que limitava o pipeline frustum.
- Mas **mais ruidoso**: muito mais ID-switches e FP → AMOTA menor (0.64). 3D RMSE comparável (1.12m).
- **Trade-off central p/ o artigo:** frustum (2-estágios) = **alta precisão, recall limitado pelo YOLO**;
  voxel (1-estágio) = **alto recall, mais ruído**. Caminho futuro: fundir os dois.

## Recommendation for the paper
- **Método principal:** YOLO→Frustum **com segmentação T-Net** (substitui a gambiarra; melhor 3D: 1.03m).
- **Ablações:** baseline, p95 (mostra que normalização não basta), depth-band, F-ConvNet.
- **Baseline alternativo:** PointPillars (mostra o trade-off precisão↔recall e motiva fusão futura).
