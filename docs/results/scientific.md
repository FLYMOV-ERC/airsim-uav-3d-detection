# Scientific evaluation — 3D drone detection/tracking pipeline (AirSim)

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status: superseded.** Output of the earlier `eval_scientific.py` protocol
> (now `archive/early_evaluation/eval_scientific.py`), which predates the dump
> protocol that produces every number in the dissertation. Kept for provenance.


## Metodologia
- **8 sequências** gravadas em AirSimNH com variação controlada:
  regime de distância (near 18-32m / mid 32-50m / far 50-78m), iluminação
  (horas 9/11/13/17), weather (clear/fog_light/rain_light), 2-3 drones,
  posições aleatórias por seed. ~90 frames/seq @ ~5 fps. Ground truth = pose
  global real dos drones (VisQuad) por frame.
- **Métricas padrão**: MOT via py-motmetrics (MOTA, IDF1, ID-switches),
  detecção (Precision/Recall/F1), e erro de localização 3D global (RMSE/MAE
  por faixa de distância). Casamento GT↔track por distância 3D, gate 4 m.
- Agregação: média ± desvio entre sequências.

## Results (mean ± standard deviation, n=8)
| Métrica            | Baseline (frustum v4) | Best (depth-band + coast + conf0.10) |
|--------------------|-----------------------|--------------------------------------|
| MOTA               | -0.59 ± 0.30          | **+0.08 ± 0.25**                     |
| IDF1               | 0.08 ± 0.09           | **0.35 ± 0.20**                      |
| Detecção F1        | 0.12 ± 0.10           | **0.45 ± 0.25**                      |
| Erro 3D RMSE       | 3.21 m                | **1.03 m**                           |
| Erro 3D (30-50m)   | 3.39 m                | **0.86 m**                           |
| ID-switches (tot)  | 13                    | 17                                   |

## Conclusions
1. **Depth-band no frustum reduz o erro 3D de 3.2m → 1.0m (3×)** — ganho dominante,
   por remover o fundo distante que inflava a escala de normalização.
2. **Config final melhora MOTA/IDF1/F1 em 3-4×** sobre o baseline em condições variadas.
3. **MOTA absoluto ainda modesto** (gate 3D estrito + recall do YOLO em alvos
   distantes/ocluídos/fog) — o gargalo restante é o detector 2D (confirmado).

## Next steps towards stronger results
- Estender a City e Coastline (generalização cross-ambiente).
- Campanha de dados + retreino YOLO com fundo difícil (lever do recall 2D).
- Corrigir cálculo de MOTP no harness (nan em sequências sem match).
