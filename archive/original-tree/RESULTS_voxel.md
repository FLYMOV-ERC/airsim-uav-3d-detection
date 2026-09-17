# Experimentos voxel adicionais — variações do detector 1-estágio (PointPillars)

60 vídeos (NH+City+Coast, 20 cada), ego em movimento, ground-truth 3D. Média entre os 3 ambientes.
Objetivo: explorar o que move o ponteiro no paradigma **voxel/cena-inteira** (sem YOLO) — quatro
modificações sobre o PointPillars-lite base, em ordem crescente de custo.

| Método (voxel, 1-estágio) | AMOTA↑ | MOTA@0.5↑ | det-P↑ | det-R↑ | det-F1↑ | RMSE 3D↓ | AMOTP↓ | IDF1↑ | ID-sw↓ |
|---------------------------|--------|-----------|--------|--------|---------|----------|--------|-------|--------|
| **Base** PointPillars (features hand-crafted) | 0.64 | 0.59 | 0.87 | 0.77 | 0.80 | 1.12 m | ~0.9 | ~0.40 | ~666 |
| **V3** resolução fina (célula 0.7 m) | 0.60 | 0.55 | 0.85 | 0.75 | 0.77 | 1.12 m | 0.98 | 0.35 | 701 |
| **V2** temporal multi-frame (3 frames, ego-comp.) | 0.52 | 0.47 | 0.82 | 0.72 | 0.74 | 1.58 m | 1.46 | 0.29 | 881 |
| **V4** voxel 3D denso (Conv3D, SECOND-style) | 0.69 | 0.69 | 0.92 | 0.81 | **0.84** | 0.85 m | 0.80 | 0.51 | 469 |
| **V1** PointPillars **PFN** (features aprendidas) | **0.69** | **0.69** | **0.92** | 0.80 | **0.84** | **0.78 m** | **0.72** | **0.54** | **429** |

## Achados (avaliação dos resultados)

**V1 — Features de pillar aprendidas (PFN): o maior ganho.** Trocar as features hand-crafted
(`[contagem, altura média, altura máx]`) por um **mini-PointNet por pillar** (MLP + max-pool sobre
os pontos da célula) melhora *tudo*: RMSE 3D **1.12→0.78 m**, AMOTA 0.64→0.69, det-F1 0.80→0.84,
ID-switches ~666→429. A rede aprende a geometria fina do drone dentro de cada célula, que as
estatísticas manuais descartam. **É o melhor localizador 3D de todo o estudo** (supera até o
frustum T-Net, 1.03 m), mantendo o recall alto do paradigma voxel (0.80) **sem gargalo do YOLO**.

**V4 — Voxel 3D denso: empata com o PFN.** Processar a altura com convoluções 3D *antes* de
colapsar (em vez de comprimi-la já na entrada, como o PointPillars) dá praticamente o mesmo
resultado (AMOTA 0.69, RMSE 0.85 m, F1 0.84) — **mesmo com grade mais grossa (2.0 m)**, pois a
cabeça de regressão recupera a precisão sub-voxel. Confirma que o ganho do V1/V4 vem de
**representar melhor a estrutura local**, não da resolução da grade. Custo: Conv3D é mais pesada;
overfitou cedo (melhor época = 2) → indica que com mais dados/regularização pode render ainda mais.

**V3 — Resolução mais fina: NÃO ajuda.** Reduzir a célula 1.25→0.7 m **piorou** (AMOTA 0.64→0.60),
sem mudar o RMSE (1.12 m). O gargalo é a **esparsidade de pontos no drone** (poucos retornos de
depth a distância), não a quantização da grade — mais células só criam mais picos espúrios (FP) e
ID-switches. Achado importante: *refinar a grade é a alavanca errada aqui.*

**V2 — Temporal multi-frame: PIORA (resultado negativo importante).** Acumular 3 frames passados
compensando o movimento do **ego** alinha o fundo estático mas **borra o drone, que está em
movimento**, em várias células (não há compensação do movimento do alvo). Resultado: RMSE
1.12→**1.58 m**, AMOTA 0.64→0.52, AMOTP dobra. Densificar a nuvem só ajuda para alvos estáticos;
para alvos móveis, o acúmulo ego-only **espalha** o objeto. Compensação por-objeto exigiria já ter
o tracking — circular. *Conclusão p/ o artigo: temporal ingênuo não serve p/ alvos rápidos.*

## Síntese
- **Melhor representação > mais resolução > mais frames.** O ganho voxel vem de aprender features
  locais (PFN) ou processar a altura em 3D (V4) — ambos levam o RMSE a ~0.8 m e a AMOTA a 0.69.
- **Recomendação p/ o artigo:** adotar **PointPillars-PFN** como o detector voxel forte (melhor
  localização 3D do estudo, 0.78 m; recall 0.80; precisão 0.92). É o candidato natural a substituir
  o PointPillars hand-crafted na comparação de paradigmas e na fusão tardia.
- **Resultados negativos a reportar:** resolução fina (sparsity-limited) e temporal ingênuo
  (smearing do alvo móvel) — ambos delimitam o espaço de design.

## Comparação com o frustum (paradigmas lado a lado)
| | Localização (RMSE 3D) | Recall | Precisão | AMOTA |
|---|---|---|---|---|
| Frustum T-Net (2-est., melhor frustum) | 1.03 m | 0.49 | 0.91 | 0.83 |
| **PointPillars-PFN (1-est., melhor voxel)** | **0.78 m** | **0.80** | 0.92 | 0.69 |
| Fusão tardia (T-Net ⊕ PP base) | 1.09 m | 0.68 | 0.89 | 0.75 |

→ O **PFN voxel** agora domina em localização e recall; o frustum T-Net ainda lidera AMOTA
(consistência temporal/menos FP). Fusão futura sugerida: **T-Net ⊕ PointPillars-PFN**
(juntar a melhor precisão temporal do frustum com o melhor recall+localização do voxel-PFN).
