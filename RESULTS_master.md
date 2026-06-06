# Comparação geral — TODOS os métodos (frustum + voxel + fusão)

60 vídeos (NH+City+Coast, 20 cada), ego em movimento, ground-truth 3D. Média entre os 3 ambientes.
Tabela mestra do artigo: reúne os métodos frustum (2-estágios, dependem do YOLO), os voxel
(1-estágio, cena inteira, sem YOLO) e a fusão tardia.

| # | Método | Paradigma | AMOTA↑ | MOTA@0.5↑ | det-P↑ | det-R↑ | det-F1↑ | RMSE 3D↓ | ID-sw↓ |
|---|--------|-----------|--------|-----------|--------|--------|---------|----------|--------|
| 1 | Baseline (frustum, max-norm) | frustum 2-est. | 0.05 | −0.23 | 0.25 | baixo | 0.15 | 2.99 m | ~13 |
| 2 | Norm. robusta (p95) | frustum 2-est. | 0.03 | — | — | — | 0.08 | 2.88 m | — |
| 3 | Depth-band (heurística) | frustum 2-est. | 0.82 | 0.45 | 0.91 | 0.48 | 0.59 | 1.20 m | ~50 |
| 4 | Frustum-ConvNet | frustum 2-est. | 0.81 | 0.41 | 0.84 | 0.44 | 0.55 | 1.15 m | ~48 |
| 5 | **Segmentação T-Net (F-PointNet)** | frustum 2-est. | **0.83** | 0.46 | 0.91 | 0.49 | 0.60 | 1.03 m | ~62 |
| 6 | PointPillars base (hand-crafted) | voxel 1-est. | 0.64 | 0.59 | 0.87 | 0.77 | 0.80 | 1.12 m | ~666 |
| 7 | PointPillars resolução fina (0.7 m) | voxel 1-est. | 0.60 | 0.55 | 0.85 | 0.75 | 0.77 | 1.12 m | 701 |
| 8 | PointPillars temporal (3 frames) | voxel 1-est. | 0.52 | 0.47 | 0.82 | 0.72 | 0.74 | 1.58 m | 881 |
| 9 | Voxel 3D denso (Conv3D) | voxel 1-est. | 0.69 | 0.69 | 0.92 | 0.81 | **0.84** | 0.85 m | 469 |
| 10 | **PointPillars-PFN (features aprendidas)** | voxel 1-est. | 0.69 | **0.69** | **0.92** | **0.80** | **0.84** | **0.78 m** | **429** |
| 11 | Fusão tardia (T-Net ⊕ PP base) | híbrido | 0.75 | — | 0.89 | 0.68 | 0.75 | 1.09 m | — |
| 12 | **Fusão T-Net ⊕ PFN (ancora T-Net)** | híbrido | **0.76** | 0.57 | 0.91 | 0.67 | 0.74 | 1.03 m | 443 |
| 13 | **Fusão T-Net ⊕ PFN (ancora PFN)** | híbrido | 0.71 | 0.65 | 0.91 | **0.81** | **0.84** | 0.79 m | ~480 |

## Líderes por métrica
- **Localização 3D (RMSE):** 🥇 **PFN 0.78 m** › Voxel-3D 0.85 m › T-Net 1.03 m › Fusão 1.09 m › PP base 1.12 m.
  *O voxel-PFN passou a ser o melhor localizador de todos — supera o melhor frustum (T-Net) em 24%.*
- **Recall (det-R):** 🥇 **Voxel-3D/PFN ~0.80** » todos os frustum ≤0.49 (teto imposto pelo YOLO).
- **Detecção F1:** 🥇 **PFN/Voxel-3D 0.84** › PP base 0.80 › Fusão 0.75 › T-Net 0.60.
- **Precisão (det-P):** empate no topo — **PFN/Voxel-3D/T-Net/depth-band ~0.91–0.92**.
- **AMOTA:** 🥇 **T-Net 0.83** › depth-band 0.82 › F-ConvNet 0.81 › Fusão 0.75 › PFN/Voxel-3D 0.69.
- **ID-switches:** frustum « voxel (frustum ~50; voxel-PFN 429, já bem melhor que o PP base ~666).

## Leitura (o que mudou com os experimentos voxel)
**Antes** a comparação era um trade-off limpo: *frustum = precisão alta + AMOTA alta, mas recall
preso pelo YOLO (~0.49); voxel = recall alto (0.77), mas ruidoso (RMSE 1.12, AMOTA 0.64)*. A fusão
tardia foi a tentativa de juntar os dois (AMOTA 0.75, recall 0.68).

**Agora**, com **features de pillar aprendidas (PFN)**, o voxel deixou de ser o lado "ruidoso":
- igualou a **precisão** do frustum (0.92 vs 0.91 do T-Net),
- manteve o **recall alto** do voxel (0.80 vs 0.49 do frustum),
- e passou a ter a **melhor localização** de todos (0.78 m vs 1.03 m do T-Net),
- com **F1 0.84** (o maior da tabela).

→ **O PFN domina 4 das 6 métricas** (RMSE, recall, F1, e empata precisão). O frustum T-Net só
mantém vantagem em **AMOTA** (0.83 vs 0.69) e **ID-switches** — ou seja, **consistência temporal**:
o frustum gera menos falsos-positivos ao longo da varredura de recall, enquanto o voxel ainda
produz mais picos espúrios que penalizam a integral AMOTA.

## Fusão tardia T-Net ⊕ PointPillars-PFN (duas ancoragens)
Refeita a fusão com o PFN no lugar do PP base. O resultado depende de **qual detector ancora**
(mantém posição/score) e qual entra como complemento descontado:
- **Ancora no T-Net** (#12): mantém as posições precisas do frustum, PFN adiciona detecções que ele
  perde → **melhor AMOTA entre as configs de recall alto (0.76)**, eleva o recall do T-Net
  **0.49→0.67**. Substitui a fusão antiga (AMOTA 0.75→0.76, RMSE 1.09→1.03 m).
- **Ancora no PFN** (#13): mantém o voxel (recall+localização), T-Net só confirma → preserva os
  pontos fortes do PFN (**recall 0.81, F1 0.84, RMSE 0.79 m**), AMOTA 0.71. Ganho do T-Net é
  marginal porque o PFN já cobre quase tudo que o T-Net detecta.

**A fusão NÃO supera o melhor standalone em AMOTA** — o **T-Net sozinho (0.83)** segue no topo, mas
com recall inaceitável (0.49, perde ~metade dos drones). O valor da fusão é **equilibrar precisão,
recall e localização numa única config**.

## Implicação para o artigo
- **Não há um vencedor único** — escolher pela aplicação:
  - **Máxima AMOTA / poucos FP (mas recall baixo):** frustum **T-Net** (0.83 / recall 0.49).
  - **Rastreio consistente com recall aceitável:** **fusão ancorada no T-Net** (AMOTA 0.76 / recall 0.67).
  - **Cobertura + localização máximas:** voxel **PFN** ou **fusão ancorada no PFN**
    (recall 0.81, RMSE 0.78–0.79 m, F1 0.84).
- **Resultados negativos a manter** (delimitam o espaço de design): p95 (normalizar não basta —
  tem de remover o fundo), resolução fina (sparsity-limited), temporal ingênuo (smearing do alvo).
