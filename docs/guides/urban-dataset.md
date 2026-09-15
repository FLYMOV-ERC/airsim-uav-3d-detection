# Urban UAM dataset — quick guide

> **Note.** Headings and titles in this document have been translated to English;
> the body text is still the original Portuguese. It is published as a working
> record, not as a polished English document.
>
> **Status.** Operator guide for the canonical collector, now
> `dataset_generation/collect_airsim.py`. It also refers to
> `generate_dataset_1000.py` and `drone_tracking_pipeline.py`, both archived under
> `archive/`.


Generalização do `generate_dataset_1000.py` para 3 ambientes urbanos.
Mesma saída (YOLO + PointNet) — só muda o background.

## Why it had to be generalised

O v1 (`generate_dataset_1000.py`) assume **mundo aberto** (Blocks):
posições amostradas de forma uniforme funcionam porque não há obstáculos.
Em cidade os drones spawnariam dentro de prédios e o `simGetDetections`
não veria nada. Solução: **teleport + checagem de colisão por timestamp**.

## Setup (1x)

1. Copie o `settings_urban_uam.json` para o local que o AirSim lê
   (no Windows: `C:\Users\<seu_user>\Documents\AirSim\settings.json`).
   Por que: ele coloca todos os drones em `Z=-120m` na spawn — acima do
   skyline dos 3 envs, evitando crash inicial.

2. Confira o IP do AirSim (cat IP_AIRSIM.txt). Default do script: `<AIRSIM_HOST>`.

## How to run it

Você roda o script **1x por ambiente** — não dá pra trocar mapa Unreal
em tempo real. O fluxo é:

```bash
# 1. Sobe CityEnviron.exe no Windows.   Depois:
python generate_dataset_urban.py --env city --frames 1000 \
       --output dataset_urban_city

# 2. Fecha CityEnviron, sobe AirSimNH.exe.  Depois:
python generate_dataset_urban.py --env neighborhood --frames 1000 \
       --output dataset_urban_nh

# 3. Fecha AirSimNH, sobe CityPark.exe (ou LandscapeMountains).  Depois:
python generate_dataset_urban.py --env citypark --frames 1000 \
       --output dataset_urban_park
```

Cada execução escreve em diretório isolado.
Mergear depois é só `rsync`/`cp` (mesma estrutura do v1):

```bash
mkdir -p dataset_urban_merged/{yolo/{images,labels}/{train,val},pointnet/{point_clouds,labels_3d}}
for env in city nh park; do
  rsync -a dataset_urban_$env/yolo/      dataset_urban_merged/yolo/
  rsync -a dataset_urban_$env/pointnet/  dataset_urban_merged/pointnet/
done
# Regenera dataset.yaml apontando pro merged
```

## Techniques used (all inside the script)

| Técnica | Onde no código | Por quê |
|---|---|---|
| **Teleport (`simSetVehiclePose`)** | `teleport()` | 50x mais rápido que `moveToPositionAsync` para *posicionar* (sem precisar de física realista durante a coleta). |
| **Collision-aware por timestamp diff** | `find_safe_pose()` | `simGetCollisionInfo` retorna a *última* colisão; comparamos `time_stamp` ANTES vs DEPOIS do teleport para detectar colisão *nova*. |
| **Sampling viesado pro FOV horizontal** | `find_safe_pose()` (`half_fov = 0.85·FOV/2`) | Sem isso ~80% dos placements caem fora do quadro e viram frame vazio. |
| **Domain randomization** | `apply_weather`, `apply_time_of_day` | Mix 50% clear + 50% adverso (fog/rain/dust). Horas 07h–18h (evita drone preto vs céu preto). |
| **Detection API (ground-truth bbox)** | `simAddDetectionFilterMeshName` + `simGetDetections` | Labels perfeitas — não dependemos de modelo. Identico ao v1. |
| **Per-env profiles** | `ENV_PROFILES` (no topo) | Cada cidade tem skyline e área operacional diferentes. Os bounds podem ser refinados se você ver muitas `safe_failures` no log. |

## What to watch in the log

```
[env-randomization] weather=fog_heavy  hour=15h
  [  100/1000]  total=124  hit_rate= 80.6%  fps=1.45  safe_failures=12
```

- `hit_rate < 50%`: aumente `search_radius` do env (drones muito longe → bbox pequena → rejeitada).
- `safe_failures` crescendo rápido: skyline mais alto que o profile pensa — sobe `safe_init_altitude` e/ou aumenta o limite superior de `drone_z_bounds`.

## Next steps after the collection

1. Treinar YOLO com `dataset_urban_merged/yolo/dataset.yaml`.
2. Plugar no `drone_tracking_pipeline.py` substituindo a chamada `simGetDetections` pelo YOLO real.
3. Avaliar tracker (SORT 3D) com o YOLO treinado em vez do ground-truth.
