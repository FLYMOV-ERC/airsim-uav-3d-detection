# airsim-uav-3d-detection

Synthetic aerial dataset generation and camera + point-cloud 3D object detection
for UAV sense-and-avoid, in the AirSim simulator.

This is the code behind **Chapter 7** of the master's dissertation *Object
Detection and Tracking Using an Unmanned Aerial Vehicle* (Eric E. Y. de Lima,
Instituto Tecnológico de Aeronáutica, defended 28 July 2026). It contains:

- the generator that builds a fully annotated synthetic aerial dataset in
  [AirSim](https://github.com/microsoft/AirSim);
- the two-stage detector — YOLOv11-small → viewing frustum → PointNet++ — and its
  variants (depth band, Frustum-ConvNet, learned T-Net segmentation);
- three single-stage voxel baselines (hand-crafted PointPillars,
  PointPillars-PFN, dense 3D voxel) and a late fusion of the two paradigms;
- the tracking stage — SORT-style 2D association plus one Extended Kalman Filter
  per track, the same filter derived in Chapter 5 — through which *every*
  detector variant is evaluated;
- the offline dump-and-recompute evaluation protocol that produces every number
  in the chapter's result tables.

```
RGB image ──► YOLOv11s (2D) ──► frustum crop ──► PointNet++ (3D box) ──► SORT ──► per-track EKF
                                     ▲
depth image ──► point cloud ─────────┘
```

No dataset, no recorded session, no trained weight and no detection dump is
committed here. See [What is deliberately not included](#what-is-deliberately-not-included).

---

## Where this repository lives

Canonical location: **`github.com/FLYMOV-ERC/airsim-uav-3d-detection`**, in the
FlyMov Engineering Research Center organisation. [`CITATION.cff`](CITATION.cff)
points there.

There is only ever one repository. It was created in the author's personal
namespace as `github.com/ericyoshida/airsim-uav-3d-detection` and later
transferred into the organisation; the transfer moved the same repository, it did
not fork or re-publish it. GitHub keeps a permanent redirect from a transferred
repository's former path, and that redirect is in place: the old URL answers
`301` to the canonical one.

That matters because the footnote printed in the dissertation (Chapter 7, Section
7.4.3, at the end of "Training and Evaluation Protocol") cites the personal-namespace
URL — the chapter was written before the transfer — so the URL as printed in the
thesis resolves here. Use the canonical URL for anything new.

---

## Repository layout

```
configs/          settings.json (the AirSim sensor suite) + default.yaml (host, port, paths)
common/           geometry (camera model, frame conversions), the threaded pipeline, VisQuad attachment,
                  airsim_optional.py (the guarded AirSim-client import that keeps the offline path runnable)
dataset_generation/
  collect_airsim.py       THE generator that produced the reported dataset
  record_campaign.py      records one evaluation sequence
  flight.py               FlightThread — the oscillating, moving ego
  measure_quadrotor.py    measures the VisQuad extents
  synthetic/              sprite capture, background capture, composition
  build/                  frustum / painted / point-cloud-cache dataset builders
detection/
  yolo/                   train and run the 2D detector
  frustum/                the adopted 3D detector + ConvNet and SegNet variants
  pointnet/               the shared PointNet++ set-abstraction architecture
  voxel/                  PointPillars, PointPillars-PFN, dense 3D voxel
tracking/                 SORT association, the Chapter 5 EKF, the multi-target wrapper
evaluation/               detection dumps, late fusion, metrics, per-stage harness, QA checks
visualization/            figure generation, result videos, BEV renderers, PLY exporters
scripts/                  end-to-end session runners
tests/                    the two tests that run without a simulator
docs/                     methodology, the compiled technical report, results, setup guides
archive/                  54 superseded variants, kept as provenance — five of them are still
                  imported by the working pipeline; see archive/README.md
experiments/              the prior question: whether any public dataset made this one
                  unnecessary. See experiments/public-dataset-feasibility/
```

### Why a synthetic dataset at all

Before any of this was built, two public routes were tried and closed.
VisDrone2019-DET has no airborne-vehicle class — its ten classes are pedestrian,
people, bicycle, car, van, truck, tricycle, awning-tricycle, bus and motor — and
NTU VIRAL's ground truth is the carrying vehicle's own Leica-prism pose, not the
pose of anything it observes. Neither can supervise a detector for an aircraft
seen from another aircraft. A third route, training on a real recording of a
quadrotor, produced a detector that scored mAP50 0.995 on held-out frames of its
own session and then found nothing at all in a second session of the same
aircraft. `experiments/public-dataset-feasibility/` carries the scripts, the run
records and the measurements behind those three results, including a retraction
of an earlier reading of the third one.

---

## Setup

```console
$ python -m venv .venv && source .venv/bin/activate     # Python 3.10
$ pip install -r requirements.txt
```

Install PyTorch separately if you need a specific CUDA build.

On the machine running the simulator, copy `configs/settings.json` to AirSim's
settings location (`~/Documents/AirSim/settings.json`, or the Windows equivalent)
and restart the simulator. That file declares the sensor suite Chapter 7
specifies: a 1280×720 camera with a 90° horizontal field of view, mounted 0.35 m
forward and 0.5 m below the body origin with a −15° pitch, a co-located 64-beam
LiDAR, and four vehicles (one ego, three targets). See
[configs/README.md](configs/README.md), which also records how that file was
recovered, which of its fields are corroborated by the code and which are not,
and the conflicting LiDAR configurations on record elsewhere in the project.

Point the client at the simulator through `configs/default.yaml`, the
`AIRSIM_HOST` / `AIRSIM_PORT` environment variables, or `--host` / `--port`.
The default is `127.0.0.1:41451`. If the client runs under WSL2 and the
simulator on the Windows host, see [docs/setup/wsl.md](docs/setup/wsl.md).

`paths.data_root` in `configs/default.yaml` is the root under which `datasets/`,
`sessions/`, `runs/` and `dumps/` are resolved. Nothing in this repository stores
an absolute path.

---

## Reproducing Chapter 7

The chapter's numbers come from an offline protocol: one heavy inference pass per
sequence writes a *dump* of raw detections, and every metric is then recomputed
from the dumps cheaply, while sweeping the detection threshold. That is why the
evaluation stage needs no GPU and why comparing eleven detector configurations
was tractable.

### 1. Build the data

```console
# synthetic composites for the 2D detector (Section 7.2.3)
$ python dataset_generation/synthetic/capture_sprites.py       # 196 sprites
$ python dataset_generation/synthetic/capture_backgrounds.py   # ~600 drone-free backgrounds
$ python dataset_generation/synthetic/compose.py               # ~2,500 pixel-exact frames

# multimodal collected set for the 3D stage (Section 7.2.4), once per environment
$ python dataset_generation/collect_airsim.py --env city         --frames 1000
$ python dataset_generation/collect_airsim.py --env neighborhood --frames 1000
$ python dataset_generation/collect_airsim.py --env citypark     --frames 1000

# the 60-sequence evaluation campaign — see the note below
$ python dataset_generation/record_campaign.py ...
```

### 2. Train

```console
$ python detection/yolo/train.py --imgsz 1280 --epochs 80        # mAP50 = 0.994
$ python dataset_generation/build/frustum_dataset.py             # frustum samples
$ python detection/frustum/train.py                              # frustum PointNet++
$ python detection/frustum/train_segnet.py                       # + learned T-Net segmentation
$ python dataset_generation/build/pc_cache.py                    # needed before any voxel run
$ python detection/voxel/pointpillars_pfn.py                     # learned-pillar voxel baseline
```

### 3. Dump, fuse, and score

```console
$ python evaluation/dump_detections.py --sessions "sessions/*_seq*"   # frustum, one pass/sequence
$ python evaluation/dump_voxel.py      --sessions "sessions/*_seq*"   # voxel, same dump format
$ python evaluation/fuse_dumps.py      --a dumps/segnet --b dumps/pfn # late fusion
$ python evaluation/metrics_ekf.py     --dumps dumps/segnet           # ← the reported numbers
$ python evaluation/driver_ekf.py                                     # per-environment aggregation
```

`evaluation/metrics_ekf.py` — not `evaluation/metrics.py` — is the reporting path
for the chapter's result tables (`tab:main` and `tab:arch`). Section 7.3.7 states
that "the track's reported three-dimensional position — and hence the
localization error of Section 7.5 — is the *filtered* estimate", and
`metrics_ekf.py` is the one that reports the EKF's filtered position. `metrics.py` scores the raw detections and is kept for
comparison.

---

## Which script produces which table or figure

### Tables

| Chapter 7 | Content | Produced by |
|---|---|---|
| **Table 7.1** | summary of the data (synthetic ≈ 2,500 frames; collected 3 × 1,000; campaign 60 sequences) | `dataset_generation/synthetic/compose.py`, `dataset_generation/collect_airsim.py`, `dataset_generation/record_campaign.py`; counts verifiable with `evaluation/qa/analyze_dataset.py` |
| **Table 7.2** | tracking-stage parameters (NMS IoU 0.4, gate 0.3, min hits 3, coast 4, max age 8; σ_a, σ_ρ, σ_α, σ_β) | the defaults of `tracking/sort.py` and `tracking/ekf_3d.py` — the table is a transcription of those constants |
| **Table 7.3** *(`tab:main`)* | full frustum pipeline per environment (AMOTA, MOTA@0.5, IDF1, precision, recall, RMSE) | `evaluation/dump_detections.py` → `evaluation/metrics_ekf.py` → `evaluation/driver_ekf.py` |
| **Table 7.4** *(`tab:arch`)* | architectural comparison — five frustum rows, five voxel rows, one fusion row | frustum rows: `detection/frustum/detector.py` under different `NORM_MODE` / `band_m` / `arch` settings; voxel rows: `detection/voxel/pointpillars.py` (with `VOXEL_CELL` and `VOXEL_TK` producing the two negative controls), `detection/voxel/pointpillars_pfn.py`, `detection/voxel/voxel3d.py`, dumped through `evaluation/dump_voxel.py`; fusion row: `evaluation/fuse_dumps.py`. All scored by `evaluation/metrics_ekf.py` |

The chapter numbers its tables in sequence; `tab:main` is the per-environment
table and `tab:arch` the architectural comparison. The names above are the LaTeX
labels, which do not move if the numbering does.

### Figures

| Chapter 7 | Content | Produced by |
|---|---|---|
| `fig:dataset` | synthetic-composition pipeline (sprites, backgrounds, composites) | `visualization/make_dissertation_figs.py` → `fig_dataset_pipeline` |
| `fig:depthband` | the frustum-normalization problem on a real frustum | `visualization/make_dissertation_figs.py` → `fig_depthband` |
| `fig:tracker` | tracking-stage flow and the track timeline with coasting | `visualization/make_dissertation_figs.py` → `fig_tracker` |
| `fig:yolo` | 2D detector metrics, mAP50 = 0.994 | `visualization/make_dissertation_figs.py` → `fig_yolo` |
| `fig:norm` | foreground-extraction ablation (max-norm / p95 / depth band / T-Net) | `visualization/make_dissertation_figs.py` → `fig_normalization` |
| `fig:mainresults` (left) | per-environment AMOTA, recall, F1, RMSE | `visualization/make_dissertation_figs.py` → `fig_perenv` |
| `fig:mainresults` (right) | 3D localization error versus range | `visualization/make_dissertation_figs.py` → `fig_range` |
| `fig:arch` | frustum / voxel / fusion comparison | `visualization/make_dissertation_figs.py` → `fig_comparison` |
| `fig:regimes` | precision-recall regimes of the two paradigms | `visualization/make_dissertation_figs.py` → `fig_regimes` |
| `fig:qual` | qualitative results | `visualization/make_dissertation_figs.py` → `fig_qualitative` |
| Chapter 4 pipeline hub | the perception-pipeline diagram | `visualization/make_dissertation_figs.py` → `fig_pipeline` |
| supplementary video | first-person result video (RGB + detections + tracks + ground truth) | `visualization/render_results.py` |
| supplementary BEV | ego-centric and world-frame bird's-eye renders | `visualization/render_bev.py`, `visualization/render_bev_global.py` |

Regenerate the figures with:

```console
$ python visualization/make_dissertation_figs.py --outdir /path/to/dissertation --data-root /path/to/data
```

which writes `<outdir>/Cap4` and `<outdir>/Cap7`.

### Figures this script does not produce

Seven of the chapter's figures are **not** generated by any script in this
repository:

`frustum_pipeline_teaser`, `depthband_diagram`, `pointnet_topology`,
`approaches_diagram`, `localization_over_time`, `pipeline_live`, `pipeline_frames`.

The first four are diagrams and appear to have been drawn by hand. The fifth,
`localization_over_time`, is quantitative — it carries the 5,359 matched pairs,
the Spearman ρ = −0.10, the Kruskal–Wallis H = 130 and the Mann–Whitney p = 0.53
of Section 7.5.4 — and the statistics code that produces it is not in this
repository. Where that code lives is unresolved; it is not reconstructed here.

---

## Missing from this repository: `record_campaign.sh`

Section 13.3 of [docs/compiled-technical-report.md](docs/compiled-technical-report.md)
lists a shell driver, `record_campaign.sh`, as core collection code, and Section
4.6 states that the whole 60-sequence campaign — 20 configurations per
environment, resumable, validating each recording against corruption — ran
through it. **That file is not in this repository.** It was not present in the
source tree this repository was reorganised from, and it has not been
reconstructed, because guessing at the configuration matrix it encoded would
misrepresent how the campaign was actually run.

`dataset_generation/record_campaign.py` records **one** sequence and is present
and complete. What is missing is the orchestration around it: the list of
20 configurations per environment (range regime, hour, weather, target count) and
the resume logic. Until that file is recovered from the machine it lives on, the
campaign of Table 7.1 is not reproducible end-to-end from what is published here,
although every individual sequence is.

---

## Known discrepancies

Stated rather than quietly reconciled, so that a reader who checks does not have
to wonder which source is wrong.

1. **The `docs/results/*.md` tables disagree with the dissertation, and the
   reason has not been traced.**
   `docs/results/master.md` gives PointPillars-PFN RMSE 0.78 m, T-Net 1.03 m and a
   best AMOTA of 0.83, where Chapter 7's Table 7.4 gives 0.87 m, 1.05 m and 0.81;
   most other rows differ slightly as well. Those documents predate the
   dissertation's final numbers and are kept for provenance; where the two
   disagree, the dissertation is the reported result.

   **No mechanism for the difference is offered here, because none has been
   established.** It is specifically *not* the held-out recomputation of Section
   7.4.3. That recomputation concerns only the voxel rows — the frustum networks
   were trained on separate data and evaluated on the whole 60-sequence campaign,
   which is therefore already a fully held-out test for them — yet the frustum
   rows differ too (T-Net RMSE 1.03 m here against 1.05 m in the chapter, AMOTA
   0.83 against 0.81). And Section 7.4.3 states that the voxel figures *printed in
   Table 7.4 are the full-campaign ones*, to be read as mildly optimistic, so the
   chapter's voxel numbers are not the corrected held-out figures either.

2. **The LiDAR channel count.** Chapter 7 §7.2.1 and §7.2.5 and
   `docs/phase1-technical-documentation.md` §4.1 all say 64 beams;
   `docs/lidar-config.md` discusses other values from a different point in the
   project's history. `configs/settings.json` ships the 64-channel configuration.
   The LiDAR is an auxiliary, visualisation-only channel — the detectors consume
   the depth-camera cloud — so no reported result depends on this.

3. **`docs/phase1-technical-documentation.md` describes Phase 1.** It presents
   several scripts as current that are now under `archive/` (the plain-PointNet
   training, the early dataset builders). It is kept because it is one of the only
   records of the sensor configuration and the early detector work, not as a
   current guide.

---

## What is deliberately not included

- **The datasets.** The synthetic composites, the three collected environment
  sets, and the 60-sequence campaign are all regenerable from the scripts here,
  given a working AirSim installation. They are large and are not version-control
  material.
- **Trained weights.** `runs/` is gitignored. Retraining reproduces them; the
  hyperparameters are in the training scripts and in Section 7.3.3.
- **Detection dumps.** `dumps/` is gitignored. They are the cheap intermediate the
  metrics are recomputed from, and they would let anyone recompute every number
  in the result tables in seconds without a GPU or a simulator — but publishing
  them is a separate decision that has not been made.
- **Any imagery.** No frame, sprite, background or rendered video is committed.
- **`record_campaign.sh`.** Missing rather than excluded; see above.
- **The AirSim environment binaries.** AirSimNH, CityEnviron and CityPark are
  distributed by Microsoft under their own terms and are not redistributed.

---

## Tests

Two tests run with no simulator connection:

```console
$ python tests/test_ekf_synthetic.py     # EKFTracker3D on a synthetic trajectory
$ python tests/test_pipeline_offline.py  # end-to-end smoke test on one recorded frame
```

`test_ekf_synthetic.py` is the only direct check of the Chapter 5 filter port.
Neither has been converted to pytest assertions yet; both print their results and
`test_pipeline_offline.py` still needs Ultralytics and a recorded frame.

---

## Relationship to the other chapters

`tracking/ekf_3d.py` is a direct port of the MATLAB `ekf_filter.m` of Chapter 5 —
the seven-state model `[x, y, z, ẋ, ẏ, ż, r]`, constant velocity with white
acceleration process noise, spherical measurement with its Jacobian — reused here
unmodified, one filter per track. It is where the detection front-end of
Chapter 7 joins the tracker of Chapter 5, and it is imported by 18 modules here
(the most imported module in the repository is `common/config.py`, at 39). The monocular tracker of Chapter 6 is a variant of the same
framework adapted to a bearing-only measurement, not this identical filter.

---

## Licence

**AGPL-3.0-or-later** — see [LICENSE](LICENSE). Every commit on `main` carries
that licence; there is no MIT-licensed commit on that branch.

There was an earlier MIT release, and it was a release of *this same repository*,
not of a predecessor: on 6 June 2026, under its former name
`ericyoshida/airsim-uav-3d-detection`, the working tree was published as a single
flat commit with `LICENSE` = MIT. That commit is preserved here under the tag
`archive/flat-dump-2026-06`. `main` was subsequently rewritten as a fresh root
commit, so the tagged commit is genuinely not an ancestor of `main` — the two are
disjoint histories of one repository, the tag holding the MIT-licensed dump and
`main` holding the AGPL-3.0 reorganisation of it. Anyone who obtained a copy under
the MIT terms keeps their MIT rights to it. See [NOTICE](NOTICE), which also
discloses what that archived commit contains.

This tree is AGPL-3.0 because the 2D detection stage is built on Ultralytics
YOLO, which is AGPL-3.0: twelve modules import `ultralytics` directly and seven
more reach it transitively through `common/pipeline.py`, and distributing that
combined work under MIT is not permitted. See [NOTICE](NOTICE) for the full
reasoning and for the third-party licences (Ultralytics AGPL-3.0, Microsoft
AirSim MIT).

---

## Citation

See [CITATION.cff](CITATION.cff). If you use this code, please cite the
dissertation, and the FUSION 2025 paper for the tracker:

> E. E. Y. de Lima, S. S. Dias, M. R. O. A. Máximo, *Extended Kalman
> Filter-Based Object Tracking Using Global and Local Frames*, IEEE International
> Conference on Information Fusion (FUSION), 2025.
