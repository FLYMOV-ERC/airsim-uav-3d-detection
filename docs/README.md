# docs/

Working documents kept from the project. **Their headings and titles have been
translated to English; the body text of every file below is still the original
Portuguese.** They are published as a record of how the work was done, not as
polished English documentation. Each file carries a status note at the top.

Where a document and the dissertation disagree, **the dissertation is correct** —
see "Known discrepancies" in the top-level [README.md](../README.md).

## The two substantive records

| file | what it is |
|---|---|
| [`compiled-technical-report.md`](compiled-technical-report.md) | The master technical record, 14 sections. Section 13.3 is a hand-written manifest of the canonical code; Section 13.4 lists every key constant verbatim; Section 4.3 documents the three failed automatic-labeling routes; Section 9.3 states the in-sample caveat for the voxel results. The single most useful file here after the code itself. |
| [`methodology.md`](methodology.md) | The most compact accurate narrative of the whole method — dataset, 2D stage, frustum, depth band, fusion, SORT, metrics — paralleling Sections 7.2 to 7.6 of the chapter. Its results table is stale. |

## Phase-1 documentation

| file | what it is |
|---|---|
| [`phase1-technical-documentation.md`](phase1-technical-documentation.md) | Phase-1 technical documentation: the multimodal collection, the sensor configuration, and the early detector work. Section 4.1 is the source of [`configs/settings.json`](../configs/settings.json). Several scripts it presents as current are now under `archive/`. |

## Results

Every file under `results/` predates the dissertation's final numbers and
disagrees with them by small amounts across most rows. **Why has not been
traced** — it is not the held-out recomputation of Section 7.4.3, which concerns
only the voxel rows, whereas the frustum rows differ here too. See "Known
discrepancies" in the top-level [README.md](../README.md); where the two
disagree, the dissertation is the reported result.

| file | what it is |
|---|---|
| [`results/master.md`](results/master.md) | The master results table: all 13 method rows (frustum + voxel + fusion) with AMOTA, MOTA@0.5, P/R/F1, RMSE and ID switches. The direct antecedent of the chapter's architectural-comparison table. |
| [`results/architectures.md`](results/architectures.md) | The architecture comparison (baseline / p95 / depth band / ConvNet / T-Net / PointPillars / late fusion). |
| [`results/voxel.md`](results/voxel.md) | The five-variant voxel campaign, including both negative controls (finer 0.7 m grid, 3-frame temporal accumulation). |
| [`results/paper.md`](results/paper.md) | A paper-oriented cut of the same results. |
| [`results/scientific.md`](results/scientific.md) | Output of the earlier `eval_scientific.py` protocol, now archived. Predates the dump protocol. |

## Setup and operation

| file | what it is |
|---|---|
| [`environments.md`](environments.md) | The AirSim environments (AirSimNH, City, Coastline and others) and how to switch between them. |
| [`lidar-config.md`](lidar-config.md) | LiDAR settings from a point in the project's history. Does not all match `configs/settings.json`; see [`configs/README.md`](../configs/README.md). |
| [`guides/urban-dataset.md`](guides/urban-dataset.md) | Operator guide for the canonical collector, `dataset_generation/collect_airsim.py`. The only how-to-run document for the canonical collection path. |
| [`setup/wsl.md`](setup/wsl.md) | WSL ↔ Windows networking, needed when the Python client and the simulator run on different machines. |
| [`setup/windows.md`](setup/windows.md) | Windows-side AirSim / Unreal setup. |

Two further guides, covering the archived multi-scenario collection workflow,
live with the code they document under [`../archive/docs/`](../archive/docs/).
