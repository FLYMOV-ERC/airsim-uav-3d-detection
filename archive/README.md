# archive/ — superseded variants, kept for provenance

Almost everything here is a superseded variant, an abandoned approach, or a
one-off study. **The exception, stated first because it is the one thing a reader
would otherwise get wrong: five modules under `early_detectors/` are imported by
code outside this directory and cannot be deleted without editing that code.**

| module | imported by |
|---|---|
| `early_detectors/detector_painted_v4.py` | `common/pipeline.py:45`, `evaluation/dump_detections.py:10` |
| `early_detectors/detector_frustum_v1.py` | `common/pipeline.py:46` |
| `early_detectors/detector_depth_direct.py` | `common/pipeline.py:48` |
| `early_detectors/train_pointnet.py` | `early_detectors/detector_frustum_v1.py:24` |
| `early_detectors/pointnet_inference.py` | `tests/test_pipeline_offline.py:22` |

The first three are the `painted`, `frustum` and `depth` branches of
`make_detector()` in `common/pipeline.py` — earlier baselines left selectable for
comparison. The detector Chapter 7 adopts is `detection/frustum/detector.py`
(`frustum_pn2`) and does not use any of them; but `common/pipeline.py` imports all
three unconditionally, and fifteen modules import `common/pipeline.py`, so these
files are on the import path of the whole working pipeline even when none of them
runs.

The rest is kept, and not deleted, for one reason: **roughly half of these files
are the evidence for statements the dissertation already makes in print.**
Chapter 7 reports the accuracy of the plain-PointNet and PointPainting baselines,
names three automatic-labeling routes that failed and says why each failed, and
states its preference for 2D over 3D identity association. Deleting the code
behind those claims would leave the claims unsupported.

These files are not maintained. Their imports were mechanically re-pointed at
the new module paths and their hardcoded simulator address was replaced by
`common.config`, but they were not otherwise updated, not re-tested, and in
several cases they read dataset directories that no longer exist. Treat them as
a record, not as a tool.

## The one thing that is *not* in here

`generate_dataset_urban.py` — now
[`dataset_generation/collect_airsim.py`](../dataset_generation/collect_airsim.py)
— **is** the generator that produced the dataset reported in Chapter 7, and it
is canonical, not archived.

Around forty collection scripts accumulated over the project. Only that one
carries all four of the properties the chapter's dataset requires:

1. the **VisQuad 4× visual mesh** attached to every target — Section 7.2.2 makes
   this mandatory ("the 2D detector is trained on it and assumes the same mesh at
   inference"), so no script lacking it can have produced the reported data;
2. **all four sensor channels** the chapter names — Scene, DepthPlanar,
   Segmentation, and LiDAR;
3. **`simPause` freezing** of the scene during the per-frame acquisition, without
   which the targets drift by about a metre across the successive RPC queries and
   the render no longer matches its annotation (Section 7.2.1);
4. **all three environments** the chapter uses — AirSimNH, City, and Coastline.

The eight generators under `dataset_generators/` are kept because each is the
last surviving representative of a *distinct approach*, not because any of them
produced published data. The remaining thirty-odd variants were iteration noise
and were removed.

## What is in each subtree

### `dataset_generators/` (8 files)

One representative per collection approach that was tried and superseded.
`generate_dataset_depth.py` is the point at which the depth camera displaced the
LiDAR as the primary cloud source; `generate_dataset_lidar_improved.py` is the
LiDAR-primary route after its vertical field of view was widened — the experiment
Section 7.2.1 refers to when it says the beams striking a distant drone are too
few; `dataset_lidar_camera_aligned.py` is the frame-alignment work behind the
shared optical frame. `collect_dataset.py`, `collect_dataset_multi.py`,
`run_multi_environments.py` and `scene_variation.py` are the documented Phase-1
multi-scenario workflow (their guides are in `archive/docs/`).
`generate_dataset_1000.py` is the 1,000-frame-per-scene precursor that the
published urban-dataset guide still references.

### `labeling_attempts/` (16 files)

The three automatic-labeling routes of Section 7.2.3 that were attempted and
abandoned, and the probes that established each failure.

* Route (i), **semantic segmentation** — `generate_dataset_segmentation.py`,
  `probe_segmentation_api.py`, and the numbered chain under `seg_pipeline/`
  (`01_mark_classes` → `03_map_seg_colors*` → `04_build_bboxes*` →
  `05_preview_overlay`, plus `check_seg_colors.py` and
  `debug_segmentation_colors.py`, the two checks that revealed the drones shared
  a stencil colour with the background).
* Route (ii), **the detection API** — `generate_dataset_detections_api.py`,
  `detection_api_probe.py` (the systematic sweep behind the chapter's "more than
  twenty mesh-name filter patterns"), `probe_detection_api.py` (the per-target
  recall measurement behind "missed roughly a third"), plus `find_mesh_names.py`
  and `find_drone_mesh_name.py`.

### `early_detectors/` (12 files)

The detector designs that preceded the adopted frustum PointNet++, including the
two whose numbers Chapter 7 reports:

* `detector_frustum_v1.py` + `train_pointnet.py` — the plain-PointNet
  classification baseline (accuracy 0.97, normalized centre error ≈ 3.6);
* `detector_painted_v4.py` + `train_pointnet_painted.py` +
  `pointnet_inference.py` + `build_pointnet_dataset.py` — the PointPainting
  whole-cloud route (centre error 3.6 → 0.003, validation F1 0.50);
* `sort_tracker_3d.py` — 3D association, the design Section 7.3.7 passes over in
  favour of associating on the 2D boxes ("Performing the association on the 2D
  boxes rather than on the noisier 3D positions keeps identities stable");
* `detector_depth_direct.py` — a network-free null model reading depth at the box
  centre. Not reported in the chapter; kept to show the design space was bounded
  from below.
* `build_frustum_dataset.py`, `extract_frustums.py`,
  `drone_tracking_pipeline.py`, `probe_pointnet_wrapper.py` — earlier dataset
  builders and the first end-to-end pipeline.

### `early_evaluation/` (5 files)

The evaluation protocol that preceded the dump protocol. `eval_scientific.py`
produced `docs/results/scientific.md`; `eval_offline.py` and `process_offline.py`
are its offline drivers. Every number in the chapter comes from the current dump
protocol (`evaluation/dump_detections.py` → `evaluation/metrics_ekf.py`), not
from these.

### `validation/` (10 files)

Substantive studies that settled specific questions, several of which the chapter
states as fact:

* `probe_manual_bbox.py`, `probe_live_bbox.py`, `probe_live_bbox_v2.py`,
  `probe_live_bbox_v3.py` — the checks of the manually constructed 3D box against
  the simulator's own box; this series is the direct evidence for the central
  annotation finding of Section 7.2.2;
* `analyze_sync_problem.py` — the measurement behind the "~1 m drift in the
  half-second spanned by successive RPC queries" that motivates `simPause`;
* `check_coordinate_system.py` — the NED-world / CV-optical frame study, and the
  R-versus-Rᵀ projection bug;
* `probe_lidar_fov.py`, `validate_lidar_config.py`, `verify_lidar_source.py` —
  the LiDAR field-of-view work behind the decision to keep LiDAR as an auxiliary
  channel only (Section 7.2.5);
* `relabel_v4_3d.py` — the migration tool that applied the corrected geometric
  annotation retroactively to an already-collected dataset.

### `lidar_fusion/` (1 file)

`lidar_camera_fusion_corrected.py` — the single surviving representative of a
thirteen-file family of LiDAR-to-image projection scripts from Phase 1. It
produces the LiDAR-returns-on-image view of Figure 7.16 (`fig:pipeline_frames`).
**This is not the chapter's fusion.** The late fusion of Table 7.4 (`tab:arch`) is
a decision-level union of two detection dumps and lives in
`evaluation/fuse_dumps.py`.

### `docs/` (2 files)

`GUIA_COLETA_MULTI_CENARIO.md` and `README_MULTI_DRONE.md`, the operator guides
for the multi-scenario collectors above. They move with the code they document.
Both are in Portuguese and were not translated, because they describe an archived
workflow.

## What was removed rather than archived

122 files were deleted from the published tree: roughly thirty near-identical
generator iterations, forty-five scratch connection probes and single-frame
grabs, twelve duplicate LiDAR-camera projection scripts, twelve debug one-offs,
and nine scripts targeting the Blocks, Mountains and AbandonedPark environments
that the chapter does not use. Every bug those debug scripts chased is fixed in
the canonical code and written up in `docs/methodology.md` and
`docs/compiled-technical-report.md`.
