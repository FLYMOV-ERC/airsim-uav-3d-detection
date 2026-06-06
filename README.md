# Aerial 3D Detection Pipeline (AirSim)

Synthetic dataset generation and 3D object-detection pipeline for UAV
sense-and-avoid, developed for the master's dissertation *Object Detection and
Tracking Using an Unmanned Aerial Vehicle* (Eric E. Y. de Lima, ITA, 2026).

This repository contains the **code** behind Chapter 7 of the dissertation: a
configurable synthetic aerial dataset built in [AirSim](https://github.com/microsoft/AirSim),
and the detection front-end that turns camera + point-cloud data into 3D obstacle
detections for tracking.

> **Note.** Datasets, model weights, videos and other large artifacts are **not**
> included (they are gitignored). The scripts regenerate the data from AirSim and
> train the models from it.

## Pipeline

```
RGB image ──► YOLOv11s (2D) ──► frustum crop ──► PointNet++ (3D box) ──► EKF tracker
```

A single-stage **voxel** family (PointPillars / learned-pillar PFN / dense 3D)
is also implemented as a comparison baseline.

## Key entry points

| Script | Purpose |
|---|---|
| `generate_dataset_urban.py`, `build_*_dataset.py` | AirSim data collection and dataset assembly (frustum / painted / pointnet) |
| `capture_backgrounds.py` | drone-free backgrounds for synthetic composition |
| `train_yolo11s_drone.py` | train the 2D detector |
| `train_pointnet*.py` | train the frustum PointNet++ / painted / segmentation variants |
| `multi_tracker_ekf.py`, `driver_ekf.py` | per-track range–bearing EKF + SORT-style association |
| `metrics_ekf_from_dump.py` | recompute detection/tracking metrics (AMOTA, etc.) |
| `make_dissertation_figs.py` | regenerate the figures used in the dissertation |

## Requirements

Python 3.10, PyTorch, Ultralytics (YOLOv11), NumPy, Matplotlib, and the AirSim
Python client. See the import headers of each script.

## Citation

If you use this code, please cite the dissertation (and the FUSION 2025 paper for
the tracker):

> E. E. Y. de Lima, S. S. Dias, M. R. O. A. Máximo, *Extended Kalman
> Filter-Based Object Tracking Using Global and Local Frames*, IEEE International
> Conference on Information Fusion (FUSION), 2025.

## License

MIT — see [LICENSE](LICENSE).
