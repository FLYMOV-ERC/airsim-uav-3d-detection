# configs/

Two files, with different jobs.

| file | consumed by | what it fixes |
|---|---|---|
| `settings.json` | the **AirSim simulator**, on the machine running the Unreal binary | the sensor suite: camera resolution, field of view, lever arm and tilt; the LiDAR; the vehicles and their spawn origins |
| `default.yaml` | the **Python client**, i.e. everything in this repository | where the simulator is (host/port), where the data lives, and a mirror of the geometry above so the client and the simulator cannot drift apart |

## `settings.json`

Copy it to the location AirSim reads on the simulator host:

* Windows — `C:\Users\<you>\Documents\AirSim\settings.json`
* Linux — `~/Documents/AirSim/settings.json`

Restart the simulator after any change; AirSim reads this file once at startup.

What it declares, and where the dissertation states the same numbers:

| setting | value | stated in |
|---|---|---|
| `front_center` capture | 1280 × 720, FOV 90° | §7.2.1 (gives *f<sub>x</sub>* = *f<sub>y</sub>* = 640 px, *(c<sub>x</sub>, c<sub>y</sub>)* = (640, 360) px) |
| camera lever arm | X 0.35, Y 0.0, Z −0.5 m | §7.2.1 ("0.35 m forward, 0.5 m below the body origin") |
| camera pitch | −15° | §7.2.1 |
| capture channels | ImageType 0, 1, 2, 5 — Scene, DepthPlanar, DepthPerspective, Segmentation | §7.2.4 names RGB, depth and segmentation; DepthPerspective (2) is declared here but never requested — `collect_airsim.py` asks only for Scene, DepthPlanar and Segmentation |
| `LidarFront` channels | 64 | §7.2.1, §7.2.5 ("a 64-beam LiDAR") |
| LiDAR lever arm / pitch | identical to the camera | §7.2.1 ("all rigidly co-located … share a single optical frame") |
| vehicles | Ego + Drone3 + Drone4 + Intruder1 | §7.2.4 (2–3 targets per sequence) |
| spawn origins | Ego (0,0,−5), Drone3 (15,−5,−5), Drone4 (15,5,−5), Intruder1 (−15,0,−5) | matched by `VEHICLE_ORIGINS` in `common/drone_visual.py` and `dataset_generation/collect_airsim.py` |

The camera geometry can be checked against the code without a simulator:

`FX`, `FY` and the pitch are computed from the field of view and from radians,
so they are rounded here; without the rounding they print as `640.0000000000001`
and `-14.999999999999998`.

```console
$ python -c "from common.geometry import FX, FY, CX, CY, CAM_OFFSET_BODY, CAMERA_PITCH_OFFSET; \
import numpy as np; print(round(FX, 6), round(FY, 6), CX, CY, CAM_OFFSET_BODY, round(np.degrees(CAMERA_PITCH_OFFSET), 6))"
640.0 640.0 640.0 360.0 [ 0.35  0.   -0.5 ] -15.0
```

### Provenance of this file, stated plainly

The live `settings.json` from the collection machine was not recoverable when this
repository was reorganised. The file shipped here is transcribed from
[`docs/phase1-technical-documentation.md` §4.1](../docs/phase1-technical-documentation.md),
which records it verbatim. How much of that transcription is independently
corroborated, and how much is not, stated field by field:

* **Corroborated by a constant in the canonical code.** The camera lever arm and
  tilt — `CAM_OFFSET_BODY = [0.35, 0.0, -0.5]` and
  `CAM_PITCH_RAD = radians(-15)`, `dataset_generation/collect_airsim.py` lines
  195–196 — and the four spawn origins, `VEHICLE_ORIGINS`, line 213. Both read
  their defaults from `default.yaml`.
* **Corroborated by the dissertation, not by code.** The 1280 × 720 capture at
  90°, from which §7.2.1 derives *f<sub>x</sub>* = *f<sub>y</sub>* = 640 px and
  (640, 360) px; and the LiDAR's 64 channels, stated in §7.2.1 and §7.2.5.
* **Not corroborated by anything here.** Every other LiDAR field —
  `PointsPerSecond`, `RotationsPerSecond`, `VerticalFOVUpper`/`Lower`,
  `HorizontalFOVStart`/`End`, `Range`, `DataFrame`. The only LiDAR value the
  canonical code contains is the sensor's *name*, `"LidarFront"`, in
  `dataset_generation/collect_airsim.py:1307`; nothing in the code reads or
  asserts a channel count, a point rate, a field of view or a range. Those
  numbers rest on the transcription alone.

One deliberate deviation from that transcription, disclosed here rather than
silently applied: `"VehicleType": "SimpleFlight"` was added to Drone3, Drone4 and
Intruder1. The transcription in the documentation omits it, and AirSim refuses to
start a vehicle without it, so the real file must have carried it. Nothing else
was changed, added, or normalised.

A different `settings.json` exists on the machine where the dissertation is being
written, at `~/Documents/AirSim/settings.json`: a 16-channel LiDAR named `Lidar1`
at 200,000 points/s, a camera at X 0.15 / Z −0.05 with zero pitch, and a single
vehicle named `Drone1`. It matches neither Chapter 7 nor any constant in this
code, so it was not shipped. If you have the original collection-machine file,
replace `settings.json` with it.

The LiDAR configuration has more than one value on record in this project, which
is worth naming before anyone reproduces it. Chapter 7 §7.2.1 and §7.2.5 and
`docs/phase1-technical-documentation.md` §4.1 all state 64 channels, and that is
what ships here. [`docs/lidar-config.md`](../docs/lidar-config.md) proposes
*raising* the count from 64 to 128, and widening the vertical FOV from a
symmetric ±45° to an asymmetric −20°/+70°. Neither change is in the shipped file,
whose LiDAR is 64-channel with a ±30° vertical FOV — so that document's own
starting point does not match it either. (It discusses only 64 and 128 channels;
the 16-channel figure comes from the dissertation machine's file described above,
not from it.) The
LiDAR is an auxiliary, visualisation-only channel — the detectors consume the
depth-camera cloud — so none of this affects any reported result, but it has not
been resolved against the original.

## `default.yaml`

Every value can be overridden, in increasing priority:

1. `configs/default.yaml` itself;
2. a YAML named by the `AIRSIM_UAV_CONFIG` environment variable;
3. `--config path/to/other.yaml` on scripts that accept it;
4. a specific flag: `--host`, `--port`, `--data-root`, or the environment
   variables `AIRSIM_HOST`, `AIRSIM_PORT`, `AIRSIM_UAV_DATA_ROOT`.

`airsim.host` defaults to `127.0.0.1`, which is correct when the client and the
simulator run on the same machine. The Chapter 7 experiments ran the Python
client under WSL2 and the simulator on the Windows host; in that arrangement the
host address is the WSL default gateway, which changes between reboots:

```console
$ ip route show | awk '/^default/ {print $3}'
172.x.y.1
$ export AIRSIM_HOST=172.x.y.1
```

The simulator side must be started with `"LocalHostIp": "0.0.0.0"` for a
non-loopback client to connect; `settings.json` already sets this. See
[`docs/setup/wsl.md`](../docs/setup/wsl.md).

`paths.data_root` (default `.`, the working directory) is the root under which
`datasets/`, `sessions/`, `runs/` and `dumps/` are resolved. Nothing in this
repository stores an absolute path.
