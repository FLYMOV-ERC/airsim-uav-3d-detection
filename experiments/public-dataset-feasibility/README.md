# Public-dataset feasibility study

This directory records an experiment that was run **before** the synthetic AirSim
dataset was built, to answer one question:

> Does a public dataset already exist that would supervise an airborne-target
> detector, so that generating a dataset in simulation is unnecessary?

Three routes were tried: two public datasets (VisDrone2019-DET and NTU VIRAL) and
one real recording made with a Holybro quadrotor. The answer was no, for reasons
set out below. This subtree is the evidence for that answer. It is distilled from
roughly 2.2 GB of working material down to about 2.6 MB by keeping only the
scripts, the configuration files, the metric tables and four figures, and by
dropping every frame dump, every checkpoint and every third-party file.

Every number below is produced by a script in `scripts/`, run against the frame
dumps and run directories named in the command. The commands are in
[Reproducing the tables](#reproducing-the-tables). Where a claim could not be
re-derived from a file that is either in this tree or named explicitly, it has
been removed.

**Where this sits in the dissertation.** The detection chapter (Chapter 7)
describes the AirSim synthetic dataset and, in the subsection *The
Automatic-Labeling Problem and the Synthetic Solution*, the three in-simulator
labelling routes that were attempted and abandoned. This study is the question
that comes before those: whether any dataset had to be generated at all. Nothing
here was used to train the pipeline that Chapter 7 evaluates. It is a negative
result that motivates the chapter.

---

## The finding, in short

The load-bearing part of the conclusion is **categorical**, and it does not depend
on any training run converging or failing:

- **VisDrone2019-DET contains no airborne-vehicle class at all.** Its ten classes
  are `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`,
  `awning-tricycle`, `bus`, `motor`. Every one of them is a ground traffic
  participant. VisDrone is imagery captured *from* drones, not imagery *of*
  drones. It cannot supply a single positive example of the target class.
- **NTU VIRAL's ground truth is the carrying vehicle's own pose**, surveyed by a
  Leica instrument tracking a prism mounted on the aircraft
  (`gndtr_topic: /leica/pose/relative`, with a static `T_Body_Prism` giving the
  prism's offset from the body frame). It is ego-pose ground truth for odometry
  and SLAM benchmarking. There are no annotations of any other object in the
  scene, so there is nothing for a detector to learn from.

Neither dataset can supervise a drone detector. That holds regardless of what any
training run did.

The third route — labelling a real recording — did produce a working detector,
but on fewer than 500 boxes from a single session, and it detected nothing at all
when pointed at a different scene. The details, and the caveats that go with
each number, are below.

---

## Route 1 — VisDrone2019-DET

Thirteen training runs were configured against VisDrone
(`results/run-args/yolov8_visdrone*.yaml`, indexed in `results/run-index.csv`).

**What the run index shows.** Twelve of the thirteen never completed a single
epoch: Ultralytics writes `results.csv` only after an epoch finishes, and those
twelve directories contain an `args.yaml` and nothing else — no metrics, no
checkpoint. Their `data:` fields take five different spellings of the dataset
descriptor across the runs (`./visdrone.yaml`,
`./VisDrone2019-DET-train/data.yaml`, `./VisDrone2019-DET/data.yaml`,
`./dataset/VisDrone2019-DET/data.yaml`,
`./datasets/VisDrone2019-DET/data.yaml`), which is the signature of dataset-layout
debugging rather than of experiments.

**The one run that trained.** `yolov8_visdrone53` reached **7 of 100 requested
epochs** in 4045.35 s on Apple MPS — about 578 s per epoch, so roughly 16 hours
projected for the full schedule — and was abandoned there. Over those seven
epochs the training losses did fall (box 5.44231 → 4.15381, classification
6.59981 → 3.89212), so the optimiser was running, but validation mAP50 never left
the noise floor: it peaked at 0.00038 in epoch 1 and was 0.00008 by epoch 7.

> **Caveat that must travel with those numbers.** Every run in this study was
> built as `YOLO('yolov8n.yaml')` — the *architecture* definition, which starts
> from random weights — while also passing `pretrained=True`, which Ultralytics
> honours only when the model is built from a checkpoint such as `yolov8n.pt`.
> See `model: yolov8n.yaml` in `results/run-args/yolov8_visdrone53.yaml`. So this
> was a from-scratch run aborted after 7 of 100 epochs. VisDrone trains perfectly
> well in the literature. **The near-zero mAP is weak evidence and proves nothing
> about VisDrone.** The reason VisDrone was rejected is the categorical one above:
> no airborne class. The training numbers are published here only so the record
> is complete.

**One observation about the local label conversion, not about the dataset.** The
class histogram in `results/visdrone/yolov8_visdrone53_labels.jpg` is dominated by
`van` (roughly 34,000 instances) and `people` (roughly 14,800), with `car` under
1,000. That is not the distribution VisDrone2019-DET is documented to have, and
the likely explanation is that the local label conversion mis-mapped class
indices. This says something about how the data was prepared locally; it says
nothing about VisDrone itself. The list of class *names* in that same figure is
what matters here, and it is correct.

## Route 2 — NTU VIRAL

Only the calibration files of the `eee_03` sequence were obtained and inspected.
Two facts settled it, both quoted here rather than redistributed (the files are
third-party; see the reference below):

- `leica_prism.yaml` gives `gndtr_topic: "/leica/pose/relative"` together with a
  static `T_Body_Prism` transform. The "ground truth" is the pose of the aircraft
  carrying the sensors, not the pose of anything it observes.
- `lidar_horz.yaml` gives `VERT_RES: 16` and `HORZ_RES: 1024` — a 16-beam lidar,
  which at the ranges of interest for an airborne target returns too few points
  to define a 3D box even if annotations had existed.

The dataset is built for visual-inertial-ranging-lidar odometry benchmarking, and
it does that well. It is simply not an object-detection dataset.

Image data from `eee_03` was also, at some point, pushed through the detector;
596 annotated frames of it survive in the working material. Those frames are
third-party and are not published here, and — for the reason set out in
[A contaminated frame dump](#a-contaminated-frame-dump) — no statistic in this
study is computed over them.

## Route 3 — a real Holybro recording

The remaining option was to record and label data directly. A quadrotor was flown
at an outdoor site and recorded to a rosbag; frames were extracted
(`scripts/extract_frames_from_rosbag.py`), labelled with a single class `drone`,
and used to train a detector (`yolov8_holybro3`).

**The headline number looks excellent and should not be believed.**
`results/holybro/yolov8_holybro3_results.csv` records 100 of 100 epochs in
3875.68 s with final precision 0.99939, recall 1.000, **mAP50 0.995** and
mAP50-95 0.63815.

Two artifacts in this directory explain why that is near-memorisation rather than
generalisation:

- `results/holybro/yolov8_holybro3_labels.jpg` — one class, with the instance
  histogram reaching just under 500. The box centres are confined to x in
  [0.45, 0.59] and y in [0.44, 0.56], and the box sizes to w in [0.014, 0.038]
  and h in [0.019, 0.053] of the image. (Those five bounds are read off the axes
  of that shipped figure, not recomputed from labels — the label files are gone;
  see [Unknowns](#unknowns).) A single narrow scale band means a single narrow
  range band: the detector was only ever shown the target at one apparent size,
  in one part of the frame.
- The train/val split, read off Ultralytics' own `train_batch0` and
  `val_batch0_pred` mosaics, whose tiles carry the source frame numbers. Training
  frames run 000103–000593; validation frames are 000001 and 000335–000411, i.e.
  **interleaved with the training range and drawn from the same continuous
  recording of the same scene**. Train and validation are neighbouring frames of
  one session, so mAP50 0.995 measures fit to that session. (Those two mosaics are
  not redistributed here — see [Redaction and privacy](#redaction-and-privacy).
  They are regenerated by any training run; the frame numbers above are the whole
  of what they establish.)

**It does fire on its own site.** Of the 1217 frames of the annotated outdoor
pass, **994 (81.7 %)** carry a green annotation overlay — the detector drew a
`drone` box on four frames in five of the recording it was trained on.
`scripts/count_overlay_frames.py` produces that count from the frame dump alone
(`results/indoor-transfer/overlay-frames-holybro2.csv`); the per-frame
green-pixel count is bimodal, either zero or several hundred, so the count does
not depend on where the threshold is put. `figures/outdoor-detection-frame.jpg`
is one such frame, `drone 0.70`.

**The flight volume was also far too small.**
`figures/holybro-out01-mocap-trajectory.png` is the VRPN motion-capture trajectory
from `HolybroOut01.bag`, which is the only place in this study where usable 6-DoF
ground truth existed. `scripts/measure_plot_extent.py` measures the plotted trace
against its own tick spacing and returns **11.53 m by 6.04 m** horizontally — an
upper bound, since the bounding box of a drawn trace includes the line and marker
width. An indoor-arena volume, an order of magnitude below the target-range
regime the detection chapter evaluates.

---

## A contaminated frame dump

Before any of the sequence statistics below can be read, one thing has to be said
about where they come from.

`output_images_holybro2/` looks like one recording of 1813 frames. It is two.
`scripts/partition_frame_dump.py` segments it on pixel size:

| frames | size | neutral-pixel fraction | source |
|---|---|---|---|
| 0–1216 (1217 frames) | 1920 × 1080 | 0.157 | the Holybro outdoor pass |
| 1217–1812 (596 frames) | 752 × 480 | 1.000 | not this recording |

The second segment is third-party. Its 752 × 480 resolution is exactly the
`image_width` / `image_height` of `eee_03/camera_left.yaml` in NTU VIRAL; its
frames are perfectly neutral, as a monochrome visual-inertial camera's are; and
the scene in them is a curved multi-storey building carrying `EEE` signage,
which is the NTU School of Electrical and Electronic Engineering block the
`eee_03` sequence is named after. The two dumps merged because frames are
written as `frame_%06d.jpg` by message index, so a second pass into the same
output directory overwrites the first only where the indices overlap and leaves
the longer run's tail behind.

**What this invalidated.** An earlier version of this README reported the
"outdoor sequence" as 1813 frames with mean 2.255, max 58.417 and 508 of 1812
pairs above the 2/255 threshold, and read those numbers as gross scene and
exposure change. They were an artifact of the merge:

- the maximum, 58.417, is the pair `frame_001216 → frame_001217` — the seam
  between the two datasets, not motion;
- of the 508 above-threshold pairs, **507 lie wholly inside the foreign
  segment** and the remaining one is the seam pair itself; none is in the
  Holybro footage;
- the Holybro segment on its own has **zero** pairs above the threshold.

It also inflated a count of detector firings: the reported "1059 of 1813 frames
(58.4 %)" is 994 of 1217 on the Holybro site added to 65 of 596 on NTU imagery.
The second term does not belong. Those 65 frames are not the drone detector
firing at all — their labels read `person`, `train`, `elephant`, `sheep`, which
are COCO classes, so they were produced by a stock 80-class checkpoint and not by
the single-class `drone` model. Only the Holybro figure, 994 of 1217, is
reported.

Every sequence statistic in this README is now computed over one recording at a
time. `frame_motion_stats.py` and `locate_moving_target.py` both refuse to run
across a directory holding more than one frame size.

---

## The indoor sequence: the empty detections file is a real miss

`results/indoor-transfer/detections.csv` is 42 bytes: a header row and nothing
else. The trained detector produced **zero detections over all 948 frames** of an
indoor sequence recorded at a different site. The frame dump agrees with the CSV:
`count_overlay_frames.py` finds an annotation overlay on **0 of those 948
frames**, so no detection was dropped between the pass and the file.

> **Correction to an earlier reading of this material.** A first pass concluded
> that the indoor sequence was *static* — that no aircraft ever flew in front of
> the camera — and therefore that the empty file was an unusable-data finding
> rather than a detection failure. **That conclusion was wrong and is retracted
> here.** It rested on a whole-frame motion statistic that is provably blind to a
> target of this size. A targeted measurement, described below, shows the
> sequence contains a complete quadrotor flight. The whole-frame CSVs are kept in
> this directory because the numbers in them are correct; what was wrong was the
> inference drawn from them.

### What the whole-frame statistic says, and why it is the wrong instrument

`scripts/frame_motion_stats.py` measures, for each consecutive pair of frames, the
mean absolute difference of the 8-bit greyscale image downsampled to 160×90:

| sequence | frames | pairs | mean | median | p95 | max | pairs > 2/255 |
|---|---|---|---|---|---|---|---|
| indoor (`HolybroStdn01`) | 948 | 947 | 0.3883 | 0.3853 | 0.3981 | 0.9426 | 0 (0.0 %) |
| outdoor, Holybro segment | 1217 | 1216 | 0.3582 | 0.3539 | 0.3967 | 0.4717 | 0 (0.0 %) |
| outdoor, overlay cells excluded | 1217 | 1216 | 0.3500 | 0.3501 | 0.3573 | 0.3630 | 0 (0.0 %) |

Per-frame values are in `results/indoor-transfer/frame-motion-indoor-948.csv`,
`frame-motion-outdoor-holybro-1217.csv` and
`frame-motion-outdoor-holybro-1217-overlay-masked.csv`.

**Both sequences are flat by this metric.** That is the point, and it is the
opposite of what this README used to claim. Once the foreign frames are removed,
the outdoor row has a maximum of 0.4717 grey levels and not one of its 1216 pairs
reaches the 2/255 threshold — the same verdict the metric returns on the indoor
sequence. There is no indoor-versus-outdoor contrast here to read anything into.
Both sequences contain an aircraft in flight; the metric reports both as flat;
therefore the metric cannot see this target. No replacement contrast is offered,
because there is none to offer.

The indoor distribution is tight but not as tight as previously stated: 899 of
947 pairs fall in the band 0.369–0.398, 48 exceed 0.398, nine exceed 0.5, and two
reach 0.91 and 0.94. The earlier description, "a flat band of 0.385–0.398 with a
single excursion to 0.943", understated the tail.

**It is not a measurement that can see a drone.** The target in these frames is a
median 36 × 20 px box in a 1920 × 1080 image — 0.035 % of the pixels. At the
160×90 analysis resolution that is about 5 pixels of 14,400. Even if every one of
them flipped by the full 255 grey levels between two frames, the frame-wide mean
absolute difference would move by **less than 0.1 grey levels**, which is below
the frame-to-frame noise floor of either sequence. A whole-frame mean is
arithmetically incapable of registering this target, so "the frame mean did not
change" carries no information about whether the target moved.

**The annotation overlay, measured rather than bounded.** The outdoor frames are
the detector's own annotated output, so a green box appears and disappears across
them; that is a confound in a frame-difference statistic, and it is not
negligible by arithmetic alone. The largest overlay in the Holybro segment covers
3048 px of 2,073,600 (0.147 %), which bounds its contribution at 0.375 grey
levels — the same order as the whole row. So it was measured directly instead:
excluding every analysis cell touched by the overlay in either frame of a pair
(`frame_motion_stats.py --mask-overlay`) moves the mean from 0.3582 to 0.3500 and
the maximum from 0.4717 to 0.3630. The overlay is indeed the largest single
contributor to what little spread the row has — the largest excursion in the
whole segment, the pair at index 114, drops from 0.4717 to 0.3529 once it is
removed — and the conclusion is unchanged either way: nothing in this sequence
comes within a factor of four of the threshold.

*(A previous version of this section bounded the overlay at 0.111 grey levels
from "902 green pixels of 2,073,600" counted in `outdoor-detection-frame.jpg`.
That does not reproduce, and it could not: the published figure is 1280 × 720 and
downscaled, so a pixel count taken from it says nothing about the 1920 × 1080
frames the statistic was actually computed over. The claim is withdrawn and
replaced by the measurement above.)*

### What a measurement sensitive to the target says

`scripts/locate_moving_target.py` does the right measurement for a fixed camera:
build a per-pixel temporal median background plate, subtract it, and localise the
largest connected foreground blob in each frame. Output:
`results/indoor-transfer/indoor-target-track.csv`.

- A foreground blob is present in **773 of the 948 frames** (81.5 %).
- Its vertical centroid rests at **y ≈ 595 px** — on the floor — through the
  opening of the sequence, climbs to a minimum of **y = 279 px** near the ceiling,
  and returns to the floor by the end. That is a takeoff, a climb, a hover, a
  descent and a landing.
- **447 frames** have the centroid above y = 550 px, i.e. the aircraft clearly
  airborne. Every one of them lies between **frame 326 and frame 843**; of the 518
  frames in that window, the other 71 either have no accepted blob or show it
  below the line. Outside 326–843 there is not one airborne frame.
- The target's apparent size is a median **36 × 20 px**, so a normalised width of
  **0.0187** — squarely inside the detector's training band of
  w ∈ [0.014, 0.038].

**Two plates, two different questions.** The 773 above is measured against a
median over the *whole* sequence, so it counts every frame containing something
that is not part of the sequence-long background — including the aircraft sitting
still on the floor. It is a foreground-presence count, not a motion count. Run
with a *local* plate instead — the median of a 61-frame window centred on each
frame, which absorbs anything that holds still for longer than the window — the
count is **419 of 948** (44.2 %), of which 359 are above the line, spanning frames
351–823 (`indoor-target-track-local-plate.csv`). That second number is the motion
count, and it is strongly window-dependent, so the window has to be quoted with
it. Neither number is the other; both are reported because the earlier version of
this README quoted the first as though it were the second.

**Minimum box size.** The track applies `--min-box-width 2` (2 analysis pixels,
8 px at full resolution). Without it the airborne count is 455 over frames
326–855, and the extra eight frames — 846, 847, 849–852, 854, 855 — are all the
same narrow vertical sliver at a fixed x = 1472 px: 4 × 32 px in seven of them and
4 × 28 px in frame 846. Enlarging the frames shows it to be a window blind
catching the light. A quadrotor seen from this camera
is wider than it is tall (median 36 × 20 px), so a blob one analysis pixel wide
cannot be the target, and excluding it by construction is better than deleting
the frames afterwards. `--min-box-width 0` reproduces the unfiltered 455/326–855.

`figures/indoor-flight-strip.jpg` shows nine frames (indices 0, 200, 400, 500,
550, 600, 700, 800, 900), each cropped around the measured centroid with the
measured box drawn on it, and confirms by eye that the blob is the quadrotor at
every stage of the flight. `figures/indoor-arena-frame.jpg` is the same scene in
full: frame 400, with the aircraft airborne in front of the green screen at the
far end of the hall.

### The conclusion that follows

A detector that scores mAP50 0.995 on held-out frames of its own recording
session, and that fires on 994 of the 1217 frames of that session, produced
**zero detections across 447 frames of a quadrotor airborne at exactly the
apparent size it was trained on**, once the scene changed. That is a
generalisation failure, and it is the near-memorisation diagnosis of Route 3
confirmed on data the detector had never seen.

What this does **not** establish, stated so nobody over-reads it:

- **Which checkpoint ran.** `detect_in_rosbag.py` defaults to
  `runs/detect/yolov8_holybro3/weights/best.pt`, which is the natural reading, but
  the pass was not otherwise logged and the weights are not shipped here. The run
  cannot be re-scored in this directory without retraining.
- **How many frames a human would have labelled.** The 773 and 447 figures come
  from the background-subtraction tracker, checked by eye on nine frames. They are
  a measurement, not hand annotation.
- **Whether a lower confidence threshold would have found anything.**
  `detections.csv` records the Ultralytics default; no threshold sweep was run.
- **Why it missed.** Scene, lighting, background clutter and camera all changed
  between the two recordings. Which of those the detector failed on is not
  separable from this data.

---

## Why a synthetic dataset followed

Putting the three routes together:

1. VisDrone2019-DET has no airborne-vehicle class, so it cannot provide a single
   positive example of the target. Categorical.
2. NTU VIRAL provides ego-pose ground truth for odometry, not annotations of
   observed objects, so it cannot supervise a detector either. Categorical.
3. Labelling a real recording produced, in practice, fewer than 500 boxes from one
   session, in one scene, in one narrow scale band, with train and validation
   frames interleaved — and the resulting detector, which fires on 81.7 % of the
   frames of its own session, found nothing at all on the first sequence from a
   different scene, despite the target appearing there at the size it was trained
   on. The only recording with usable 6-DoF ground truth covered an
   11.53 m × 6.04 m volume. Scaling this route to the scene diversity, range
   diversity and annotation volume a 3D detector needs means many more flight
   campaigns and a labelling effort that scales with them — and it still yields no
   3D boxes.

What was actually needed was a large set of frames with an airborne target at
known range, across varied scenes, poses and scales, with **exact** 2D and 3D
boxes. None of the three routes supplies that. A simulator does: the target's pose
is known exactly, so the label can be constructed geometrically instead of
annotated by hand, at whatever scale is required and at negligible cost per frame.
That is the reasoning behind the AirSim dataset described in Chapter 7 — and the
labelling problem does not disappear inside the simulator either, which is what
the chapter's subsection on automatic labelling addresses.

---

## Layout

```
public-dataset-feasibility/
├── README.md                                  this file
├── scripts/
│   ├── train_yolov8.py                        training entry point for all 16 runs
│   ├── extract_frames_from_rosbag.py          rosbag topic -> JPEG frames
│   ├── detect_in_rosbag.py                    trained detector over a rosbag -> detections.csv
│   ├── plot_mocap_trajectory.py               VRPN pose topic -> trajectory figure
│   ├── summarize_runs.py                      run directories -> results/run-index.csv
│   ├── partition_frame_dump.py                frame dump -> the recordings it actually holds
│   ├── frame_motion_stats.py                  whole-frame motion (coarse; see the caveat above)
│   ├── locate_moving_target.py                background subtraction -> per-frame target track
│   ├── count_overlay_frames.py                annotated dump -> which frames the detector fired on
│   ├── measure_plot_extent.py                 rendered plot -> extent of the plotted trace
│   ├── redact_signage.py                      blur a third party's signage out of a figure
│   └── docker/
│       ├── Dockerfile                         ROS 1 Noetic + cv_bridge + rosbag, CPU-only torch
│       └── docker-compose.yml                 mounts the project root at /workspace
├── results/
│   ├── run-index.csv                          all 16 runs in one table
│   ├── run-args/                              the 16 args.yaml, flattened and renamed
│   ├── holybro/                               the one fully trained run (100 epochs)
│   ├── visdrone/                              the one VisDrone run that produced metrics
│   ├── redaction/                             the blur boxes applied to the three figures
│   └── indoor-transfer/
│       ├── detections.csv                     42 bytes: header only, 0 detections / 948 frames
│       ├── indoor-target-track.csv            the flight the detector missed (global plate)
│       ├── indoor-target-track-local-plate.csv  the same, against a 61-frame local plate
│       ├── frame-motion-indoor-948.csv        whole-frame motion, indoor
│       ├── frame-motion-outdoor-holybro-1217.csv          whole-frame motion, outdoor
│       ├── frame-motion-outdoor-holybro-1217-overlay-masked.csv   the same, overlay cells excluded
│       ├── frame-dump-segments-holybro2.csv   per-frame size and neutrality: the contamination
│       └── overlay-frames-holybro2.csv        per-frame green-pixel count: which frames fired
├── figures/
│   ├── indoor-arena-frame.jpg                 frame 400: the hall, aircraft airborne
│   ├── indoor-flight-strip.jpg                nine frames of the flight, with measured boxes
│   ├── outdoor-detection-frame.jpg            detector firing on its own site, "drone 0.70"
│   └── holybro-out01-mocap-trajectory.png     11.53 m x 6.04 m flight volume
└── metrics/
    └── findings.csv                           claim -> value -> evidence ledger
```

`metrics/findings.csv` is the quickest way in. Each row carries a claim, its exact
value, the script that produces it where a script does, the file that backs it,
and a strength label.
The rows marked `caveat` and `correction` exist to stop the weaker rows from being
quoted more strongly than the evidence allows.

## Reproducing the tables

Every table and figure caption in this README comes from one of the commands
below. `summarize_runs.py` needs only a directory of Ultralytics run folders;
`measure_plot_extent.py` needs only a file already in this tree; the rest need a
directory of frames.

**Always partition a frame dump before measuring it.** That is what the first
command is for, and it is not optional — see
[A contaminated frame dump](#a-contaminated-frame-dump).

```sh
python scripts/partition_frame_dump.py \
    --images /path/to/output_images_holybro2 \
    --out results/indoor-transfer/frame-dump-segments-holybro2.csv

python scripts/count_overlay_frames.py \
    --images /path/to/output_images \
    --out results/indoor-transfer/overlay-frames-indoor.csv
    # F15b: 0 of the 948 indoor frames carry a drawn detection overlay

python scripts/locate_moving_target.py \
    --images /path/to/extracted_images --min-box-width 2 \
    --out results/indoor-transfer/extracted-531-track.csv
    # the 531-frame extraction: mean 0.3117, max 0.3271, 82 of 531 with a blob

python scripts/summarize_runs.py \
    --runs /path/to/runs/detect \
    --out results/run-index.csv

python scripts/frame_motion_stats.py \
    --images /path/to/indoor_frames \
    --out results/indoor-transfer/frame-motion-indoor-948.csv

python scripts/frame_motion_stats.py \
    --images /path/to/output_images_holybro2 --size 1920x1080 \
    --out results/indoor-transfer/frame-motion-outdoor-holybro-1217.csv

python scripts/frame_motion_stats.py \
    --images /path/to/output_images_holybro2 --size 1920x1080 --mask-overlay \
    --out results/indoor-transfer/frame-motion-outdoor-holybro-1217-overlay-masked.csv

python scripts/count_overlay_frames.py \
    --images /path/to/output_images_holybro2 \
    --out results/indoor-transfer/overlay-frames-holybro2.csv

python scripts/locate_moving_target.py \
    --images /path/to/indoor_frames \
    --out results/indoor-transfer/indoor-target-track.csv \
    --montage figures/indoor-flight-strip.jpg \
    --montage-indices 0,200,400,500,550,600,700,800,900

python scripts/locate_moving_target.py \
    --images /path/to/indoor_frames --background local --bg-window 61 \
    --out results/indoor-transfer/indoor-target-track-local-plate.csv

python scripts/measure_plot_extent.py \
    --figure figures/holybro-out01-mocap-trajectory.png --tick-spacing 2
```

`locate_moving_target.py` assumes a fixed camera and a single dominant moving
object, which holds for the indoor sequence and does not hold in general.

The four rosbag-facing scripts need ROS 1 (`rosbag`, `cv_bridge`), which has no
usable macOS build; `scripts/docker/` is the container that was used. It mounts
the project root — this directory — at `/workspace`, so the scripts are at
`/workspace/scripts/` inside the container. Put the bags anywhere under the
project root (`*.bag` is gitignored). Every script takes its inputs as
command-line arguments; the originals hardcoded absolute container paths.

```sh
cd scripts/docker && docker compose up -d && docker compose exec ros-yolo bash
# inside the container, at /workspace:
python3 scripts/extract_frames_from_rosbag.py --bag HolybroOut01.bag \
    --topic /camera/color/image_raw --out extracted_images
python3 scripts/plot_mocap_trajectory.py --bag HolybroOut01.bag \
    --topic /vrpn_client_node/holybro/pose --out trajectory.png
python3 scripts/detect_in_rosbag.py --bag HolybroStdn01.bag --weights best.pt \
    --out-images output_images_stdn01 --out-csv detections.csv
```

Give every pass its own `--out-images` directory. Sharing one is how the two
datasets in `output_images_holybro2/` came to be merged.

Training was run natively on the host against Apple MPS rather than inside the
container, because the container installs the CPU-only torch wheels. That is what
the `device: mps` field in `results/run-args/*.yaml` records.

```sh
python scripts/train_yolov8.py \
    --data ./datasets/holybro/yolo_dataset/data.yaml \
    --name yolov8_holybro3 --device mps
```

Note that `train_yolov8.py` still defaults to `--model yolov8n.yaml`, which starts
from random weights. That default is **deliberately preserved** so the command
above reproduces the recorded runs; it is also the caveat behind the VisDrone
numbers. Pass `--model yolov8n.pt` to start from the COCO checkpoint instead.

## What is deliberately not included, and why

Sizes are in MB of 10⁶ bytes, measured over the working tree.

| not included | why |
|---|---|
| The four frame dumps (3,302 JPEGs, 1,291 MB — of which 596 frames and 52 MB are third-party NTU VIRAL imagery, leaving 2,706 frames of this study's own recordings) | Bulk pixels. Everything they support is captured by `detections.csv`, the derived CSVs and four figures. The study's own frames are regenerable from the bags with `extract_frames_from_rosbag.py` and `detect_in_rosbag.py`; the NTU frames are not ours to publish in any case. |
| All `.pt` checkpoints (198.8 MB across the tree: 49.3 MB in the run directories, and 136.7 + 6.5 + 6.3 MB at its root) | A detector trained on fewer than 500 boxes from one session, and a VisDrone run abandoned at epoch 7. No reuse value. `run-args/` plus `results.csv` record everything needed to retrain. One 136.7 MB checkpoint matches no run in the training record at all and could not be documented honestly, so it is not published. |
| `yolov8n.pt` (6.5 MB, and counted in the 198.8 MB above) and `yolov8n.torchscript` (13.0 MB) | Upstream Ultralytics checkpoints. Fetch them, do not vendor them. |
| The NTU VIRAL `eee_03` calibration YAMLs, and the 596 annotated `eee_03` frames | Third-party files. The two facts that mattered are quoted above and cited to the original publication instead of redistributed. |
| VisDrone2019-DET itself | Third-party dataset; its licence does not permit rehosting. |
| The fourteen run directories that never trained, as directories | They hold an `args.yaml` and an empty `weights/` folder. The `args.yaml` files are kept flattened in `run-args/`; the empty scaffolding carries nothing. |
| The Python virtual environment (703 MB) | `scripts/docker/` is the real environment record. |
| Redundant Ultralytics plots (P/R/F1 curves, confusion matrices, correlograms, the remaining train/val batch mosaics) | Superseded by the PR curve, `results.png`, `labels.jpg` and the single train/val batch pair kept as the split-leakage evidence. Single-class confusion matrices in particular carry no information. |

## Redaction and privacy

Three images in this tree show real places. What was altered in each, and why:

**A third party's business signage — blurred.** The outdoor recordings were made
on commercial premises that are not the author's, and two signs on the building —
a roof wordmark and a board carrying a company name and web address — were
legible in `figures/outdoor-detection-frame.jpg` and identified the site, while
the README described it only as "an outdoor site". They carry no evidentiary
value — nothing here depends on which building the aircraft was flown past — so
they are blurred rather than merely disclosed. `scripts/redact_signage.py` did
it, pixelating before blurring so the information is destroyed rather than
smeared; the regions are shipped in `results/redaction/outdoor-detection-frame.json`
so the redaction can be inspected. The matcher is run liberally, so a good
fraction of the 19 regions are false positives on sky, foliage and asphalt, where
the blur is invisible and harmless: over-blurring a façade costs nothing, missing
a sign costs the point of the exercise.

Re-deriving the regions instead of reading them from the JSON needs an
unredacted source frame, which is not shipped. `--reference <frame> --template
x0,y0,x1,y1` does that, and the two template crops used were `820,230,1160,310`
and `695,395,945,495` of `output_images_holybro2/frame_000900.jpg`, the frame
`figures/outdoor-detection-frame.jpg` was made from.

**Two Ultralytics training mosaics — withheld rather than redacted.** Ultralytics
writes `train_batch0.jpg` and `val_batch0_pred.jpg` into every run directory. Both
were prepared for publication here and then dropped, because redacting them could
not be made reliable: the mosaic augmentation cuts each source frame at tile
boundaries, so a template matcher never fires on the partial wordmarks that
survive at the seams, and an audit found three such fragments still legible after
a hand pass. The same tiles also carry, unblurred, the upper-floor window band
that was blurred out of caution in `figures/outdoor-detection-frame.jpg`. Since
the only thing those mosaics establish is the train/val frame numbering — recorded
verbatim above and in `metrics/findings.csv` F13 — withholding them costs no
evidence and removes the whole class of risk. They regenerate on any training run.

**People — checked for, and two regions blurred.** Every published frame was
checked for people first. Two regions were blurred before downscaling: an
upper-floor window band in `figures/outdoor-detection-frame.jpg`, which contains
dark unidentifiable silhouettes behind glass, and a coat rack in the right-hand
corner of `figures/indoor-arena-frame.jpg`, whose hanging clothing is ambiguous at
that distance. Neither region is in the shipped signage box list, because both
were blurred by hand before the signage pass; the crops in
`figures/indoor-flight-strip.jpg` contain no people.

## Unknowns

These are recorded rather than guessed at.

- **Where the rosbags are.** `HolybroStdn01.bag` and `HolybroOut01.bag` are not in
  this tree and were not in the source material either — the original scripts
  referenced them at `/workspace/*.bag` inside the container. They are the only
  irreplaceable raw artifact of the real-recording route.
- **Where the labelled Holybro frames are, and how they were labelled.** The
  dataset directory `datasets/holybro/yolo_dataset/` that
  `results/run-args/yolov8_holybro*.yaml` points at is absent, which is why the
  label-distribution bounds above are read off the shipped figure rather than
  recomputed. If those labels came from projecting the VRPN motion-capture pose
  into the image rather than from hand annotation, that would itself be an
  automatic-labelling attempt on real data and belongs in the Chapter 7 narrative.
  Unknown.
- **Which checkpoint produced the outdoor and indoor inference passes.**
  `detect_in_rosbag.py` writes to a fixed output directory, so the outdoor pass is
  a renamed earlier run with no recorded provenance.
  `figures/outdoor-detection-frame.jpg` is therefore captioned as *an* inference
  pass on the outdoor recording, not as the output of a named checkpoint. The same
  gap applies to the indoor pass behind `detections.csv`. What is now certain is
  that the 596 NTU frames in the same directory came from a *different*
  checkpoint: their labels are COCO classes.
- **Which segment of which bag each frame dump came from.** Not recorded anywhere.
  A separate 531-frame extraction from `HolybroOut01.bag` measures at mean 0.3117,
  max 0.3271 and 0 of 530 pairs above 2/255 by the whole-frame metric — flat, like
  every other sequence here — yet `locate_moving_target.py` finds a foreground blob
  in 82 of its 531 frames, and enlarging the largest of them (index 427 of that
  extraction, `frame_001028.jpg`) shows the quadrotor in flight against the
  building wall. That is a third independent
  demonstration that the whole-frame metric is the wrong instrument for this
  target. Those frames are superseded by the annotated outdoor pass and are not
  shipped, so this particular measurement has no artifact in this directory.

## References

- P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu and H. Ling, "Detection and
  Tracking Meet Drones Challenge," *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, 2021. Dataset:
  <https://github.com/VisDrone/VisDrone-Dataset>
- T.-M. Nguyen, S. Yuan, M. Cao, Y. Lyu, T. H. Nguyen and L. Xie, "NTU VIRAL: A
  Visual-Inertial-Ranging-Lidar Dataset, From an Aerial Vehicle Viewpoint," *The
  International Journal of Robotics Research*, 2022. Dataset:
  <https://ntu-aris.github.io/ntu_viral_dataset/>
- Ultralytics YOLOv8: <https://github.com/ultralytics/ultralytics>
