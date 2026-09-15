#!/usr/bin/env python3
"""Measure how much the *whole frame* changes between consecutive frames.

Metric: for each consecutive pair of frames, the mean absolute difference of the
8-bit greyscale image after downsampling to 160x90. Downsampling averages away
JPEG blocking and sensor noise; what survives is gross scene change --
re-exposure, a camera nudge, a cut between recordings.

What this metric cannot do
--------------------------
**A flat band close to zero does not mean nothing moved.** An earlier pass
through this material read it that way, concluded the indoor sequence was
static, and was wrong; the retraction is in the README under "The indoor
sequence: the empty detections file is a real miss", and in ``metrics/
findings.csv`` as F20.

The arithmetic is unforgiving. A quadrotor 36 x 20 px across in a 1920 x 1080
frame covers 0.035 % of the pixels. Even if every one of those pixels flipped by
the full 255 grey levels between two frames, a whole-frame mean absolute
difference can move by at most 0.09 grey levels -- below the frame-to-frame
noise floor of both sequences measured here. The statistic is *arithmetically
incapable* of registering a target this small, so its verdict on such a target
carries no information either way.

Both sequences in this study come out flat by this metric, and in both an
aircraft is flying. Use ``locate_moving_target.py``, which subtracts a
background plate and looks at one blob instead of the whole frame, when the
question is whether a small target moved. Use this script for what it can see:
whether two frames came from the same camera pointed at the same scene.

Mixed frame dumps
-----------------
The script refuses to run across frames of different pixel sizes. A dump that
holds two recordings will otherwise report the seam between them as its largest
"motion". Run ``partition_frame_dump.py`` first and pass ``--size`` to select
one recording.

Example
-------
    python frame_motion_stats.py --images output_images \
        --out results/indoor-transfer/frame-motion-indoor-948.csv
"""

import argparse
import csv
import glob
import os
from collections import Counter

import numpy as np
from PIL import Image

THUMB = (160, 90)


def overlay_cells(image, thumb):
    """Analysis cells touched by the detector's pure-green annotation overlay.

    ``detect_in_rosbag.py`` draws boxes and labels in OpenCV green (0, 255, 0).
    Those pixels are not scene content: they appear and vanish with the
    detector's output, and a pair of frames that differ only by a box drawn on
    one of them would otherwise register as motion.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (green >= 200) & (red <= 100) & (blue <= 100)
    if not mask.any():
        return np.zeros((thumb[1], thumb[0]), dtype=bool)
    as_image = Image.fromarray((mask.astype(np.uint8) * 255))
    # BOX resampling averages over each cell, so any green pixel inside a cell
    # gives it a non-zero value.
    return np.asarray(as_image.resize(thumb, Image.BOX)) > 0


def measure(files, thumb=THUMB, mask_overlay=False):
    """Return (rows, diffs). rows is one record per frame; the first frame has
    an empty difference because it has no predecessor."""
    rows = []
    diffs = []
    prev = None
    prev_mask = None
    for index, path in enumerate(files):
        image = Image.open(path)
        arr = np.asarray(image.convert("L").resize(thumb), dtype=np.int16)
        mask = overlay_cells(image, thumb) if mask_overlay else None
        if prev is None:
            diff = ""
            excluded = ""
        else:
            absolute = np.abs(arr - prev)
            if mask_overlay:
                keep = ~(mask | prev_mask)
                excluded = int((~keep).sum())
                diff = round(float(absolute[keep].mean()), 4)
            else:
                excluded = ""
                diff = round(float(absolute.mean()), 4)
            diffs.append(diff)
        row = [index, os.path.basename(path), diff]
        if mask_overlay:
            row.append(excluded)
        rows.append(row)
        prev = arr
        prev_mask = mask
    return rows, np.asarray(diffs, dtype=float)


def select(files, size):
    """Keep only frames of one pixel size, and refuse a mixed dump otherwise."""
    sizes = [Image.open(path).size for path in files]
    counts = Counter(sizes)
    if size:
        wanted = tuple(int(v) for v in size.lower().split("x"))
        kept = [path for path, s in zip(files, sizes) if s == wanted]
        if not kept:
            raise SystemExit(
                "no frames of size %dx%d; this dump holds %s"
                % (wanted[0], wanted[1], dict(counts))
            )
        return kept, wanted
    if len(counts) > 1:
        raise SystemExit(
            "this directory holds frames of %d different sizes (%s), so it "
            "holds more than one recording. Run partition_frame_dump.py and "
            "then pass --size WxH to measure one of them."
            % (len(counts), dict(counts))
        )
    return files, sizes[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--images", required=True, help="directory of frames, sorted by filename"
    )
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--pattern", default="*.jpg", help="glob (default: *.jpg)")
    ap.add_argument(
        "--size",
        help="select only frames of this pixel size, e.g. 1920x1080. Required "
        "when the directory holds more than one recording.",
    )
    ap.add_argument(
        "--mask-overlay",
        action="store_true",
        help="exclude analysis cells covered by the detector's green "
        "annotation overlay in either frame of a pair",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="grey levels out of 255; pairs above it are counted as moving "
        "(default: 2.0)",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, args.pattern)))
    if len(files) < 2:
        raise SystemExit("need at least 2 frames in %s" % args.images)
    files, size = select(files, args.size)
    if len(files) < 2:
        raise SystemExit("need at least 2 frames of the selected size")

    rows, diffs = measure(files, mask_overlay=args.mask_overlay)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = ["index", "filename", "mean_abs_diff_vs_prev_gray8"]
    if args.mask_overlay:
        header.append("cells_excluded_as_overlay")
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    above = int((diffs > args.threshold).sum())
    print("source            %s" % args.images)
    print("frame size        %dx%d" % size)
    print("frames            %d" % len(files))
    print("consecutive pairs %d" % diffs.size)
    if args.mask_overlay:
        print("overlay cells     excluded from every pair")
    print("mean              %.4f" % diffs.mean())
    print("median            %.4f" % float(np.median(diffs)))
    print("p95               %.4f" % float(np.percentile(diffs, 95)))
    print("max               %.4f" % diffs.max())
    print(
        "pairs > %.1f/255    %d  (%.1f%%)"
        % (args.threshold, above, 100.0 * above / diffs.size)
    )
    print("wrote             %s" % args.out)


if __name__ == "__main__":
    main()
