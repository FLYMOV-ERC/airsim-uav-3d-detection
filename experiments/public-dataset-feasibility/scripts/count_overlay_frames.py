#!/usr/bin/env python3
"""Count the frames of a detector pass that carry an annotation overlay.

``detect_in_rosbag.py`` writes one annotated JPEG per frame and draws its boxes
and labels in OpenCV green, ``(0, 255, 0)``. Nothing else in these recordings is
that colour, so counting saturated-green pixels recovers, from the frame dump
alone, which frames the detector fired on -- useful when the detections CSV for
a pass was overwritten or never kept.

The criterion is stated rather than tuned: a pixel counts as overlay when
``G >= 200`` and ``R <= 100`` and ``B <= 100``. On the dumps in this study the
per-frame count is bimodal with nothing in between -- a frame has either zero
such pixels or several hundred -- so ``--min-green`` does not have to be chosen
carefully.

Frames are grouped by pixel size, because a dump can hold more than one
recording (see ``partition_frame_dump.py``) and a count over the mixture is
meaningless. Each group is reported separately.

The script also reports, per group, the largest overlay footprint as a fraction
of the frame and the bound that puts on a whole-frame mean absolute difference:
if the entire overlay appeared or vanished between two frames and every one of
its pixels swung the full 255 grey levels, the frame mean could move by at most
that much. It is an upper bound on the overlay as a confound in
``frame_motion_stats.py``; the direct measurement is that script's
``--mask-overlay``.

Example
-------
    python count_overlay_frames.py --images output_images_holybro2 \
        --out results/indoor-transfer/overlay-frames-holybro2.csv
"""

import argparse
import csv
import glob
import os
from collections import OrderedDict

import numpy as np
from PIL import Image


def green_pixels(path, g_min, rb_max):
    image = Image.open(path)
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (green >= g_min) & (red <= rb_max) & (blue <= rb_max)
    return image.size, int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="directory of annotated frames")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument(
        "--min-green",
        type=int,
        default=50,
        help="pixels above which a frame counts as carrying an overlay "
        "(default: 50)",
    )
    ap.add_argument("--g-min", type=int, default=200)
    ap.add_argument("--rb-max", type=int, default=100)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, args.pattern)))
    if not files:
        raise SystemExit("no frames matching %s in %s" % (args.pattern, args.images))

    rows = []
    groups = OrderedDict()
    for index, path in enumerate(files):
        size, count = green_pixels(path, args.g_min, args.rb_max)
        rows.append([index, os.path.basename(path), size[0], size[1], count])
        groups.setdefault(size, []).append(count)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "filename", "width", "height", "green_px"])
        writer.writerows(rows)

    print("source        %s" % args.images)
    print("frames        %d" % len(files))
    print("criterion     G>=%d, R<=%d, B<=%d; overlay when more than %d px"
          % (args.g_min, args.rb_max, args.rb_max, args.min_green))
    print("")
    for size, counts in groups.items():
        counts = np.asarray(counts)
        total = size[0] * size[1]
        with_overlay = int((counts > args.min_green).sum())
        ambiguous = int(((counts > 0) & (counts <= args.min_green)).sum())
        print("  %dx%d  %d frames" % (size[0], size[1], counts.size))
        print(
            "    with an overlay      %d  (%.1f%%)"
            % (with_overlay, 100.0 * with_overlay / counts.size)
        )
        print("    between 1 and %-4d px %d frames" % (args.min_green, ambiguous))
        if with_overlay:
            largest = int(counts.max())
            print(
                "    largest footprint    %d px of %d  (%.3f%%)"
                % (largest, total, 100.0 * largest / total)
            )
            print(
                "    bounds a whole-frame mean absolute difference at "
                "%.4f grey levels" % (255.0 * largest / total)
            )
    print("")
    print("wrote         %s" % args.out)


if __name__ == "__main__":
    main()
