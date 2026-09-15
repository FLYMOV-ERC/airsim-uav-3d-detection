#!/usr/bin/env python3
"""Measure the extent of a plotted trace from the rendered figure.

``figures/holybro-out01-mocap-trajectory.png`` is the only record left of the
motion-capture volume: the rosbag it came from is not in this tree, so the
flight extent cannot be recomputed from the poses. It can, however, be measured
off the image, and that is what this does -- so the number in the README is
derived by a script a reader can re-run against a file that is here, rather than
read off the screen by hand.

Method
------
* The axes frame is the set of rows and columns that are almost entirely dark.
* Tick marks are the short dark runs just outside that frame. Their mean spacing
  in pixels, against ``--tick-spacing`` in data units, is the scale. Using the
  spacing rather than the labels means no text has to be read; it also averages
  over every tick, so a one-pixel error anywhere is divided by the tick count.
* The trace is every pixel close to ``--colour`` (matplotlib's default first
  colour by default).

The reported extent is the bounding box of the trace pixels. It therefore
includes the width of the drawn line and the size of the markers, which inflates
it by roughly one marker diameter -- a few centimetres at this scale. Read it as
an upper bound.

Example
-------
    python measure_plot_extent.py \
        --figure figures/holybro-out01-mocap-trajectory.png --tick-spacing 2
"""

import argparse

import numpy as np
from PIL import Image


def runs_to_centres(values):
    """Collapse consecutive integers into the centre of each run."""
    centres = []
    start = previous = values[0]
    for value in values[1:]:
        if value - previous > 1:
            centres.append((start + previous) / 2.0)
            start = value
        previous = value
    centres.append((start + previous) / 2.0)
    return centres


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figure", required=True, help="rendered plot (PNG)")
    ap.add_argument(
        "--tick-spacing",
        type=float,
        required=True,
        help="data units between two adjacent ticks, e.g. 2 for a 2 m grid",
    )
    ap.add_argument(
        "--colour",
        default="31,119,180",
        help="R,G,B of the trace (default: matplotlib C0)",
    )
    ap.add_argument("--colour-tolerance", type=int, default=40)
    ap.add_argument(
        "--dark",
        type=int,
        default=120,
        help="a pixel is axis ink when all channels are below this (default: 120)",
    )
    ap.add_argument("--units", default="m")
    args = ap.parse_args()

    arr = np.asarray(Image.open(args.figure).convert("RGB")).astype(np.int16)
    height, width = arr.shape[:2]
    ink = arr.max(axis=2) < args.dark

    spine_rows = np.nonzero(ink.sum(axis=1) > width // 3)[0]
    spine_cols = np.nonzero(ink.sum(axis=0) > height // 3)[0]
    if spine_rows.size < 2 or spine_cols.size < 2:
        raise SystemExit("could not find the axes frame; adjust --dark")
    top, bottom = int(spine_rows.min()), int(spine_rows.max())
    left, right = int(spine_cols.min()), int(spine_cols.max())

    below = ink[bottom + 2 : bottom + 6, :]
    x_ticks = runs_to_centres(np.nonzero(below.sum(axis=0) >= 3)[0])
    beside = ink[:, left - 6 : left - 2]
    y_ticks = runs_to_centres(np.nonzero(beside.sum(axis=1) >= 3)[0])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise SystemExit("found fewer than two ticks on an axis")

    px_per_tick_x = float(np.mean(np.diff(x_ticks)))
    px_per_tick_y = float(np.mean(np.diff(y_ticks)))

    target = [int(v) for v in args.colour.split(",")]
    trace = np.ones(arr.shape[:2], dtype=bool)
    for channel in range(3):
        trace &= np.abs(arr[..., channel] - target[channel]) <= args.colour_tolerance
    ys, xs = np.nonzero(trace)
    if xs.size == 0:
        raise SystemExit("no trace pixels matched --colour")

    span_x = (xs.max() - xs.min()) / px_per_tick_x * args.tick_spacing
    span_y = (ys.max() - ys.min()) / px_per_tick_y * args.tick_spacing

    print("figure            %s  (%dx%d)" % (args.figure, width, height))
    print("axes frame        x %d..%d, y %d..%d" % (left, right, top, bottom))
    print(
        "ticks             %d on x, %d on y; %.2f / %.2f px per %.3g %s"
        % (
            len(x_ticks),
            len(y_ticks),
            px_per_tick_x,
            px_per_tick_y,
            args.tick_spacing,
            args.units,
        )
    )
    print("trace pixels      %d" % int(trace.sum()))
    print(
        "extent            %.2f %s x %.2f %s  (bounding box of the drawn "
        "trace, so an upper bound by about one marker diameter)"
        % (span_x, args.units, span_y, args.units)
    )


if __name__ == "__main__":
    main()
