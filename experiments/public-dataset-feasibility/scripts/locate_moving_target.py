#!/usr/bin/env python3
"""Localise the single moving object in a fixed-camera frame sequence.

Why this exists
---------------
``frame_motion_stats.py`` reports a whole-frame statistic, and a whole-frame
statistic is blind to a small target. A quadrotor 36 x 20 px across in a
1920 x 1080 frame covers 0.035 % of the image; even if every one of its pixels
flipped by the full 255 grey levels, the frame-wide mean absolute difference
would move by less than 0.1 grey levels, which is below the sensor-noise floor
of a real camera. A sequence can therefore look perfectly flat by that measure
while an aircraft flies through it -- and in this study, both sequences did.

This script does the measurement that is actually sensitive to such a target.
The camera is fixed, so a per-pixel temporal median is a clean background plate;
anything that differs from it shows up as a compact foreground blob. For each
frame it reports the largest blob's pixel count, centroid and bounding box, in
original-image coordinates.

Two background plates, two different questions
----------------------------------------------
``--background global`` (the default) takes the median over the whole sequence.
A frame's blob then answers *is something here that is not part of the
sequence-long background?* -- which includes an aircraft parked on the floor,
because the floor position is not in the sequence-long median. It is a
foreground-presence measure, not a motion measure.

``--background local`` takes the median over a window of ``--bg-window`` frames
centred on the current one. A blob then answers *did something change relative
to its own immediate temporal neighbourhood?* -- which is motion. An object that
holds still for longer than the window is absorbed into its own background and
disappears. The count this produces is therefore strongly window-dependent, and
the window has to be quoted with it.

Report whichever you use, labelled for what it measures. They are not
interchangeable.

Minimum box width
-----------------
``--min-box-width`` rejects blobs narrower than a few analysis pixels. This is
not cosmetic. On the indoor sequence the unfiltered global-plate track ends with
eight frames whose "target" is a 4 x 32 px vertical sliver at a fixed image
position -- a window louvre catching the light, visually confirmed as such.
A quadrotor seen side-on is wider than it is tall (median 36 x 20 px here), so a
blob one analysis pixel wide cannot be this target at this scale, and excluding
it by construction is more honest than deleting the frames afterwards. Pass
``--min-box-width 0`` to reproduce the unfiltered numbers.

Assumes a static camera and a single dominant moving object. It will not work on
a moving platform.

Example
-------
    python locate_moving_target.py --images output_images \
        --out results/indoor-transfer/indoor-target-track.csv \
        --montage figures/indoor-flight-strip.jpg
"""

import argparse
import csv
import glob
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

WORK = (480, 270)  # analysis resolution; the target is still several px across


def build_background(files, stride, size):
    frames = [
        np.asarray(Image.open(p).convert("L").resize(size), dtype=np.uint8)
        for p in files[::stride]
    ]
    return np.median(np.stack(frames), axis=0).astype(np.float32), len(frames)


def largest_blob(mask):
    """Largest 8-connected component of a boolean mask, as (ys, xs), or None."""
    labels = -np.ones(mask.shape, dtype=np.int32)
    best = None
    ys_all, xs_all = np.nonzero(mask)
    for seed_y, seed_x in zip(ys_all, xs_all):
        if labels[seed_y, seed_x] >= 0:
            continue
        pixels = []
        stack = [(seed_y, seed_x)]
        labels[seed_y, seed_x] = 1
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < mask.shape[0]
                        and 0 <= nx < mask.shape[1]
                        and mask[ny, nx]
                        and labels[ny, nx] < 0
                    ):
                        labels[ny, nx] = 1
                        stack.append((ny, nx))
        if best is None or len(pixels) > len(best):
            best = pixels
    if best is None:
        return None
    arr = np.asarray(best)
    return arr[:, 0], arr[:, 1]


def track(
    files,
    threshold,
    min_blob,
    min_box_width,
    stride,
    size,
    background="global",
    window=61,
):
    if background == "global":
        plate, n_bg = build_background(files, stride, size)
        stack = None
    else:
        stack = np.stack(
            [
                np.asarray(Image.open(p).convert("L").resize(size), dtype=np.uint8)
                for p in files
            ]
        )
        plate, n_bg = None, window
    probe = Image.open(files[0])
    full_w, full_h = probe.size
    sx, sy = full_w / size[0], full_h / size[1]

    rows = []
    for index, path in enumerate(files):
        gray = np.asarray(
            Image.open(path).convert("L").resize(size), dtype=np.float32
        )
        if plate is None:
            half = window // 2
            lo, hi = max(0, index - half), min(len(files), index + half + 1)
            local = np.median(stack[lo:hi:stride], axis=0).astype(np.float32)
            reference = local
        else:
            reference = plate
        mask = np.abs(gray - reference) > threshold
        blob = largest_blob(mask) if mask.any() else None
        if blob is None or len(blob[0]) < min_blob:
            rows.append([index, os.path.basename(path), 0, "", "", "", ""])
            continue
        ys, xs = blob
        box_w = int(xs.max() - xs.min() + 1)
        box_h = int(ys.max() - ys.min() + 1)
        if box_w < min_box_width:
            rows.append([index, os.path.basename(path), 0, "", "", "", ""])
            continue
        rows.append(
            [
                index,
                os.path.basename(path),
                int(len(ys)),
                round(float(xs.mean()) * sx, 1),
                round(float(ys.mean()) * sy, 1),
                round(box_w * sx, 1),
                round(box_h * sy, 1),
            ]
        )
    return rows, n_bg, (full_w, full_h)


def write_montage(files, rows, path, indices, crop=(300, 220), cols=3):
    cw, ch = crop
    present = [r for r in rows if r[2] > 0]
    fallback = (
        (present[len(present) // 2][3], present[len(present) // 2][4])
        if present
        else (files and 0, 0)
    )
    n = len(indices)
    n_rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, ch * n_rows), (255, 255, 255))
    label = ImageDraw.Draw(sheet)
    for k, index in enumerate(indices):
        row = rows[index]
        cx, cy = (row[3], row[4]) if row[2] > 0 else fallback
        box = (
            int(cx - cw / 2),
            int(cy - ch / 2),
            int(cx + cw / 2),
            int(cy + ch / 2),
        )
        tile = Image.open(files[index]).convert("RGB").crop(box)
        if row[2] > 0:
            draw = ImageDraw.Draw(tile)
            bw, bh = row[5], row[6]
            draw.rectangle(
                [cw / 2 - bw / 2, ch / 2 - bh / 2, cw / 2 + bw / 2, ch / 2 + bh / 2],
                outline=(255, 0, 0),
                width=2,
            )
        x, y = (k % cols) * cw, (k // cols) * ch
        sheet.paste(tile, (x, y))
        label.text(
            (x + 5, y + 5),
            "frame %d  cy=%s  blob=%s px" % (index, row[4], row[2]),
            fill=(255, 0, 0),
        )
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="directory of frames")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument(
        "--threshold",
        type=float,
        default=45.0,
        help="grey levels a pixel must differ from the background plate "
        "(default: 45)",
    )
    ap.add_argument(
        "--min-blob",
        type=int,
        default=6,
        help="smallest accepted blob, in analysis-resolution pixels (default: 6)",
    )
    ap.add_argument(
        "--min-box-width",
        type=int,
        default=2,
        help="smallest accepted bounding-box width, in analysis-resolution "
        "pixels (default: 2, i.e. 8 px at 1920x1080). Excludes one-pixel-wide "
        "vertical slivers, which cannot be this target. 0 disables it.",
    )
    ap.add_argument(
        "--background",
        choices=("global", "local"),
        default="global",
        help="median over the whole sequence (foreground presence) or over a "
        "sliding window (motion). Default: global.",
    )
    ap.add_argument(
        "--bg-window",
        type=int,
        default=61,
        help="frames in the sliding window when --background local "
        "(default: 61)",
    )
    ap.add_argument(
        "--bg-stride",
        type=int,
        default=8,
        help="use every Nth frame to build the background plate (default: 8). "
        "With --background local it subsamples inside the window instead.",
    )
    ap.add_argument(
        "--montage",
        help="optional JPEG showing crops centred on the tracked blob",
    )
    ap.add_argument(
        "--montage-frames",
        type=int,
        default=9,
        help="how many evenly spaced frames the montage shows (default: 9)",
    )
    ap.add_argument(
        "--montage-indices",
        help="comma-separated frame indices for the montage, overriding "
        "--montage-frames. Use it to pick legible frames; state which ones "
        "you picked.",
    )
    ap.add_argument(
        "--airborne-above",
        type=float,
        default=550.0,
        help="report how many frames have the centroid above this image row, "
        "in original-image pixels (default: 550)",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, args.pattern)))
    if len(files) < 2:
        raise SystemExit("need at least 2 frames in %s" % args.images)
    sizes = Counter(Image.open(p).size for p in files)
    if len(sizes) > 1:
        raise SystemExit(
            "this directory holds frames of %d different sizes (%s), so it "
            "holds more than one recording. A single background plate over a "
            "mixture is meaningless. Run partition_frame_dump.py first."
            % (len(sizes), dict(sizes))
        )

    stride = args.bg_stride if args.background == "global" else max(1, args.bg_stride // 2)
    rows, n_bg, full = track(
        files,
        args.threshold,
        args.min_blob,
        args.min_box_width,
        stride,
        WORK,
        args.background,
        args.bg_window,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["index", "filename", "blob_px", "cx_px", "cy_px", "bw_px", "bh_px"]
        )
        writer.writerows(rows)

    present = [r for r in rows if r[2] > 0]
    print("frames                %d  (%dx%d)" % (len(files), full[0], full[1]))
    if args.background == "global":
        print("background plate      global median of %d frames" % n_bg)
    else:
        print(
            "background plate      local median, %d-frame window, every %dth "
            "frame in it" % (args.bg_window, stride)
        )
    print("min box width         %d analysis px" % args.min_box_width)
    print(
        "frames with a blob    %d  (%.1f%%)"
        % (len(present), 100.0 * len(present) / len(rows))
    )
    if present:
        cy = np.array([r[4] for r in present])
        bw = np.array([r[5] for r in present])
        bh = np.array([r[6] for r in present])
        print("centroid y            %.0f .. %.0f px" % (cy.min(), cy.max()))
        print(
            "bbox w                %.0f .. %.0f px (median %.0f)"
            % (bw.min(), bw.max(), np.median(bw))
        )
        print(
            "bbox h                %.0f .. %.0f px (median %.0f)"
            % (bh.min(), bh.max(), np.median(bh))
        )
        airborne = [r for r in present if r[4] < args.airborne_above]
        if airborne:
            indices = [r[0] for r in airborne]
            print(
                "centroid above y=%.0f  %d frames, none outside %d..%d "
                "(%d frames in that window have no accepted blob or sit below "
                "the line)"
                % (
                    args.airborne_above,
                    len(airborne),
                    min(indices),
                    max(indices),
                    max(indices) - min(indices) + 1 - len(airborne),
                )
            )
    print("wrote                 %s" % args.out)

    if args.montage:
        if args.montage_indices:
            indices = [int(v) for v in args.montage_indices.split(",")]
        else:
            step = max(1, len(files) // args.montage_frames)
            indices = list(range(0, len(files), step))[: args.montage_frames]
        write_montage(files, rows, args.montage, indices)
        print("wrote                 %s" % args.montage)


if __name__ == "__main__":
    main()
