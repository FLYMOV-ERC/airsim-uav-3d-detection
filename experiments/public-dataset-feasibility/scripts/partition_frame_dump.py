#!/usr/bin/env python3
"""Split a directory of extracted frames into the recordings it actually holds.

Why this exists
---------------
``output_images_holybro2/`` looked like one recording. It is not: 1217 frames of
the Holybro outdoor session are followed by 596 frames of a different site
entirely, at a different resolution, from a different camera. Every statistic
this study first computed over "the outdoor sequence" was computed over that
mixture, and the largest frame-to-frame difference in the whole sequence turned
out to be the seam between the two recordings rather than anything moving.

A frame dump written by ``extract_frames_from_rosbag.py`` or
``detect_in_rosbag.py`` is named ``frame_%06d.jpg`` by message index, so running
the script twice into the same output directory silently interleaves two bags:
the second run overwrites the first only where the indices overlap and leaves
the tail of the longer run behind. Nothing in the filenames records that.

So: before measuring anything over a frame dump, partition it.

What it measures
----------------
Frames are segmented on pixel dimensions alone, because that is the one
property a single camera cannot change mid-recording. For each frame the script
also records the fraction of pixels whose three channels are equal, which
separates a greyscale camera stored in an RGB JPEG (fraction ~1.0) from a
colour one (well below it). That fraction is reported per segment rather than
segmented on, because a detector pass draws coloured boxes onto some frames and
would otherwise split one recording into dozens of runs.

This is a *sufficient* check, not a complete one: two recordings from the same
camera at the same resolution will not be separated by it. It catches the case
this study actually had, and it is cheap.

Example
-------
    python partition_frame_dump.py --images output_images_holybro2 \
        --out results/indoor-transfer/frame-dump-segments-holybro2.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image


def probe(path, tolerance=2):
    """Return (width, height, neutral_fraction) for one frame."""
    image = Image.open(path)
    width, height = image.size
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)
    return width, height, float((spread <= tolerance).mean())


def segment(rows):
    """Collapse per-frame records into contiguous runs of the same resolution."""
    segments = []
    for index, (_name, width, height, neutral) in enumerate(rows):
        key = (width, height)
        if segments and segments[-1][0] == key:
            segments[-1][2] = index
            segments[-1][3].append(neutral)
        else:
            segments.append([key, index, index, [neutral]])
    return segments


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="directory of frames")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="max channel spread still counted as greyscale (default: 2), "
        "which allows for JPEG chroma rounding",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, args.pattern)))
    if not files:
        raise SystemExit("no frames matching %s in %s" % (args.pattern, args.images))

    rows = []
    for path in files:
        width, height, neutral = probe(path, args.tolerance)
        rows.append([os.path.basename(path), width, height, neutral])

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["index", "filename", "width", "height", "neutral_pixel_fraction"]
        )
        for index, row in enumerate(rows):
            writer.writerow([index, row[0], row[1], row[2], round(row[3], 4)])

    segments = segment(rows)
    print("source    %s" % args.images)
    print("frames    %d" % len(files))
    print("segments  %d" % len(segments))
    print("")
    print(
        "  %-6s %-6s %-8s %-11s %-9s %s"
        % ("first", "last", "frames", "size", "neutral", "reads as")
    )
    for (width, height), first, last, neutrals in segments:
        median = float(np.median(neutrals))
        print(
            "  %-6d %-6d %-8d %-11s %-9.3f %s"
            % (
                first,
                last,
                len(neutrals),
                "%dx%d" % (width, height),
                median,
                "greyscale camera" if median >= 0.9 else "colour camera",
            )
        )
    print("")
    print("wrote     %s" % args.out)
    if len(segments) > 1:
        print("")
        print(
            "This dump holds more than one recording. Measure each segment "
            "separately; a statistic over the whole directory mixes sources "
            "and its largest frame-to-frame difference will be the seam."
        )


if __name__ == "__main__":
    main()
