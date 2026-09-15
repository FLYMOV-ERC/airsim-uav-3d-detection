#!/usr/bin/env python3
"""Blur a third party's business signage out of a published figure.

Why this exists
---------------
The outdoor recordings were made on commercial premises that are not the
author's. Two signs on the building -- a roof wordmark and a smaller board with
a company name and web address -- are legible in several of the published
figures, and between them they identify the site. They carry no evidentiary
value: nothing in this study depends on *which* building the aircraft was flown
in front of. So they are blurred, and this script is the record of how.

How the regions are found
-------------------------
Template matching (``cv2.TM_CCOEFF_NORMED``) against two crops of the signs
taken from a source frame, swept over scale and over the horizontal mirror,
because the Ultralytics training mosaics resize and flip their tiles. Everything
scoring above ``--threshold`` is blurred.

The sweep starts at ``--min-scale`` because a sign rendered smaller than that is
already illegible; there is no point blurring what cannot be read.

**The matcher over-fires, and that is deliberate.** At these thresholds it also
flags patches of sky, foliage and asphalt that happen to correlate with the
template. Those are blurred too. A blurred patch of asphalt costs nothing; a
missed sign costs the thing this script exists for. The box list is shipped in
``results/redaction/`` so the redaction can be inspected and replayed without
the source frames.

What is protected
-----------------
``--protect-blue`` keeps the Ultralytics annotation colour sharp: in the
training mosaics a sign sits directly behind the drawn target box, and blurring
through it would damage the evidence the figure exists to carry. Those pixels
are composited back unblurred.

Two ways to run it
------------------
Re-derive the boxes (needs a source frame, which is not in this tree)::

    python redact_signage.py --image fig.jpg --out fig.jpg \
        --reference /path/to/frame_000900.jpg \
        --template 820,230,1160,310 --template 695,395,945,495 \
        --boxes-out results/redaction/fig.json

Replay a shipped box list (needs nothing but this tree)::

    python redact_signage.py --image fig.jpg --out fig.jpg \
        --boxes-json results/redaction/fig.json
"""

import argparse
import json
import os

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - reported at runtime
    cv2 = None


def parse_box(text):
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    return parts


def find_boxes(image, templates, threshold, min_scale, max_scale, step):
    """Multi-scale, mirrored template match. Returns [x, y, w, h, score, tpl]."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    found = []
    for name, tpl in templates:
        tgray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hits = []
        for mirrored in (False, True):
            base = tgray[:, ::-1].copy() if mirrored else tgray
            for scale in np.arange(min_scale, max_scale, step):
                th, tw = int(base.shape[0] * scale), int(base.shape[1] * scale)
                if th < 8 or tw < 12 or th >= gray.shape[0] or tw >= gray.shape[1]:
                    continue
                scaled = cv2.resize(base, (tw, th))
                response = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.nonzero(response > threshold)
                for y, x in zip(ys, xs):
                    hits.append(
                        [int(x), int(y), tw, th, float(response[y, x]), name]
                    )
        hits.sort(key=lambda h: -h[4])
        kept = []
        for hit in hits:
            x, y, w, h = hit[:4]
            if any(
                not (
                    x + w <= k[0]
                    or k[0] + k[2] <= x
                    or y + h <= k[1]
                    or k[1] + k[3] <= y
                )
                for k in kept
            ):
                continue
            kept.append(hit)
        found.extend(kept)
    return found


def blue_mask(image, dilate=2):
    """Pixels of the Ultralytics annotation colour, dilated by a few pixels."""
    b, g, r = (image[..., i].astype(np.int16) for i in range(3))
    mask = (b >= 150) & (b - np.maximum(g, r) >= 60)
    if dilate:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask


def redact(image, boxes, pad, protect_blue):
    out = image.copy()
    protect = blue_mask(image) if protect_blue else None
    height, width = image.shape[:2]
    for box in boxes:
        x, y, w, h = box[:4]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = out[y0:y1, x0:x1]
        # Downsample-then-upsample first: that throws the information away
        # rather than smearing it, so no amount of sharpening brings the
        # lettering back. The Gaussian afterwards only hides the blocking.
        # Both kernels are tied to the box height, so a sign is destroyed to
        # the same degree whatever scale it was rendered at.
        ph, pw = patch.shape[:2]
        factor = max(2, ph // 6)
        small = cv2.resize(
            patch,
            (max(1, pw // factor), max(1, ph // factor)),
            interpolation=cv2.INTER_AREA,
        )
        blurred = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
        k = max(5, int(ph / 2) | 1)
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)
        if protect is not None:
            keep = protect[y0:y1, x0:x1]
            blurred[keep] = patch[keep]
        out[y0:y1, x0:x1] = blurred
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True, help="figure to redact")
    ap.add_argument("--out", required=True, help="where to write it")
    ap.add_argument(
        "--reference", help="source frame the sign templates are cropped from"
    )
    ap.add_argument(
        "--template",
        action="append",
        type=parse_box,
        default=[],
        help="x0,y0,x1,y1 of a sign inside --reference; repeatable",
    )
    ap.add_argument("--boxes-json", help="replay a previously written box list")
    ap.add_argument("--boxes-out", help="write the box list that was applied")
    ap.add_argument("--threshold", type=float, default=0.63)
    ap.add_argument("--min-scale", type=float, default=0.17)
    ap.add_argument("--max-scale", type=float, default=0.95)
    ap.add_argument("--scale-step", type=float, default=0.01)
    ap.add_argument(
        "--pad", type=int, default=4, help="grow each box by this many pixels"
    )
    ap.add_argument(
        "--protect-blue",
        action="store_true",
        help="composite the annotation-blue pixels back unblurred",
    )
    args = ap.parse_args()

    if cv2 is None:
        raise SystemExit("this script needs opencv-python")

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit("could not read %s" % args.image)

    if args.boxes_json:
        boxes = json.load(open(args.boxes_json))
        print("boxes from       %s" % args.boxes_json)
    else:
        if not args.reference or not args.template:
            raise SystemExit("give either --boxes-json or --reference/--template")
        reference = cv2.imread(args.reference)
        if reference is None:
            raise SystemExit("could not read %s" % args.reference)
        templates = [
            ("tpl%d" % i, reference[b[1] : b[3], b[0] : b[2]])
            for i, b in enumerate(args.template)
        ]
        boxes = find_boxes(
            image,
            templates,
            args.threshold,
            args.min_scale,
            args.max_scale,
            args.scale_step,
        )
        print("reference        %s" % args.reference)

    out = redact(image, boxes, args.pad, args.protect_blue)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out, out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    if args.boxes_out:
        json_dir = os.path.dirname(os.path.abspath(args.boxes_out))
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.boxes_out, "w") as fh:
            json.dump(boxes, fh, indent=1)
        print("wrote boxes      %s" % args.boxes_out)

    print("image            %s  (%dx%d)" % (args.image, image.shape[1], image.shape[0]))
    print("regions blurred  %d" % len(boxes))
    print("wrote            %s" % args.out)


if __name__ == "__main__":
    main()
