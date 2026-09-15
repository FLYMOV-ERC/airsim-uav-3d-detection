#!/usr/bin/env python3
"""Collapse a directory of Ultralytics run folders into one index CSV.

Reads only ``args.yaml`` and ``results.csv`` from each run, so it needs neither
the weights nor the datasets nor the rosbags. That is the point: it lets the
training record survive in a few kilobytes after the 49.3 MB of checkpoints in
those run directories and the third-party datasets have been left out.

The column that carries the finding is ``epochs_completed``. Ultralytics writes
``results.csv`` only once an epoch has finished, so a run with no ``results.csv``
never completed a single epoch -- in this study, those runs died while resolving
their dataset descriptor. The ``data_yaml`` column shows what each one was
pointing at.

Example
-------
    python summarize_runs.py --runs /path/to/runs/detect --out results/run-index.csv
"""

import argparse
import csv
import os
import re

PRECISION = "metrics/precision(B)"
RECALL = "metrics/recall(B)"
MAP50 = "metrics/mAP50(B)"
MAP50_95 = "metrics/mAP50-95(B)"

COLUMNS = [
    "run",
    "data_yaml",
    "model",
    "device",
    "epochs_requested",
    "epochs_completed",
    "weight_files",
    "wall_s_last_epoch",
    "final_precision",
    "final_recall",
    "final_mAP50",
    "final_mAP50_95",
    "best_mAP50",
]


def read_args_yaml(path):
    """Flat key: value reader. args.yaml is flat, so a real YAML parser is not
    needed and this keeps the script dependency-free."""
    cfg = {}
    with open(path) as fh:
        for line in fh:
            match = re.match(r"^([A-Za-z_][A-Za-z_0-9]*):\s*(.*)$", line.strip())
            if match:
                cfg[match.group(1)] = match.group(2).strip()
    return cfg


def read_results_csv(path):
    with open(path) as fh:
        rows = [
            {k.strip(): v.strip() for k, v in row.items() if k is not None}
            for row in csv.DictReader(fh)
        ]
    if not rows:
        return 0, {}, "", ""
    last = rows[-1]
    best50 = max(float(row[MAP50]) for row in rows)
    return len(rows), last, last.get("time", ""), "%.5f" % best50


def summarize(runs_dir):
    out_rows = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        args_yaml = os.path.join(run_dir, "args.yaml")
        if not os.path.isfile(args_yaml):
            continue

        cfg = read_args_yaml(args_yaml)
        results_csv = os.path.join(run_dir, "results.csv")
        if os.path.isfile(results_csv):
            n_epochs, last, wall, best50 = read_results_csv(results_csv)
        else:
            n_epochs, last, wall, best50 = 0, {}, "", ""

        weights_dir = os.path.join(run_dir, "weights")
        n_weights = (
            len([f for f in os.listdir(weights_dir) if f.endswith(".pt")])
            if os.path.isdir(weights_dir)
            else 0
        )

        out_rows.append(
            [
                name,
                cfg.get("data", ""),
                cfg.get("model", ""),
                cfg.get("device", ""),
                cfg.get("epochs", ""),
                n_epochs,
                n_weights,
                wall,
                last.get(PRECISION, ""),
                last.get(RECALL, ""),
                last.get(MAP50, ""),
                last.get(MAP50_95, ""),
                best50,
            ]
        )
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs",
        required=True,
        help="directory holding the run subdirectories (e.g. runs/detect)",
    )
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    rows = summarize(args.runs)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    trained = sum(1 for r in rows if r[5] > 0)
    print("wrote %s" % args.out)
    print("runs indexed:            %d" % len(rows))
    print("runs with >=1 epoch:     %d" % trained)
    print("runs with 0 epochs:      %d" % (len(rows) - trained))


if __name__ == "__main__":
    main()
