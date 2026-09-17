#!/usr/bin/env python3
"""Temporal analysis of the 3D localization error (Section 7.5.4, fig. ``localization_over_time``).

The chapter's tables report campaign-wide averages. This script recovers the same
matched (ground-truth, track) pairs that produce Table 7.3 and the error-versus-range
figure -- identical dumps, identical tracker (``evaluation/metrics_ekf.py``), identical
operating point (fused score >= 0.5) and identical 3D gate -- but indexes them by FRAME
instead of by range, so the evolution of the error along a sequence can be characterized
statistically: Spearman correlation against sequence time, Kruskal-Wallis across time
bins, and Mann-Whitney between the first five frames and the rest.

Sequences differ in length (50-92 frames), so aggregation is done on normalized sequence
time tau = i/(n-1) in [0, 1].

Written for the examining-board revision (August 2026).

Usage::

    python evaluation/temporal_error.py --data-root /path/to/data --outdir /path/to/dissertation

reads ``<data-root>/<dumps>/*.json`` and ``<data-root>/sessions/<seq>/meta.json`` and writes
``<outdir>/Cap7/localization_over_time.pdf`` plus the statistics as JSON (``--json``).
"""
import sys, json, glob, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from tracking.sort import SortTracker
from tracking.ekf_3d import EKFTracker3D, rotation_matrix, spherical_from_cartesian
from scipy.optimize import linear_sum_assignment
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GATE = 4.0
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", grey="#7F7F7F")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

def run_seq(frames, xpof, fuse_w=0.5, thr=0.5, max_age=15, min_hits=3, max_dist=90):
    """Same tracker as evaluation/metrics_ekf.run_tracker; returns the per-frame matches."""
    trk = SortTracker(iou_thr=0.3, max_age=max_age, min_hits=min_hits, nms_iou=0.4)
    ekfs = {}; prev_ts = None; out = []
    n = len(frames)
    for i, fr in enumerate(frames):
        xp = xpof.get(fr['frame']); ts = fr.get('ts')
        dt = 0.2
        if prev_ts is not None and ts is not None: dt = min(1.0, max(0.05, ts - prev_ts))
        if ts is not None: prev_ts = ts
        dets = []
        for d in fr['dets']:
            f = fuse_w * d['yolo'] + (1 - fuse_w) * d['pn']
            if f >= thr and d['d'] <= max_dist:
                dets.append({'bbox': tuple(d['bbox']), 'fused': f, 'd': d['d'],
                             'gpos': np.array(d['gpos'])})
        tracks = trk.update(dets)
        hyp_pos = []; hyp_age = []
        for t in tracks:
            if t.id in ekfs: ekfs[t.id].predict(dt)
            if t.time_since_update == 0 and xp is not None:
                g = np.array(t.det['gpos']); R = rotation_matrix(xp[6], xp[7], xp[8])
                pc = R @ (g - xp[:3]); dd, phi, th = spherical_from_cartesian(pc)
                meas = np.array([dd, phi, th, 1.0])
                if t.id in ekfs: ekfs[t.id].update(meas, xp)
                else: ekfs[t.id] = EKFTracker3D(meas, xp)
            hyp_pos.append(ekfs[t.id].position_global() if t.id in ekfs
                           else np.array(t.det['gpos']))
            hyp_age.append(t.hits)
        gt_pos = [np.array(v) for v in fr['gt'].values()]
        if gt_pos and hyp_pos:
            D = np.full((len(gt_pos), len(hyp_pos)), np.nan)
            for a in range(len(gt_pos)):
                for b in range(len(hyp_pos)):
                    e = np.linalg.norm(gt_pos[a] - hyp_pos[b])
                    if e < GATE: D[a, b] = e
            Cm = np.where(np.isnan(D), 1e6, D)
            ri, ci = linear_sum_assignment(Cm)
            for a, b in zip(ri, ci):
                if Cm[a, b] < GATE:
                    out.append(dict(i=i, tau=(i / (n - 1) if n > 1 else 0.0),
                                    err=float(Cm[a, b]),
                                    rng=float(np.linalg.norm(gt_pos[a])),
                                    age=int(hyp_age[b])))
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=".", help="directory holding the dumps and sessions/")
    ap.add_argument("--dumps", default="det_dumps_best", help="dump directory, relative to --data-root")
    ap.add_argument("--sessions", default=None, help="sessions directory (default: <data-root>/sessions)")
    ap.add_argument("--outdir", default=".", help="the figure goes to <outdir>/Cap7/")
    ap.add_argument("--json", default=None, help="statistics output (default: <outdir>/Cap7/localization_over_time.json)")
    args = ap.parse_args()
    root = Path(args.data_root)
    sessions = Path(args.sessions) if args.sessions else root / "sessions"
    cap7 = Path(args.outdir) / "Cap7"; cap7.mkdir(parents=True, exist_ok=True)

    rows = []
    lens = []
    for dp in sorted(glob.glob(str(root / args.dumps / "*.json"))):
        name = Path(dp).stem; env = name.split('_')[0]
        if env not in ('nh', 'city', 'coast'): continue
        frames = json.load(open(dp))['frames']
        mp = sessions / name / "meta.json"
        xpof = {f["frame"]: np.array(f["x_plat"]) for f in json.load(open(mp))["frames"]} \
            if mp.exists() else {}
        m, n = run_seq(frames, xpof)
        lens.append(n)
        for r in m:
            r['env'] = env; r['seq'] = name
            rows.append(r)
    tau = np.array([r['tau'] for r in rows])
    err = np.array([r['err'] for r in rows])
    idx = np.array([r['i'] for r in rows])
    env = np.array([r['env'] for r in rows])
    print(f"sequences={len(lens)}  frames {min(lens)}-{max(lens)} (median {int(np.median(lens))})"
          f"  matched samples={len(rows)}")

    # ---------- statistics ----------
    NB = 12
    edges = np.linspace(0, 1, NB + 1)
    ctr, med, q25, q75, p90, cnt = [], [], [], [], [], []
    for i in range(NB):
        m = (tau >= edges[i]) & (tau < edges[i + 1] if i < NB - 1 else tau <= edges[i + 1])
        if m.sum() < 10: continue
        ctr.append(0.5 * (edges[i] + edges[i + 1])); cnt.append(int(m.sum()))
        med.append(np.median(err[m])); q25.append(np.percentile(err[m], 25))
        q75.append(np.percentile(err[m], 75)); p90.append(np.percentile(err[m], 90))
    ctr, med, q25, q75, p90 = map(np.array, (ctr, med, q25, q75, p90))

    rho, pval = stats.spearmanr(tau, err)
    # first-frames transient: by absolute frame index within the sequence
    early = err[idx < 5]; late = err[idx >= 5]
    u, pu = stats.mannwhitneyu(early, late, alternative='two-sided')
    # Kruskal-Wallis across the normalized-time bins
    groups = [err[(tau >= edges[i]) & (tau < edges[i + 1])] for i in range(NB)]
    groups = [g for g in groups if len(g) >= 10]
    hstat, ph = stats.kruskal(*groups)

    stats_out = dict(
        dumps=args.dumps, n_sequences=len(lens), n_samples=len(rows),
        seq_len_min=int(min(lens)), seq_len_max=int(max(lens)),
        seq_len_median=float(np.median(lens)),
        overall=dict(median=float(np.median(err)), mean=float(err.mean()),
                     rmse=float(np.sqrt(np.mean(err ** 2))),
                     p90=float(np.percentile(err, 90)), max=float(err.max())),
        spearman_rho=float(rho), spearman_p=float(pval),
        kruskal_H=float(hstat), kruskal_p=float(ph),
        early_median=float(np.median(early)), early_n=int(early.size),
        late_median=float(np.median(late)), late_n=int(late.size),
        mannwhitney_p=float(pu),
        band_median_min=float(med.min()), band_median_max=float(med.max()),
        band_p90_min=float(p90.min()), band_p90_max=float(p90.max()),
        by_env={e: dict(n=int((env == e).sum()),
                        median=float(np.median(err[env == e])),
                        rmse=float(np.sqrt(np.mean(err[env == e] ** 2))))
                for e in ('nh', 'city', 'coast')},
        bins=[dict(tau=float(c), n=int(k), median=float(m2), q25=float(a), q75=float(b),
                   p90=float(p))
              for c, k, m2, a, b, p in zip(ctr, cnt, med, q25, q75, p90)],
    )
    # error by track age (initialization transient)
    age = np.array([r['age'] for r in rows])
    stats_out['by_age'] = [dict(age=int(a), n=int((age == a).sum()),
                                median=float(np.median(err[age == a])))
                           for a in range(1, 9) if (age == a).sum() >= 10]
    print(json.dumps({k: v for k, v in stats_out.items() if k not in ('bins',)}, indent=1))

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.9))
    a = ax[0]
    a.fill_between(ctr, q25, q75, color=C["blue"], alpha=0.22, label="interquartile range")
    a.plot(ctr, p90, "--", color=C["red"], lw=1.2, label="90th percentile")
    a.plot(ctr, med, "-o", color=C["blue"], lw=1.6, ms=3.5, label="median")
    a.set_xlabel("normalized sequence time $\\tau$")
    a.set_ylabel("3D centroid error (m)")
    a.set_ylim(0, max(2.6, p90.max() * 1.12))
    a.set_xlim(0, 1)
    a.set_title("(a) Error over sequence time")
    a.legend(loc="upper right", framealpha=0.9, fontsize=7.5)

    b = ax[1]
    se = np.sort(err); cdf = np.arange(1, len(se) + 1) / len(se)
    b.plot(se, cdf, "-", color=C["blue"], lw=1.8, label=f"all frames ($n={len(se)}$)")
    for e, col, lab in (("nh", C["green"], "AirSimNH"), ("city", C["orange"], "City"),
                        ("coast", C["purple"], "Coastline")):
        s2 = np.sort(err[env == e]); c2 = np.arange(1, len(s2) + 1) / len(s2)
        b.plot(s2, c2, "-", color=col, lw=1.0, alpha=0.85, label=lab)
    b.axvline(np.median(err), ls=":", color="k", lw=1)
    b.text(np.median(err), 0.04, f" median $=$ {np.median(err):.2f} m", fontsize=7.5,
           rotation=90, va="bottom")
    b.set_xlabel("3D centroid error (m)"); b.set_ylabel("empirical CDF")
    b.set_xlim(0, GATE); b.set_ylim(0, 1.02)
    b.set_title("(b) Per-frame error distribution")
    b.legend(loc="lower right", framealpha=0.9, fontsize=7.5)
    fig.tight_layout()
    out = cap7 / "localization_over_time.pdf"
    fig.savefig(out)
    plt.close(fig); print("wrote", out)
    jp = Path(args.json) if args.json else cap7 / "localization_over_time.json"
    json.dump(stats_out, open(jp, "w"), indent=2); print("wrote", jp)


if __name__ == "__main__":
    main()
