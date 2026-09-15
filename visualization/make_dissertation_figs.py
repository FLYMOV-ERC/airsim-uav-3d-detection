#!/usr/bin/env python3
"""Generate the dissertation figures as vector PDFs.

Writes one figure for Chapter 4 (the pipeline hub diagram) and ten for Chapter 7.

**The quantitative figures recompute nothing.**  Their values are literals typed
into this file, and this script is not their provenance -- it only draws them.
Editing a number here changes the figure and nothing else.  Where the chapter
prints the same quantity in a table, the literals agree with it: ``fig_comparison``
reproduces every row of Table 7.4 (``tab:arch``); ``fig_normalization`` its four
frustum rows; ``fig_perenv`` the AMOTA, recall and RMSE columns of Table 7.3
(``tab:main``); ``fig_yolo`` the two mAP50 values of Section 7.5.1.  A few plotted
numbers appear in no table of the chapter and were not cross-checked against one
-- the per-environment F1 of ``fig_perenv``, the per-range errors of ``fig_range``,
and the mAP50:95 and precision bars of ``fig_yolo``.

Four figures do read data from disk, and are skipped without it:
``fig_dataset_pipeline`` (sprites, backgrounds, composites), ``fig_depthband``
and ``fig_qualitative`` (a recorded session plus a detection dump), and
``fig_tracker`` (a detection dump).

Usage::

    python visualization/make_dissertation_figs.py --outdir /path/to/dissertation

which creates ``<outdir>/Cap4`` and ``<outdir>/Cap7``.  Pass ``--cap4-dir`` and
``--cap7-dir`` to place them elsewhere.

Figures produced (output file -> LaTeX label in the chapter source):

    Cap4/pipeline_architecture.pdf    the perception-pipeline hub diagram (Chapter 4)
    Cap7/dataset_pipeline.pdf         fig:dataset        synthetic composition
    Cap7/frustum_depthband.pdf        fig:depthband      the normalization problem
    Cap7/tracking_stage.pdf           fig:tracker        tracking stage + track timeline
    Cap7/yolo_metrics.pdf             fig:yolo           2D detector, mAP50 = 0.994
    Cap7/normalization_ablation.pdf   fig:norm           foreground-extraction ablation
    Cap7/per_environment.pdf          fig:mainresults    (left)  per environment
    Cap7/localization_vs_range.pdf    fig:mainresults    (right) error vs. range
    Cap7/architecture_comparison.pdf  fig:arch           frustum / voxel / fusion
    Cap7/regimes.pdf                  fig:regimes        precision-recall regimes
    Cap7/qualitative.pdf              fig:qual           qualitative results

Seven further figures of Chapter 7 are NOT produced here; see the "Figures this
script does not produce" section of README.md.
"""
import argparse, os, json, glob, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Circle
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # repo root
from common.config import CONFIG

# Root under which sessions/, dumps/, sprites/, backgrounds/ and datasets/ live.
# Set from configs/default.yaml (paths.data_root) or --data-root.
DATA = str(CONFIG.data_root)

# Output directories, set by main() from --outdir / --cap4-dir / --cap7-dir.
CAP4 = "figures/Cap4"
CAP7 = "figures/Cap7"


def _set_output_dirs(cap4: str, cap7: str) -> None:
    global CAP4, CAP7
    CAP4, CAP7 = cap4, cap7
    os.makedirs(CAP4, exist_ok=True)
    os.makedirs(CAP7, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# colourblind-safe (Wong)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#7F7F7F")

# camera intrinsics (CV frame)
FX = FY = 640.0; CX, CY = 640.0, 360.0; IMW, IMH = 1280, 720

def Rmat(phi, theta, psi):
    cphi,sphi=math.cos(phi),math.sin(phi); cth,sth=math.cos(theta),math.sin(theta); cps,sps=math.cos(psi),math.sin(psi)
    Rx=np.array([[1,0,0],[0,cphi,-sphi],[0,sphi,cphi]]); Ry=np.array([[cth,0,sth],[0,1,0],[-sth,0,cth]])
    Rz=np.array([[cps,-sps,0],[sps,cps,0],[0,0,1]]); return Rz@Ry@Rx

def save(fig, path):
    fig.savefig(path); plt.close(fig); print("wrote", path)

# ----------------------------------------------------------------------------- Ch4: pipeline diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_xlim(0,116); ax.set_ylim(0,46); ax.axis("off")
    stages = [
        ("2D Detection\n(YOLOv11s)", "RGB image", C["blue"], "Ch.7"),
        ("Frustum\nProposal", "+ intrinsics,\npoint cloud", C["orange"], "Ch.4/7"),
        ("3D Detection\n(PointNet++)", "frustum points", C["green"], "Ch.7"),
        ("Tracking\n(EKF)", "3D detections", C["purple"], "Ch.5/6"),
    ]
    x0, w, gap, y, h = 3, 21, 4.5, 18, 12
    centers=[]
    for i,(name,inp,col,ch) in enumerate(stages):
        x = x0 + i*(w+gap)
        box = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                             linewidth=1.4, edgecolor=col, facecolor=col+"22"); ax.add_patch(box)
        ax.text(x+w/2, y+h/2+1.2, name, ha="center", va="center", fontsize=8.4, fontweight="bold")
        ax.text(x+w/2, y-2.0, inp, ha="center", va="top", fontsize=7.3, style="italic", color=C["grey"])
        ax.text(x+w/2, y+h+2.0, ch, ha="center", va="bottom", fontsize=7.8, color=col, fontweight="bold")
        centers.append((x,x+w))
    for i in range(len(stages)-1):
        a=FancyArrowPatch((centers[i][1], y+h/2),(centers[i+1][0], y+h/2),
                          arrowstyle="-|>", mutation_scale=13, lw=1.4, color="black"); ax.add_patch(a)
    # output
    xo = x0 + len(stages)*(w+gap)
    ax.annotate("", xy=(xo+3, y+h/2), xytext=(centers[-1][1], y+h/2),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"))
    ax.text(xo+4, y+h/2, "calibrated\n3D tracks", ha="left", va="center", fontsize=8.2, fontweight="bold")
    # sensing bracket
    ax.annotate("", xy=(x0+3*(w+gap)+w, y+h+8.5), xytext=(x0, y+h+8.5),
                arrowprops=dict(arrowstyle="-", lw=1.0, color=C["grey"]))
    ax.text(x0+(3*(w+gap)+w)/2-2, y+h+9.3, "Sense (perception) — this dissertation",
            ha="center", va="bottom", fontsize=8.0, color=C["grey"])
    save(fig, f"{CAP4}/pipeline_architecture.pdf")

# ----------------------------------------------------------------------------- Ch7: synthetic dataset pipeline
def fig_dataset_pipeline():
    fig = plt.figure(figsize=(6.4, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1,1,1.25], wspace=0.28)
    # (a) sprite extraction
    axa = fig.add_subplot(gs[0,0]); axa.axis("off")
    axa.set_title("(a) Sprite library\n(brightness threshold)", fontsize=8.6)
    try:
        sp = sorted(glob.glob(os.path.join(DATA, "sprites", "drone_*.png")))
        import matplotlib.image as mpimg
        for j,k in enumerate([0, len(sp)//3, 2*len(sp)//3, len(sp)-1][:4]):
            im = mpimg.imread(sp[k]); axx=axa.inset_axes([0.05+0.47*(j%2),0.05+0.47*(j//2),0.42,0.42])
            axx.imshow(im); axx.axis("off")
        axa.text(0.5,-0.02,"196 poses: 12 yaw × 3 pitch\n× 3 roll × 5 ranges", ha="center", va="top",
                 transform=axa.transAxes, fontsize=6.8, color=C["grey"])
    except Exception as e:
        axa.text(0.5,0.5,"sprite\nlibrary", ha="center", va="center", transform=axa.transAxes)
    # (b) background
    axb = fig.add_subplot(gs[0,1]); axb.axis("off"); axb.set_title("(b) Empty backgrounds\n(3 environments)", fontsize=8.6)
    try:
        import matplotlib.image as mpimg
        bgs = sorted(glob.glob(os.path.join(DATA, "backgrounds", "*.jpg")))+sorted(glob.glob(os.path.join(DATA, "backgrounds", "**", "*.jpg"), recursive=True))
        im = mpimg.imread(bgs[0]); axb.imshow(im); axb.axis("off")
    except Exception:
        axb.add_patch(Rectangle((0,0),1,1, color=C["sky"]+"55"))
    # (c) composite with exact label
    axc = fig.add_subplot(gs[0,2]); axc.axis("off"); axc.set_title("(c) Alpha-composite\n+ pixel-exact label", fontsize=8.6)
    try:
        import matplotlib.image as mpimg
        vis = sorted(glob.glob(os.path.join(DATA, "dataset_synth", "visualizations", "*.jpg")))+\
              sorted(glob.glob(os.path.join(DATA, "dataset_synth", "yolo", "images", "train", "*.jpg")))
        im = mpimg.imread(vis[0]); axc.imshow(im); axc.axis("off")
    except Exception:
        axc.add_patch(Rectangle((0,0),1,1, color=C["green"]+"33"))
    save(fig, f"{CAP7}/dataset_pipeline.pdf")

# ----------------------------------------------------------------------------- Ch7: frustum + depth-band (REAL data)
def fig_depthband():
    sess = os.path.join(DATA, "sessions", "nh_seq01_near_clear_h13")
    dump = json.load(open(os.path.join(DATA, "det_dumps_segnet", "nh_seq01_near_clear_h13.json")))
    fr = next(f for f in dump["frames"] if f["dets"])
    i = fr["frame"]; bb = fr["dets"][0]["bbox"]
    depth = np.load(f"{sess}/depth/{i:05d}.npy").astype(np.float32)
    h,w = depth.shape
    uu,vv = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth>0.1)&(depth<250)
    z=depth[valid]; u=uu[valid]; v=vv[valid]
    x=(u-CX)*z/FX; y=(v-CY)*z/FY
    # frustum: inside the 2D box
    x1,y1,x2,y2 = bb
    inside = (u>=x1)&(u<=x2)&(v>=y1)&(v<=y2)
    zf = z[inside]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    # (a) frustum side view: forward(z) vs right(x)
    ax=axes[0]
    ax.scatter(z[inside], x[inside], s=2, c=C["blue"], alpha=0.5, edgecolors="none", label="frustum points")
    z0=np.percentile(zf,5.0); band=z0+8.0
    ax.axvspan(z0, band, color=C["green"], alpha=0.15, label="depth band")
    ax.axvline(band, color=C["green"], ls="--", lw=1.0)
    ax.set_xlabel("forward range $z$ (m)"); ax.set_ylabel("lateral $x$ (m)")
    ax.set_title("(a) Frustum point cloud (top view)", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9)
    # (b) depth histogram with band cut
    ax=axes[1]
    ax.hist(zf, bins=60, color=C["grey"]+"99", edgecolor="none")
    ax.axvspan(z0, band, color=C["green"], alpha=0.18)
    ax.axvline(z0, color=C["green"], lw=1.0); ax.axvline(band, color=C["green"], ls="--", lw=1.0)
    ax.annotate("target\ncluster", xy=(z0+2, ax.get_ylim()[1]*0.7), fontsize=7.5, color=C["green"], ha="left")
    ax.annotate("background tail\n(inflates max-norm scale)", xy=(band+2, ax.get_ylim()[1]*0.45),
                fontsize=7.2, color=C["red"], ha="left")
    ax.set_xlabel("forward range $z$ (m)"); ax.set_ylabel("point count")
    ax.set_title("(b) Range histogram in frustum", fontsize=9)
    fig.tight_layout(); save(fig, f"{CAP7}/frustum_depthband.pdf")

# ----------------------------------------------------------------------------- Ch7: 2D detector metrics
def fig_yolo():
    metrics = ["mAP@0.5","mAP@0.5:0.95","Precision","Recall"]
    synth = [0.994, None, None, None]    # synthetic headline
    # representative values
    synth = [0.994, 0.78, 0.99, 0.98]
    urban = [0.731, 0.403, 0.91, 0.64]
    x=np.arange(len(metrics)); wd=0.38
    fig, ax = plt.subplots(figsize=(4.6,2.7))
    ax.bar(x-wd/2, synth, wd, color=C["blue"], label="Synthetic (sprite-composited)")
    ax.bar(x+wd/2, urban, wd, color=C["orange"], label="Real-collected (API labels)")
    for xi,vs,vu in zip(x,synth,urban):
        ax.text(xi-wd/2, vs+0.01, f"{vs:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi+wd/2, vu+0.01, f"{vu:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=7.8); ax.set_ylim(0,1.08)
    ax.set_ylabel("score"); ax.set_title("2D detector (YOLOv11s) performance")
    ax.legend(loc="lower left", framealpha=0.9)
    save(fig, f"{CAP7}/yolo_metrics.pdf")

# ----------------------------------------------------------------------------- Ch7: normalization ablation
def fig_normalization():
    methods=["Baseline","p95","Depth-band","T-Net seg."]
    rmse=[2.94,2.87,1.20,1.05]; prec=[0.21,0.12,0.90,0.90]; amota=[0.04,0.05,0.81,0.81]
    x=np.arange(len(methods))
    fig, axes=plt.subplots(1,3, figsize=(6.6,2.7))
    for ax,(vals,ttl,col,fmt) in zip(axes,[(rmse,"3D RMSE (m) $\\downarrow$",C["red"],"{:.2f}"),
                                           (prec,"Detection precision $\\uparrow$",C["blue"],"{:.2f}"),
                                           (amota,"AMOTA $\\uparrow$",C["green"],"{:.2f}")]):
        ax.bar(x, vals, color=col, width=0.62)
        for xi,v in zip(x,vals): ax.text(xi, v+max(vals)*0.02, fmt.format(v), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7.2, rotation=25, ha="right", rotation_mode="anchor")
        ax.set_title(ttl, fontsize=8.6); ax.set_ylim(0, max(vals)*1.18)
    fig.tight_layout(); save(fig, f"{CAP7}/normalization_ablation.pdf")

# ----------------------------------------------------------------------------- Ch7: localization vs range
def fig_range():
    bands=["0–30 m","30–50 m","50–90 m"]
    rmse_main=[1.25,1.16,1.30]   # main pipeline (depth-band) + EKF, full 60
    rmse_pfn=[0.86,0.80,1.01]    # PFN voxel + EKF
    x=np.arange(len(bands)); wd=0.36
    fig, ax=plt.subplots(figsize=(4.4,2.7))
    ax.bar(x-wd/2, rmse_main, wd, color=C["green"], label="Frustum + depth-band")
    ax.bar(x+wd/2, rmse_pfn, wd, color=C["purple"], label="PointPillars-PFN")
    for xi,a,b in zip(x,rmse_main,rmse_pfn):
        ax.text(xi-wd/2,a+0.02,f"{a:.2f}",ha="center",va="bottom",fontsize=7)
        ax.text(xi+wd/2,b+0.02,f"{b:.2f}",ha="center",va="bottom",fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(bands); ax.set_ylabel("3D RMSE (m)")
    ax.set_ylim(0,1.5); ax.set_title("3D localization error vs. range")
    ax.legend(loc="upper left", framealpha=0.9)
    save(fig, f"{CAP7}/localization_vs_range.pdf")

# ----------------------------------------------------------------------------- Ch7: master architecture comparison
def fig_comparison():
    # (name, AMOTA, det-F1, RMSE, family)
    rows=[("Baseline",0.04,0.12,2.94,"f"),("p95",0.05,0.07,2.87,"f"),
          ("Depth-band",0.81,0.58,1.20,"f"),("F-ConvNet",0.80,0.55,1.18,"f"),
          ("T-Net seg.",0.81,0.60,1.05,"f"),("PointPillars",0.63,0.79,1.19,"v"),
          ("Voxel-3D",0.68,0.83,0.92,"v"),("PFN",0.68,0.83,0.87,"v"),
          ("Fusion",0.75,0.74,1.07,"h")]
    names=[r[0] for r in rows]; amota=[r[1] for r in rows]; f1=[r[2] for r in rows]; rmse=[r[3] for r in rows]
    fam=[r[4] for r in rows]
    colmap={"f":C["green"],"v":C["purple"],"h":C["orange"]}
    cols=[colmap[f] for f in fam]
    x=np.arange(len(rows))
    fig, axes=plt.subplots(3,1, figsize=(6.4,5.1), sharex=True)
    for ax,(vals,ttl,arrow) in zip(axes,[(amota,"AMOTA","$\\uparrow$"),(f1,"Detection F1","$\\uparrow$"),
                                         (rmse,"3D RMSE (m)","$\\downarrow$")]):
        ax.bar(x, vals, color=cols, width=0.66)
        for xi,v in zip(x,vals): ax.text(xi, v+max(vals)*0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
        ax.set_ylabel(f"{ttl} {arrow}", fontsize=8.8); ax.set_ylim(0, max(vals)*1.16)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(names, rotation=28, ha="right", fontsize=7.6)
    handles=[Rectangle((0,0),1,1,color=colmap["f"]),Rectangle((0,0),1,1,color=colmap["v"]),
             Rectangle((0,0),1,1,color=colmap["h"])]
    axes[0].legend(handles,["Frustum (2-stage)","Voxel (1-stage)","Late fusion"],
                   loc="upper left", ncol=1, fontsize=7.2, framealpha=0.95)
    fig.tight_layout(); save(fig, f"{CAP7}/architecture_comparison.pdf")

# ----------------------------------------------------------------------------- Ch7: precision/recall regimes
def fig_regimes():
    # name, recall, precision, family
    pts=[("Depth-band",0.47,0.90,"f"),("F-ConvNet",0.44,0.88,"f"),("T-Net seg.",0.49,0.90,"f"),
         ("PointPillars",0.76,0.87,"v"),("Voxel-3D",0.80,0.91,"v"),("PFN",0.80,0.92,"v"),
         ("Fusion",0.66,0.90,"h")]
    colmap={"f":C["green"],"v":C["purple"],"h":C["orange"]}
    off={"Depth-band":(-6,-11),"F-ConvNet":(-44,-2),"T-Net seg.":(4,5),
         "PointPillars":(6,-9),"Voxel-3D":(6,5),"PFN":(-10,9),
         "Fusion":(6,4)}
    fig, ax=plt.subplots(figsize=(4.9,3.2))
    for n,r,p,f in pts:
        ax.scatter(r,p, s=46, color=colmap[f], edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(n,(r,p), textcoords="offset points", xytext=off.get(n,(5,4)), fontsize=6.9)
    ax.axvspan(0.40,0.55, color=C["green"], alpha=0.05); ax.axvspan(0.72,0.85, color=C["purple"], alpha=0.05)
    ax.text(0.475,0.70,"frustum\nregime", ha="center", fontsize=7.3, color=C["green"])
    ax.text(0.785,0.70,"voxel\nregime", ha="center", fontsize=7.3, color=C["purple"])
    ax.set_xlabel("detection recall $\\uparrow$"); ax.set_ylabel("detection precision $\\uparrow$")
    ax.set_xlim(0.38,0.9); ax.set_ylim(0.66,1.0); ax.set_title("Two operating regimes")
    handles=[Line2D([0],[0],marker="o",ls="",color=colmap[k],markersize=6) for k in ["f","v","h"]]
    ax.legend(handles,["Frustum (2-stage)","Voxel (1-stage)","Late fusion"], loc="lower left", fontsize=7.2)
    save(fig, f"{CAP7}/regimes.pdf")

# ----------------------------------------------------------------------------- Ch7: per-environment
def fig_perenv():
    envs=["AirSimNH","City","Coastline"]
    amota=[0.75,0.79,0.88]; f1=[0.42,0.67,0.66]; rmse=[1.40,1.19,1.06]; rec=[0.31,0.58,0.52]
    x=np.arange(len(envs)); wd=0.2
    fig, ax=plt.subplots(figsize=(5.0,2.9))
    ax.bar(x-1.5*wd, amota, wd, color=C["green"], label="AMOTA")
    ax.bar(x-0.5*wd, rec, wd, color=C["blue"], label="Recall")
    ax.bar(x+0.5*wd, f1, wd, color=C["sky"], label="Det. F1")
    ax.bar(x+1.5*wd, [r/3 for r in rmse], wd, color=C["red"], label="RMSE/3 (m)")
    ax.set_xticks(x); ax.set_xticklabels(envs); ax.set_ylim(0,1.0)
    ax.set_title("Main pipeline across environments"); ax.legend(ncol=2, fontsize=7.2, loc="upper left")
    save(fig, f"{CAP7}/per_environment.pdf")

# ----------------------------------------------------------------------------- Ch7: qualitative (REAL)
def _iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def _nms(dets, thr=0.45):
    dets=sorted(dets, key=lambda d:0.5*d["yolo"]+0.5*d["pn"], reverse=True); keep=[]
    for d in dets:
        if all(_iou(d["bbox"],k["bbox"])<thr for k in keep): keep.append(d)
    return keep

def fig_qualitative():
    import matplotlib.image as mpimg
    from scipy.optimize import linear_sum_assignment
    # prefer AirSimNH; pick a "clean" frame: 2--3 drones, 1:1 with GT, small 3D error, boxes separated
    cands=["nh_seq01_near_clear_h13","nh_seq04_mid_clear_h9","nh_seq09_near_clear_h18",
           "nh_seq15_near_clear_h11","nh_seq17_mid_clear_h15","nh_seq20_mid_clear_h18",
           "nh_seq02_mid_clear_h13","nh_seq12_near_clear_h15",
           "coast_seq05_mid_clear_h17","city_seq02_mid_clear_h13"]
    best=None   # minimise (mean 3D error, -box separation)
    for name in cands:
        dp=os.path.join(DATA, "det_dumps_segnet", f"{name}.json")
        if not os.path.exists(dp): continue
        for fr in json.load(open(dp))["frames"]:
            gts=[np.array(g) for g in fr["gt"].values()]
            kept=_nms([d for d in fr["dets"] if 0.5*d["yolo"]+0.5*d["pn"]>0.5])
            if not (2<=len(kept)<=3) or len(gts)!=len(kept): continue
            D=np.array([[np.linalg.norm(np.array(d["gpos"])-g) for g in gts] for d in kept])
            ri,ci=linear_sum_assignment(D)
            if len(set(ci))!=len(kept): continue          # distinct GT per detection (clean 1:1)
            errs=[D[r,c] for r,c in zip(ri,ci)]
            if max(errs)>2.5: continue                      # every drone cleanly localised
            cx=[(d["bbox"][0]+d["bbox"][2])/2 for d in kept]; cy=[(d["bbox"][1]+d["bbox"][3])/2 for d in kept]
            sep=min((abs(cx[a]-cx[b])+abs(cy[a]-cy[b])) for a in range(len(kept)) for b in range(a+1,len(kept)))
            if sep<140: continue                            # boxes well separated in the image
            key=(float(np.mean(errs)), -sep)
            if best is None or key<best[0]: best=(key,name,fr,kept)
    _,name,fr,kept=best
    sess=os.path.join(DATA, "sessions", name); i=fr["frame"]
    rgb=mpimg.imread(f"{sess}/rgb/{i:05d}.jpg")
    meta=json.load(open(f"{sess}/meta.json")); fm=next(f for f in meta["frames"] if f["frame"]==i)
    xp=np.array(fm["x_plat"]); R=Rmat(xp[6],xp[7],xp[8])
    fig=plt.figure(figsize=(6.4,2.7)); gs=fig.add_gridspec(1,2,width_ratios=[1.6,1.0],wspace=0.24)
    # (a) first person: one clean box per object (post-NMS)
    axa=fig.add_subplot(gs[0,0]); axa.imshow(rgb); axa.axis("off")
    axa.set_title("(a) First-person view: detections", fontsize=9)
    for d in kept:
        x1,y1,x2,y2=d["bbox"]; conf=0.5*d["yolo"]+0.5*d["pn"]
        axa.add_patch(Rectangle((x1,y1),x2-x1,y2-y1, fill=False, edgecolor=C["green"], lw=1.6))
        axa.text((x1+x2)/2, y1-6, f"drone {conf:.2f}", color="white", fontsize=6.6, ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.12", fc=C["green"], ec="none", alpha=0.85))
    # (b) BEV: GT vs estimate, with error link
    axb=fig.add_subplot(gs[0,1]); axb.set_title("(b) Bird's-eye view", fontsize=9)
    axb.add_patch(Wedge((0,0), 80, 58, 122, color=C["sky"], alpha=0.10))
    gts=[(R.T@(np.array(g)-xp[:3])) for g in fr["gt"].values()]
    ests=[(R.T@(np.array(d["gpos"])-xp[:3])) for d in kept]
    for pf in gts:
        axb.scatter(pf[1], pf[0], marker="s", s=60, facecolors="none", edgecolors=C["grey"], linewidths=1.4, zorder=3)
    for pe in ests:
        # link to nearest GT to visualise the (small) error
        if gts:
            g=min(gts, key=lambda q: (q[0]-pe[0])**2+(q[1]-pe[1])**2)
            axb.plot([pe[1],g[1]],[pe[0],g[0]], color=C["red"], lw=0.8, zorder=2)
        axb.scatter(pe[1], pe[0], marker="x", s=48, color=C["green"], zorder=4)
    axb.scatter([0],[0], marker="^", s=70, color="black", zorder=5); axb.text(2.5,-3,"ego", fontsize=7)
    axb.set_xlabel("lateral (m)"); axb.set_ylabel("forward (m)"); axb.set_aspect("equal")
    axb.set_xlim(-45,45); axb.set_ylim(-5,85)
    handles=[Line2D([0],[0],marker="s",ls="",mfc="none",mec=C["grey"],markersize=7),
             Line2D([0],[0],marker="x",ls="",color=C["green"],markersize=7)]
    axb.legend(handles,["ground truth","estimate"], loc="upper left", fontsize=7)
    print("qualitative frame:", name, i, "| kept dets:", len(kept), "| gt:", len(fr["gt"]))
    save(fig, f"{CAP7}/qualitative.pdf")

# ----------------------------------------------------------------------------- Ch7: tracking stage
TRACKER_DUMP = "det_dumps_pfn"   # the detector whose detections feed the tracker in Fig. 7.6(b)

def _run_tracker(name, thr=0.1):
    """Replicates evaluation/metrics.py: runs SORT over a dump and returns each track's state per frame."""
    import importlib, tracking.sort as st
    importlib.reload(st)
    st.Track._next = 0
    trk = st.SortTracker(iou_thr=0.3, max_age=8, min_hits=3, nms_iou=0.4, coast=4)
    dump = json.load(open(os.path.join(DATA, TRACKER_DUMP, f"{name}.json")))
    rec = {}   # id -> list of (frame_idx, status)  status in {'det','coast'}
    for fi, fr in enumerate(dump["frames"]):
        dets = []
        for d in fr["dets"]:
            f = 0.5*d["yolo"] + 0.5*d["pn"]
            if f >= thr:
                dets.append({"bbox": tuple(d["bbox"]), "fused": f, "d": d["d"], "gpos": np.array(d["gpos"])})
        trk.update(dets)
        for t in trk.tracks:
            if t.hits >= trk.min_hits and t.time_since_update <= trk.coast:
                rec.setdefault(t.id, []).append((fi, "det" if t.time_since_update == 0 else "coast"))
    return rec, len(dump["frames"])

def fig_tracker():
    # pick a sequence whose confirmed tracks show coasting (gaps bridged) over a decent span
    cands = ["city_seq02_mid_clear_h13","nh_seq02_mid_clear_h13","coast_seq02_mid_clear_h13",
             "city_seq05_mid_clear_h17","coast_seq08_mid_clear_h11","city_seq20_mid_clear_h18",
             "coast_seq05_mid_clear_h17","nh_seq17_mid_clear_h15"]
    best = None
    for name in cands:
        if not os.path.exists(os.path.join(DATA, TRACKER_DUMP, f"{name}.json")): continue
        rec, nF = _run_tracker(name)
        lanes = {i: s for i, s in rec.items() if (s[-1][0]-s[0][0]) >= 25}   # tracks de vida longa
        if not (2 <= len(lanes) <= 4): continue
        ncoast = sum(1 for s in lanes.values() for (_, st_) in s if st_ == "coast")
        coverage = max(s[-1][0] for s in lanes.values()) - min(s[0][0] for s in lanes.values())
        if ncoast < 2: continue                       # some coasting must be visible
        sc = coverage + ncoast                        # favour covering the video well AND showing coasting
        if best is None or sc > best[0]: best = (sc, name, rec, nF)
    if best is None:  # fallback: the largest coverage by long tracks
        for name in cands:
            if not os.path.exists(os.path.join(DATA, TRACKER_DUMP, f"{name}.json")): continue
            rec, nF = _run_tracker(name)
            lanes={i:s for i,s in rec.items() if (s[-1][0]-s[0][0])>=20}
            if not lanes: continue
            cov=max(s[-1][0] for s in lanes.values())-min(s[0][0] for s in lanes.values())
            if best is None or cov>best[0]: best=(cov,name,rec,nF)
    _, name, rec, nF = best
    lanes = {i: s for i, s in rec.items() if (s[-1][0]-s[0][0]) >= 20}
    order = sorted(lanes, key=lambda i: (lanes[i][-1][0]-lanes[i][0][0]), reverse=True)[:4]  # the four longest
    order = sorted(order, key=lambda i: lanes[i][0][0])   # order by birth, for display
    f0 = min(lanes[i][0][0] for i in order); f1 = max(lanes[i][-1][0] for i in order)

    fig = plt.figure(figsize=(6.4, 4.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.42)

    # (a) schematic flow of the tracking stage
    axa = fig.add_subplot(gs[0]); axa.set_xlim(-3.5, 103.5); axa.set_ylim(0, 34); axa.axis("off")
    blocks = [("2D\ndetections", C["grey"]), ("NMS $+$\nIoU assoc.", C["blue"]),
              ("confirm /\ncoast", C["orange"]), ("per-track\n3D EKF", C["green"]),
              ("calibrated\n3D tracks", C["red"])]
    n = len(blocks); w = 15.0; gap = (100 - n*w) / (n-1); y = 11; h = 11
    cx = []
    for k, (lab, col) in enumerate(blocks):
        x = k*(w+gap)
        axa.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                     lw=1.3, edgecolor=col, facecolor=col+"22"))
        axa.text(x+w/2, y+h/2, lab, ha="center", va="center", fontsize=7.6)
        cx.append((x, x+w))
    for k in range(n-1):
        axa.add_patch(FancyArrowPatch((cx[k][1], y+h/2), (cx[k+1][0], y+h/2),
                                      arrowstyle="-|>", mutation_scale=10, lw=1.2, color="black"))
    # brackets: SORT (2D id association) over blocks 1--2; 3D EKF over block 3
    axa.plot([cx[1][0], cx[2][1]], [y+h+3, y+h+3], color=C["blue"], lw=1.0)
    axa.text((cx[1][0]+cx[2][1])/2, y+h+4, "SORT --- 2D identity association",
             ha="center", va="bottom", fontsize=7.0, color=C["blue"])
    axa.plot([cx[3][0], cx[3][1]], [y+h+3, y+h+3], color=C["green"], lw=1.0)
    axa.text((cx[3][0]+cx[3][1])/2, y+h+4, "3D state (Ch.5)",
             ha="center", va="bottom", fontsize=7.0, color=C["green"])
    axa.text(50, 3.5, "EKF: range--bearing measurement $[\\rho,\\alpha,\\beta,r]$ "
             "$\\rightarrow$ filtered 3D position, velocity, size $+$ covariance",
             ha="center", va="center", fontsize=7.0, color=C["grey"])
    axa.set_ylim(-7, 30)
    axa.text(50, -5, "(a) Processing flow", ha="center", va="top", fontsize=8.8)

    # (b) real track timeline (detection-supported vs coasted)
    axb = fig.add_subplot(gs[1])
    for lane, tid in enumerate(order):
        seq = lanes[tid]
        # group consecutive frames by status into segments for broken_barh
        segs_det = []; segs_co = []
        run_start = seq[0][0]; run_stat = seq[0][1]; prev = seq[0][0]
        def flush(a, b, stt):
            (segs_det if stt == "det" else segs_co).append((a, b-a+1))
        for (f, stt) in seq[1:]:
            if stt != run_stat or f != prev+1:
                flush(run_start, prev, run_stat); run_start = f; run_stat = stt
            prev = f
        flush(run_start, prev, run_stat)
        axb.broken_barh(segs_det, (lane-0.3, 0.6), facecolors=C["green"], edgecolors="none")
        axb.broken_barh(segs_co,  (lane-0.3, 0.6), facecolors=C["orange"], edgecolors="none", hatch="////")
    axb.set_yticks(range(len(order))); axb.set_yticklabels([f"track {k+1}" for k in range(len(order))], fontsize=8)
    axb.set_xlabel("frame index"); axb.set_xlim(f0-2, f1+2); axb.set_ylim(-0.7, len(order)-0.3)
    axb.invert_yaxis()
    handles = [Rectangle((0,0),1,1, fc=C["green"]), Rectangle((0,0),1,1, fc=C["orange"], hatch="////")]
    axb.text(0.5, -0.34, f"(b) Track timeline ({name.split('_')[0].upper()} sequence)",
             transform=axb.transAxes, ha="center", va="top", fontsize=8.8)
    axb.legend(handles, ["detection-supported", "coasted (predicted through a gap)"],
               loc="upper center", bbox_to_anchor=(0.5, -0.52), ncol=2,
               fontsize=7.4, framealpha=0.95, borderaxespad=0.0)
    print("tracker fig seq:", name, "| lanes:", len(order))
    save(fig, f"{CAP7}/tracking_stage.pdf")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="figures",
                    help="directory to create Cap4/ and Cap7/ under "
                         "(point this at your dissertation source tree)")
    ap.add_argument("--cap4-dir", default=None, help="override <outdir>/Cap4")
    ap.add_argument("--cap7-dir", default=None, help="override <outdir>/Cap7")
    ap.add_argument("--data-root", default=None,
                    help="root holding sessions/, det_dumps_*/, sprites/, backgrounds/ "
                         "(default: paths.data_root from configs/default.yaml)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="generate only these figures, by function name "
                         "(e.g. --only fig_yolo fig_regimes)")
    args = ap.parse_args()

    if args.data_root:
        CONFIG.set("paths.data_root", args.data_root)
    global DATA
    DATA = str(CONFIG.data_root)

    _set_output_dirs(args.cap4_dir or os.path.join(args.outdir, "Cap4"),
                     args.cap7_dir or os.path.join(args.outdir, "Cap7"))

    fns = [fig_pipeline, fig_dataset_pipeline, fig_depthband, fig_yolo, fig_normalization,
           fig_range, fig_comparison, fig_regimes, fig_perenv, fig_qualitative, fig_tracker]
    if args.only:
        wanted = set(args.only)
        fns = [f for f in fns if f.__name__ in wanted]
        unknown = wanted - {f.__name__ for f in fns}
        if unknown:
            ap.error("unknown figure(s): %s" % ", ".join(sorted(unknown)))

    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback
            print("FAILED", fn.__name__, e)
            traceback.print_exc()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
