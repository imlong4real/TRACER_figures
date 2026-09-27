#!/usr/bin/env python3
"""Xenium mechanistic introducer - the visuals that show what TRACER DOES.

M1 transcript-fate alluvial across the TRACER phase ladder
M3 3-D reconstruction: intact cells vs TRACER-rebuilt partial cells
M4 ROI that is immune-poor under 10X segmentation and immune-resolved after

Restyled from scripts/reproducibility/fig2 onto the light, Arial/Arimo,
colour-blind-safe theme used by the rest of the figure, and re-derived from the
streamed fate tables rather than the original dark-canvas scripts.
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch, Rectangle
from matplotlib.path import Path as MPath
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

SRC = Path("/scratch4/adeshpa6/TRACER/datasets/pancreas_cancer_xenium_10x/"
           "pdac_io_partition_sequential.parquet")
CLS = ["original cell", "partial", "neighboring cell", "unassigned"]
CCOL = {"original cell": "#0072B2", "partial": "#E69F00",
        "neighboring cell": "#009E73", "unassigned": "#C4C4C4"}
CSHORT = {"original cell": "in 10X cell", "partial": "partial cell",
          "neighboring cell": "neighbour cell", "unassigned": "unassigned"}

# ------------------------------------------------------------ M1 alluvial ---
TR = pd.read_csv(S.TAB / "xenium_phase_transitions.tsv", sep="\t")
STEPS = [("input", "phase1", "Prune"), ("phase1", "group", "Group"),
         ("group", "finalize", "Rescue + Stitch")]
FL = pd.read_csv(S.TAB / "xenium_phase_class_counts.tsv", sep="\t")
phases = ["input", "phase1", "group", "finalize"]
tot = FL[FL.phase == "input"].n.sum()

pos = {}
for ph in phases:
    d = FL[FL.phase == ph].set_index("cls").n.reindex(CLS).fillna(0)
    y = 0.0; p = {}
    for c in CLS:
        h = d[c] / tot
        p[c] = (y, y + h); y += h + 0.022
    pos[ph] = p

fig, ax = plt.subplots(figsize=(5.6, 2.6))
X = np.linspace(0, 1, len(phases))
BW = 0.026
for xi, ph in zip(X, phases):
    for c in CLS:
        y0, y1 = pos[ph][c]
        if y1 - y0 <= 0:
            continue
        ax.add_patch(Rectangle((xi - BW / 2, y0), BW, y1 - y0, facecolor=CCOL[c],
                               edgecolor="none", zorder=4))
for (a, b, lab), xa, xb in zip(STEPS, X[:-1], X[1:]):
    t = TR[TR.transition == f"{a}->{b}"]
    off_a = {c: pos[a][c][0] for c in CLS}
    off_b = {c: pos[b][c][0] for c in CLS}
    t = t.sort_values("n", ascending=False)
    for _, r in t.iterrows():
        if r.src not in CCOL or r.dst not in CCOL or r.n / tot < 0.0015:
            continue
        h = r.n / tot
        ya, yb = off_a[r.src], off_b[r.dst]
        off_a[r.src] += h; off_b[r.dst] += h
        xm = (xa + xb) / 2
        verts = [(xa + BW / 2, ya), (xm, ya), (xm, yb), (xb - BW / 2, yb),
                 (xb - BW / 2, yb + h), (xm, yb + h), (xm, ya + h), (xa + BW / 2, ya + h),
                 (xa + BW / 2, ya)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=CCOL[r.dst],
                               alpha=0.42, edgecolor="none", zorder=2))
    ax.text((xa + xb) / 2, 1.10, lab, ha="center", va="bottom", fontsize=7.5,
            style="italic", color="#333333")
for xi, ph, nm in zip(X, phases, ["10X input", "after Prune", "after Group", "TRACER final"]):
    ax.text(xi, -0.055, nm, ha="center", va="top", fontsize=7.5)
for c in CLS:
    y0, y1 = pos["finalize"][c]
    if y1 - y0 > 0.03:
        ax.text(X[-1] + 0.035, (y0 + y1) / 2, f"{CSHORT[c]}  {100*(y1-y0):.0f}%",
                va="center", ha="left", fontsize=7, color=CCOL[c] if c != "unassigned" else "#777777",
                fontweight="bold")
ax.set_xlim(-0.06, 1.30); ax.set_ylim(-0.16, 1.18)
ax.axis("off")
S.save(fig, "M1_xenium_transcript_fate_alluvial")

# --------------------------------------------------------------- M3 3-D ------
# Each entity is drawn as the convex hull of its own transcripts, so a cell reads
# as a reconstructed volume rather than a point cloud. Lower slab: cells the 10X
# segmentation already had. Upper slab: partial cells TRACER rebuilt.
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import colorsys

ROI = dict(x0=4645, y0=3545, size=95)
cols = ["x", "y", "z", "label", "etype_at_finalize"]
pf = pq.ParquetFile(SRC)
keep = []
for i in range(pf.metadata.num_row_groups):
    d = pf.read_row_group(i, columns=cols).to_pandas()
    m = ((d.x >= ROI["x0"]) & (d.x < ROI["x0"] + ROI["size"]) &
         (d.y >= ROI["y0"]) & (d.y < ROI["y0"] + ROI["size"]) &
         d.etype_at_finalize.isin([0, 1]))
    if m.any():
        keep.append(d[m])
    del d
R = pd.concat(keep, ignore_index=True); del keep; gc.collect()
print(f"[M3] ROI transcripts {len(R):,}  entities {R.label.nunique():,}", flush=True)

sz = R.groupby("label").size()
R = R[R.label.isin(sz[sz >= 10].index)].copy()
zr = max(R.z.max() - R.z.min(), 1.0)
LIFT = zr * 1.9


def hue_walk(n, h0, h1, sat, val, seed=0):
    """n distinguishable colours confined to one hue family, so the two slabs
    stay visually separable while individual cells remain distinct."""
    g = 0.6180339887
    out = []
    for i in range(n):
        f = (seed + i * g) % 1.0
        h = h0 + f * (h1 - h0)
        v = val - 0.16 * ((i * 0.37) % 1.0)
        sfac = sat - 0.18 * ((i * 0.71) % 1.0)
        out.append(colorsys.hsv_to_rgb(h % 1.0, sfac, v))
    return out


fig = plt.figure(figsize=(3.8, 3.9))
ax = fig.add_subplot(111, projection="3d")
counts = {}
for etype, lift, h0, h1, sat, val, seed in (
        (0, 0.0,  0.50, 0.62, 0.62, 0.88, 0.31),      # intact  -> blue/teal family
        (1, LIFT, 0.02, 0.12, 0.92, 0.97, 0.11)):     # partial -> orange/red family
    sub = R[R.etype_at_finalize == etype]
    ids = list(sub.label.unique())
    counts[etype] = len(ids)
    pal = hue_walk(len(ids), h0, h1, sat, val, seed)
    for c, colr in zip(ids, pal):
        p = sub[sub.label == c][["x", "y", "z"]].to_numpy().astype(float)
        if len(p) < 6:
            continue
        p[:, 2] = p[:, 2] + lift
        p = p + np.random.default_rng(abs(hash(c)) % 2**31).normal(0, 0.06, p.shape)
        try:
            h = ConvexHull(p)
        except Exception:
            continue
        tri = [p[s] for s in h.simplices]
        pc = Poly3DCollection(tri, alpha=0.38, facecolor=colr, edgecolor=colr,
                              linewidths=0.2)
        ax.add_collection3d(pc)
ax.set_xlim(R.x.min(), R.x.max()); ax.set_ylim(R.y.min(), R.y.max())
ax.set_zlim(R.z.min() - 1, R.z.max() + LIFT + 1)
ax.set_axis_off()
ax.view_init(elev=14, azim=-60)
ax.set_box_aspect((1, 1, 1.15))
ax.text2D(0.00, 0.955, f"TRACER partial cells   n={counts[1]:,}", transform=ax.transAxes,
          fontsize=8, color="#C77800", fontweight="bold")
ax.text2D(0.00, 0.030, f"cells 10X already had   n={counts[0]:,}", transform=ax.transAxes,
          fontsize=8, color="#2E6E9E", fontweight="bold")
S.save(fig, "M3_xenium_3d_reconstruction")
del R; gc.collect()

# ----------------------------------------------------- M4 immune-resolved ROI -
def load(n):
    d = pd.read_parquet(S.XEN / f"vista_hypoxia_{n}.parquet")
    return d[d.lt_conf >= 0.5]


IMM = ["T cell", "Macrophage cell", "B cell"]
do, dt = load("original"), load("tracer")
W = 220.0
xs = np.arange(do.centroid_x.min(), do.centroid_x.max() - W, 110)
ys = np.arange(do.centroid_y.min(), do.centroid_y.max() - W, 110)


def count(d, x0, y0):
    m = ((d.centroid_x >= x0) & (d.centroid_x < x0 + W) &
         (d.centroid_y >= y0) & (d.centroid_y < y0 + W))
    return int(m.sum()), int((m & d.cell_type.isin(IMM)).sum())


best = None
for x0 in xs:
    for y0 in ys:
        no, io = count(do, x0, y0)
        nt, it = count(dt, x0, y0)
        if no < 60 or io > 12:            # must be immune-POOR under 10X
            continue
        gain = it - io
        if best is None or gain > best[0]:
            best = (gain, x0, y0, io, it, no, nt)
gain, x0, y0, io, it, no, nt = best
print(f"[M4] ROI x0={x0:.0f} y0={y0:.0f}  immune {io} -> {it}  (all cells {no} -> {nt})",
      flush=True)

fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.35))
for ax, d, lab in ((axes[0], do, "10X"), (axes[1], dt, "TRACER")):
    m = ((d.centroid_x >= x0) & (d.centroid_x < x0 + W) &
         (d.centroid_y >= y0) & (d.centroid_y < y0 + W))
    r = d[m]
    o = r[~r.cell_type.isin(IMM)]
    ax.scatter(o.centroid_x, o.centroid_y, s=5, c="#DFDFDF", lw=0, rasterized=True)
    for ct in IMM:
        q = r[r.cell_type == ct]
        ax.scatter(q.centroid_x, q.centroid_y, s=17, c=S.CT[ct], lw=0.3,
                   edgecolors="white", label=S.CT_SHORT[ct], zorder=3)
    ni = int(r.cell_type.isin(IMM).sum())
    ax.set_xlim(x0, x0 + W); ax.set_ylim(y0, y0 + W); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{lab}\nimmune {ni}", fontsize=7.5)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7)
axes[1].plot([x0 + W - 60, x0 + W - 10], [y0 + 10, y0 + 10], color="black", lw=2.2)
axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(-0.06, 1.16),
               handlelength=0.9, handletextpad=0.3, columnspacing=0.9, markerscale=1.1)
fig.subplots_adjust(wspace=0.08)
S.save(fig, "M4_xenium_roi_immune_resolved")
print("done")
