#!/usr/bin/env python3
"""VisiumHD VISTA-PSGL1 spatial panels.

M12 ROI triptych  pre | whole | whole+partial, centred on the largest checkpoint
    domain (largest mixed connected component at R=50 um in post_all). The ROI
    rule is fixed in advance and reported, so it is not outcome-cherry-picked.
M13 whole-tissue H&E with the checkpoint populations overlaid and the ROI boxed.

Entity positions are centroids. The H&E underlay is ~5 um/pixel (the highest
resolution 10x distributes for these sections) and does NOT resolve membranes;
it is anatomical context only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

AFF = json.load(open(S.PDAC6 / "audit_v2/image_affines.json"))
R = 50.0
WIN = 400.0
COLS = ["x", "y", "cell_type", "VSIR", "SELPLG"]


def to_img(P, x, y):
    a = AFF[P]; cx, cy = a["px"], a["py"]; s = a["hires_scalef"]
    return (cx[0] * x + cx[1] * y + cx[2]) * s, (cy[0] * x + cy[1] * y + cy[2]) * s


def he(P, step=1):
    """H&E as uint8 (mpimg returns float32; the 5 GB login cgroup cannot hold
    several float copies of a 6000 px image)."""
    a = mpimg.imread(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/binned_outputs/"
                     "square_002um/spatial/tissue_hires_image.png")
    if a.dtype != np.uint8:
        a = (np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8)
    else:
        a = a[..., :3]
    return np.ascontiguousarray(a[::step, ::step])


def crop(img, x0, x1, y0, y1, pad=6):
    """Sub-image plus the imshow extent, so only the ROI is resident."""
    h, w = img.shape[:2]
    c0 = max(int(np.floor(min(x0, x1))) - pad, 0); c1 = min(int(np.ceil(max(x0, x1))) + pad, w)
    r0 = max(int(np.floor(min(y0, y1))) - pad, 0); r1 = min(int(np.ceil(max(y0, y1))) + pad, h)
    return np.ascontiguousarray(img[r0:r1, c0:c1]), (c0, c1, r1, r0)


def nodes(d):
    lig = (d.cell_type == "Macrophage cell").to_numpy() & (d.VSIR.to_numpy() > 0)
    rec = (d.cell_type == "T cell").to_numpy() & (d.SELPLG.to_numpy() > 0)
    return lig, rec


def graph(xy, is_lig, R):
    if len(xy) < 2:
        return np.zeros((0, 2), int), np.zeros(len(xy), int)
    pr = cKDTree(xy).query_pairs(R, output_type="ndarray")
    if len(pr) == 0:
        return pr, np.arange(len(xy))
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(xy),) * 2)
    _, lab = connected_components(g, directed=False)
    return pr, lab


def roi_centre(P):
    """Centre of the largest mixed component in post_all - fixed rule."""
    d = S.load_arm(P, "post_all", columns=COLS)
    lig, rec = nodes(d)
    xy = np.vstack([d[["x", "y"]].to_numpy()[lig], d[["x", "y"]].to_numpy()[rec]])
    il = np.r_[np.ones(lig.sum(), bool), np.zeros(rec.sum(), bool)]
    pr, lab = graph(xy, il, R)
    df = pd.DataFrame({"lab": lab, "lig": il})
    hb = df.groupby("lab").lig.nunique() == 2
    ml = list(hb[hb].index)
    if not ml:
        return None
    sz = df[df.lab.isin(ml)].groupby("lab").size()
    best = sz.idxmax()
    sel = xy[lab == best]
    return sel.mean(0), int(sz.max())


def draw(ax, P, d, x0, y0, img, show_ctx=True):
    lig, rec = nodes(d)
    m = ((d.x >= x0) & (d.x < x0 + WIN) & (d.y >= y0) & (d.y < y0 + WIN)).to_numpy()
    ix, iy = to_img(P, d.x.to_numpy(), d.y.to_numpy())
    cxs = np.array([x0, x0 + WIN, x0, x0 + WIN])
    cys = np.array([y0, y0, y0 + WIN, y0 + WIN])
    xa, ya = to_img(P, cxs, cys)
    sub, ext = crop(img, xa.min(), xa.max(), ya.min(), ya.max())
    ax.imshow(sub, extent=ext, zorder=0, interpolation="bilinear")
    if show_ctx:
        k = m & ~lig & ~rec
        ax.scatter(ix[k], iy[k], s=0.7, c="#5A5A5A", alpha=0.30, lw=0,
                   rasterized=True, zorder=1)
    # links between the two populations inside the ROI
    sub = np.flatnonzero(m & (lig | rec))
    if len(sub) > 1:
        xy = np.c_[d.x.to_numpy()[sub], d.y.to_numpy()[sub]]
        il = lig[sub]
        pr = cKDTree(xy).query_pairs(R, output_type="ndarray")
        if len(pr):
            mx = pr[il[pr[:, 0]] != il[pr[:, 1]]]
            if len(mx):
                segs = [[(ix[sub[a]], iy[sub[a]]), (ix[sub[b]], iy[sub[b]])] for a, b in mx]
                ax.add_collection(LineCollection(segs, colors=S.LINK_C, linewidths=1.1,
                                                 alpha=0.85, zorder=4))
    for msk, col, mk, sz in ((lig & m, S.VISTA_C, "o", 26), (rec & m, S.PSGL1_C, "^", 30)):
        ax.scatter(ix[msk], iy[msk], s=sz, c=col, marker=mk, lw=0.5,
                   edgecolors="white", zorder=5)
    ax.set_xlim(xa.min(), xa.max()); ax.set_ylim(ya.max(), ya.min())
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.7); s_.set_color("#333333")
    return int((lig & m).sum()), int((rec & m).sum())


rois = {}
for P in S.PTS:
    c = roi_centre(P)
    if c is not None:
        rois[P] = dict(cx=float(c[0][0]), cy=float(c[0][1]), comp_size=c[1])
        print(f"[{P}] largest domain {c[1]} cells at ({c[0][0]:.0f}, {c[0][1]:.0f}) um", flush=True)
json.dump(rois, open(S.TAB / "vista_roi_centres.json", "w"), indent=1)

# ---------------------------------------------------- M12 ROI triptych -------
for P, tag in (("HC08", "responder"), ("HC01", "nonresponder")):
    r = rois[P]; x0, y0 = r["cx"] - WIN / 2, r["cy"] - WIN / 2
    img = he(P)
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.05))
    for ax, arm in zip(axes, S.ARMS):
        d = S.load_arm(P, arm, columns=COLS)
        nl, nr = draw(ax, P, d, x0, y0, img)
        del d
        ax.set_xlabel(f"{S.ARMLAB[arm]}\nVISTA$^+$ {nl}   PSGL-1$^+$ {nr}", fontsize=7,
                      labelpad=3)
    # 100 um scale bar, bottom-right, in image pixels
    ax = axes[-1]
    a = AFF[P]
    px_per_um = np.hypot(a["px"][0], a["py"][0]) * a["hires_scalef"]
    xl, yl = ax.get_xlim(), ax.get_ylim()
    L = 100.0 * px_per_um
    xr = xl[1] - 0.06 * (xl[1] - xl[0]); yb = yl[0] - 0.075 * (yl[0] - yl[1])
    ax.plot([xr - L, xr], [yb, yb], color="black", lw=2.4, zorder=9,
            solid_capstyle="butt")
    fig.subplots_adjust(wspace=0.06)
    S.save(fig, f"M12_hd_vista_roi_{P}_{tag}")
    del img; gc.collect()

# --------------------------------------------------- M13 whole tissue --------
for P, tag in (("HC08", "responder"), ("HC01", "nonresponder")):
    d = S.load_arm(P, "post_all", columns=COLS)
    lig, rec = nodes(d)
    STEP = 3
    img = he(P, step=STEP)
    ix, iy = to_img(P, d.x.to_numpy(), d.y.to_numpy())
    fig, ax = plt.subplots(figsize=(3.1, 3.1))
    ax.imshow(img, extent=(0, img.shape[1] * STEP, img.shape[0] * STEP, 0),
              zorder=0, interpolation="bilinear")
    ctx = np.flatnonzero(~lig & ~rec)
    if len(ctx) > 60000:                     # visually identical at s=0.1/alpha=0.1,
        rs = np.random.default_rng(0)        # and keeps the raster layer inside the
        ctx = rs.choice(ctx, 60000, replace=False)   # 5 GB login cgroup
    ax.scatter(ix[ctx], iy[ctx], s=0.12, c="#6A6A6A", alpha=0.12,
               lw=0, rasterized=True, zorder=1)
    ax.scatter(ix[lig], iy[lig], s=5.5, c=S.VISTA_C, lw=0, alpha=0.95,
               rasterized=True, zorder=3)
    ax.scatter(ix[rec], iy[rec], s=7.0, c=S.PSGL1_C, marker="^", lw=0, alpha=0.95,
               rasterized=True, zorder=4)
    r = rois[P]; x0, y0 = r["cx"] - WIN / 2, r["cy"] - WIN / 2
    bx, by = to_img(P, np.array([x0, x0 + WIN]), np.array([y0, y0 + WIN]))
    ax.add_patch(Rectangle((min(bx), min(by)), abs(bx[1] - bx[0]), abs(by[1] - by[0]),
                           fill=False, edgecolor="#111111", lw=1.3, zorder=6))
    pad = 0.03 * (ix.max() - ix.min())
    ax.set_xlim(ix.min() - pad, ix.max() + pad)
    ax.set_ylim(iy.max() + pad, iy.min() - pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(True); s_.set_linewidth(0.6)
    S.save(fig, f"M13_hd_vista_wholetissue_{P}_{tag}")
    del img, d; gc.collect()

# ----------------------------------------------------------- legend ---------
fig, ax = plt.subplots(figsize=(3.4, 0.36)); ax.axis("off")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.VISTA_C, ms=5,
                          label="VISTA$^+$ TAM"),
                   Line2D([], [], marker="^", ls="", color=S.PSGL1_C, ms=5.5,
                          label="PSGL-1$^+$ T"),
                   Line2D([], [], color=S.LINK_C, lw=1.4, label="≤50 µm link")],
          ncol=3, loc="center", handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
S.save(fig, "M12b_legend_vista")
print("done")
