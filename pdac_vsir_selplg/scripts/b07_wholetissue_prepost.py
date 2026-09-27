#!/usr/bin/env python3
"""M16 - whole-section H&E, immune domains BEFORE vs AFTER TRACER.

Domains are rasterised on a 25 um grid: a bin is drawn for a cell type where
that type reaches >=25% of the bin's entities, with alpha scaled by the local
fraction, so domain structure reads at whole-tissue scale instead of drawing
10^5 markers. Shows that TRACER changes the immune *map*, not just the count.
"""
from __future__ import annotations
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

AFF = json.load(open(S.PDAC6 / "audit_v2/image_affines.json"))
GRID = 25.0
SHOW = ["T cell", "Macrophage cell"]
COLS = ["x", "y", "cell_type"]


def to_img(P, x, y):
    a = AFF[P]; cx, cy = a["px"], a["py"]; s = a["hires_scalef"]
    return (cx[0] * x + cx[1] * y + cx[2]) * s, (cy[0] * x + cy[1] * y + cy[2]) * s


def he(P, step=3):
    a = mpimg.imread(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/binned_outputs/"
                     "square_002um/spatial/tissue_hires_image.png")
    a = (np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a[..., :3]
    out = np.ascontiguousarray(a[::step, ::step]); del a; gc.collect()
    return out, step


def domains(ax, P, d):
    gx = np.floor(d.x / GRID).astype(np.int64)
    gy = np.floor(d.y / GRID).astype(np.int64)
    key = gx * 200000 + gy
    tot = pd.Series(1, index=key).groupby(level=0).size()
    for ct in SHOW:
        m = (d.cell_type == ct).to_numpy()
        if m.sum() == 0:
            continue
        cnt = pd.Series(1, index=key[m]).groupby(level=0).size()
        frac = (cnt / tot.reindex(cnt.index)).fillna(0)
        sel = frac[frac >= 0.25]
        if not len(sel):
            continue
        kx = (sel.index // 200000) * GRID + GRID / 2
        ky = (sel.index % 200000) * GRID + GRID / 2
        ix, iy = to_img(P, kx.to_numpy(), ky.to_numpy())
        a = np.clip(0.28 + 0.66 * sel.to_numpy(), 0, 0.92)
        ax.scatter(ix, iy, s=2.1, c=S.CT[ct], lw=0, alpha=a, rasterized=True,
                   zorder=3 if ct == "T cell" else 2)
        yield ct, int(m.sum()), len(sel)


for P, tag in (("HC05", "responder"), ("HC01", "nonresponder")):
    img, step = he(P)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
    stats = {}
    for ax, arm in zip(axes, ["pre", "post_all"]):
        d = S.load_arm(P, arm, columns=COLS)
        ax.imshow(img, extent=(0, img.shape[1] * step, img.shape[0] * step, 0),
                  zorder=0, interpolation="bilinear")
        stats[arm] = list(domains(ax, P, d))
        ix, iy = to_img(P, d.x.to_numpy(), d.y.to_numpy())
        pad = 0.03 * (ix.max() - ix.min())
        ax.set_xlim(ix.min() - pad, ix.max() + pad)
        ax.set_ylim(iy.max() + pad, iy.min() - pad)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.6)
        nT = next((c for t, c, _ in stats[arm] if t == "T cell"), 0)
        bT = next((b for t, _, b in stats[arm] if t == "T cell"), 0)
        ax.set_xlabel(f"{'10X' if arm=='pre' else 'TRACER'}\n"
                      f"T cells {nT:,}   T domains {bT:,}", fontsize=7.5, labelpad=3)
        del d, ix, iy; gc.collect()
    fig.subplots_adjust(wspace=0.06)
    S.save(fig, f"M16_hd_wholetissue_prepost_{P}_{tag}")
    print(f"[{P}] " + " | ".join(f"{a}: " + ", ".join(f"{t} n={c} bins={b}" for t, c, b in v)
                                for a, v in stats.items()), flush=True)
    del img; gc.collect()

fig, ax = plt.subplots(figsize=(2.6, 0.34)); ax.axis("off")
ax.legend(handles=[Patch(facecolor=S.CT[c], label=S.CT_SHORT[c]) for c in SHOW],
          ncol=2, loc="center", handlelength=1.1, handletextpad=0.4, columnspacing=1.3)
S.save(fig, "M16b_legend_domains")
print("done")
