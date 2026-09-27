#!/usr/bin/env python3
"""Supplementary panels.

S2 TIGIT-NECTIN2 regional niche + interface ROI (moved out of the main figure;
   NECTIN2+ tumour now drawn as a filled high-contrast marker, it was an
   unfilled pink ring that vanished against pink H&E)
S6 secondary checkpoint axes (TIGIT-PVR, GAL9-TIM3) vs the same within-type null
Existing valid supplementary panels are carried across from panels_v3/supp.
The corrected P7 is also re-emitted under its original panels_v3 name.
"""
from __future__ import annotations
import sys, json, gc, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

AFF = json.load(open(S.PDAC6 / "audit_v2/image_affines.json"))
SZ = 300.0
R_CONTACT = 20.0
COLS = ["x", "y", "cell_type", "NECTIN2", "TIGIT", "PVR", "LGALS9", "HAVCR2", "VSIR", "SELPLG"]
NEC_C = "#009E73"        # filled, reads on pink H&E
TIG_C = "#000000"


def to_img(P, x, y):
    a = AFF[P]; cx, cy = a["px"], a["py"]; s = a["hires_scalef"]
    return (cx[0] * x + cx[1] * y + cx[2]) * s, (cy[0] * x + cy[1] * y + cy[2]) * s


def he_crop(P, box, pad=8):
    a = mpimg.imread(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/binned_outputs/"
                     "square_002um/spatial/tissue_hires_image.png")
    a = (np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8) if a.dtype != np.uint8 else a[..., :3]
    c0, c1, r0, r1 = box
    h, w = a.shape[:2]
    C0 = max(int(c0) - pad, 0); C1 = min(int(c1) + pad, w)
    R0 = max(int(r0) - pad, 0); R1 = min(int(r1) + pad, h)
    out = np.ascontiguousarray(a[R0:R1, C0:C1]); del a; gc.collect()
    return out, (C0, C1, R1, R0)


def sets(d, lg, lctx, rg):
    ctx = {"TUMOR": d.cell_type.isin(S.TUMOR), "TAM": d.cell_type == "Macrophage cell"}
    return d[(d[lg] > 0) & ctx[lctx]], d[(d[rg] > 0) & (d.cell_type == "T cell")]


# ---- pick the ROI with the most ligand-receptor pairs within R (post_all) ----
d = S.load_arm("HC08", "post_all", columns=COLS)
L, Rr = sets(d, "NECTIN2", "TUMOR", "TIGIT")
t = cKDTree(Rr[["x", "y"]].to_numpy())
pairs = t.query_ball_point(L[["x", "y"]].to_numpy(), R_CONTACT)
rows = [(int(r.x // SZ), int(r.y // SZ), len(p))
        for (_, r), p in zip(L.iterrows(), pairs) if len(p)]
cand = (pd.DataFrame(rows, columns=["tx", "ty", "n"]).groupby(["tx", "ty"]).n.sum()
        .reset_index().sort_values("n", ascending=False))
cand.to_csv(S.TAB / "roi_candidates_TIGIT_NECTIN2.tsv", sep="\t", index=False)
x0, y0 = float(cand.iloc[0].tx * SZ), float(cand.iloc[0].ty * SZ)
print(f"[S2] TIGIT-NECTIN2 ROI x0={x0:.0f} y0={y0:.0f} pairs={int(cand.iloc[0].n)}", flush=True)
del d, L, Rr; gc.collect()

cs = [to_img("HC08", x0 + dx, y0 + dy) for dx, dy in [(0, 0), (SZ, 0), (0, SZ), (SZ, SZ)]]
box = (min(c[0] for c in cs), max(c[0] for c in cs),
       min(c[1] for c in cs), max(c[1] for c in cs))
img, ext = he_crop("HC08", box)

fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.05), gridspec_kw={"wspace": 0.04})
for j, arm in enumerate(S.ARMS):
    d = S.load_arm("HC08", arm, columns=COLS)
    L, Rr = sets(d, "NECTIN2", "TUMOR", "TIGIT")
    ax = axs[j]
    ax.imshow(img, extent=ext, zorder=0, interpolation="bilinear")
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    cut = lambda q: q[(q.x >= x0) & (q.x < x0 + SZ) & (q.y >= y0) & (q.y < y0 + SZ)]
    cl, cr = cut(L), cut(Rr)
    if len(cl) and len(cr):
        tt = cKDTree(cr[["x", "y"]].to_numpy())
        segs = []
        for _, row in cl.iterrows():
            for k in tt.query_ball_point([row.x, row.y], R_CONTACT):
                a1, b1 = to_img("HC08", row.x, row.y)
                a2, b2 = to_img("HC08", cr.iloc[k].x, cr.iloc[k].y)
                segs.append([(a1, b1), (a2, b2)])
        if segs:
            ax.add_collection(LineCollection(segs, colors="#000000", linewidths=0.8,
                                             alpha=0.85, zorder=8))
    if len(cr):
        ix, iy = to_img("HC08", cr.x.to_numpy(), cr.y.to_numpy())
        ax.scatter(ix, iy, s=22, marker="D", facecolors=TIG_C, edgecolors="white",
                   lw=0.6, zorder=6)
    if len(cl):
        ix, iy = to_img("HC08", cl.x.to_numpy(), cl.y.to_numpy())
        ax.scatter(ix, iy, s=34, facecolors=NEC_C, edgecolors="white", lw=0.8, zorder=7)
    ax.set_xlabel(f"{S.ARMLAB[arm]}\nNECTIN2$^+$ {len(cl)}   TIGIT$^+$ {len(cr)}",
                  fontsize=7, labelpad=2)
    del d, L, Rr; gc.collect()
S.save(fig, "S2_hd_tigit_nectin2_interface", supp=True)
for e, dpi in (("pdf", 600), ("png", 400), ("svg", 600)):
    fig.savefig(S.PDAC6 / "panels_v3" / f"P7_TIGIT_NECTIN2_interface.{e}",
                dpi=dpi, bbox_inches="tight")
plt.close(fig)
del img; gc.collect()

fig, ax = plt.subplots(figsize=(3.2, 0.36)); ax.axis("off")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=NEC_C, mec="white", mew=0.8,
                          ms=6, label="NECTIN2$^+$ tumour"),
                   Line2D([], [], marker="D", ls="", mfc=TIG_C, mec="white", mew=0.8,
                          ms=5.5, label="TIGIT$^+$ T"),
                   Line2D([], [], color="#000000", lw=1.0, label="≤20 µm")],
          ncol=3, loc="center", handlelength=1.1, handletextpad=0.4, columnspacing=1.1)
S.save(fig, "S2b_legend_tigit_nectin2", supp=True)

# ---------------------------------------- S6 secondary checkpoint axes -------
TP = pd.read_csv(S.PDAC6 / "audit_v3/topology.tsv", sep="\t")
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.05))
for ax, axis in zip(axes, ["TIGIT-PVR (tumour)", "GAL9-TIM3"]):
    q = TP[(TP.axis == axis) & (TP.R == 50) & (TP.status == "ok") &
           (TP.arm == "post_all") & TP.enrich_mixed_edges.notna()]
    q = q.set_index("patient").reindex(S.PTS)
    x = np.arange(len(S.PTS))
    v = q.enrich_mixed_edges.to_numpy(dtype=float)
    ax.bar(x, np.nan_to_num(v), 0.6, color=[S.RC[S.RESP[p]] for p in S.PTS])
    ax.axhline(1.0, color="#666666", lw=0.8, ls=(0, (3, 2)))
    for xi, val in zip(x, v):
        if np.isnan(val):
            ax.text(xi, 0.06, "n/a", ha="center", fontsize=6, color="#999999", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(S.PTS, rotation=45, ha="right")
    ax.set_ylabel("Mixed-edge enrichment")
    ax.set_xlabel(axis.replace(" (tumour)", ""), fontsize=7.5)
    ax.set_ylim(0, max(3.0, np.nanmax(v) * 1.2 if np.isfinite(np.nanmax(v)) else 3.0))
fig.subplots_adjust(wspace=0.42)
S.save(fig, "S6_hd_secondary_axes", supp=True)

# ------------------------------- carry across the valid existing supplements --
CARRY = {"S1_boundary_gradient": "S8_hd_boundary_gradient",
         "S2_response_screen_chance": "S3_hd_response_screen_chance",
         "S3_null_comparison": "S4_hd_null_comparison",
         "S4_topology_radius": "S5_hd_topology_radius",
         "S5_label_confidence": "S7_hd_label_confidence",
         "S6_roi_candidates": "S9_hd_roi_candidates"}
src = S.PDAC6 / "panels_v3/supp"
for old, new in CARRY.items():
    for e in ("pdf", "png", "svg"):
        p = src / f"{old}.{e}"
        if p.exists():
            shutil.copy2(p, S.PANEL_SUPP / f"{new}.{e}")
    print(f"  carried {old} -> {new}", flush=True)
print("done")
