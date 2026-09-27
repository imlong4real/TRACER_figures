#!/usr/bin/env python3
"""M15 - VisiumHD ROI: 10X segmentation polygons vs TRACER entity footprints.

Left to right: H&E | 10X cell polygons | TRACER whole-cell footprints |
                    + reconstructed partial-cell footprints.

The 10X panel draws the TRUE vendor segmentation polygons. The TRACER panels
draw INFERRED footprints - the convex hull of each entity's own transcripts -
and are labelled as such; the ~5 um/px H&E does not resolve membranes and is
anatomical context only.

Memory-safe rewrite of audit_v3/42_panels_roi.py (that version json-loads the
whole 98 MB geojson, ~2 GB resident, and OOMs in the shared login cgroup).
Also re-emits the fixed panel into panels_v3/ under its original name.
"""
from __future__ import annotations
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S
import geoutil

AFF = json.load(open(S.PDAC6 / "audit_v2/image_affines.json"))
SRC = Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
SZ = 400.0


def to_img(P, x, y):
    a = AFF[P]; cx, cy = a["px"], a["py"]; s = a["hires_scalef"]
    return (cx[0] * x + cx[1] * y + cx[2]) * s, (cy[0] * x + cy[1] * y + cy[2]) * s


def box_of(P, x0, y0):
    cs = [to_img(P, x0 + dx, y0 + dy) for dx, dy in [(0, 0), (SZ, 0), (0, SZ), (SZ, SZ)]]
    return (min(c[0] for c in cs), max(c[0] for c in cs),
            min(c[1] for c in cs), max(c[1] for c in cs))


def he_crop(P, box, pad=8):
    a = mpimg.imread(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/binned_outputs/"
                     "square_002um/spatial/tissue_hires_image.png")
    if a.dtype != np.uint8:
        a = (np.clip(a[..., :3], 0, 1) * 255).astype(np.uint8)
    else:
        a = a[..., :3]
    c0, c1, r0, r1 = box
    h, w = a.shape[:2]
    C0 = max(int(c0) - pad, 0); C1 = min(int(c1) + pad, w)
    R0 = max(int(r0) - pad, 0); R1 = min(int(r1) + pad, h)
    out = np.ascontiguousarray(a[R0:R1, C0:C1])
    del a; gc.collect()
    return out, (C0, C1, R1, R0)


def roi_transcripts(P, x0, y0):
    f = SRC / P / "data/tracer_results/outputs/transcripts_tracer_refined.parquet"
    pf = pq.ParquetFile(f); out = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["x", "y", "tracer_id", "_etype"])
        x = t.column("x").to_numpy(); y = t.column("y").to_numpy()
        m = (x >= x0) & (x < x0 + SZ) & (y >= y0) & (y < y0 + SZ)
        if m.any():
            out.append(pd.DataFrame({
                "x": x[m], "y": y[m],
                "tracer_id": t.column("tracer_id").to_pandas()[m].astype(str).to_numpy(),
                "etype": t.column("_etype").to_pandas()[m].astype(str).to_numpy()}))
        del t
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def hull_polys(sub, cmap, P):
    verts, cols = [], []
    for k, g in sub.groupby("tracer_id"):
        if len(g) < 3:
            continue
        pts = g[["x", "y"]].to_numpy()
        try:
            h = ConvexHull(pts)
        except Exception:
            continue
        ix, iy = to_img(P, pts[h.vertices, 0], pts[h.vertices, 1])
        verts.append(np.column_stack([ix, iy]))
        cols.append(S.CT.get(cmap.get(k), "#FFFFFF00"))
    return verts, cols


def frame(ax, img, ext, box):
    ax.imshow(img, extent=ext, zorder=0, interpolation="bilinear")
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)


roi = pd.read_csv(S.PDAC6 / "audit_v2/roi_top_per_patient.tsv", sep="\t").groupby("patient").head(1)
for P, tag, v3name in [("HC05", "M15_hd_roi_HC05_responder", "P3_roi_HC05_responder"),
                       ("HC01", "M15b_hd_roi_HC01_nonresponder", "P3b_roi_HC01_nonresponder")]:
    r = roi[roi.patient == P].iloc[0]
    x0, y0 = float(r.x0), float(r.y0)
    box = box_of(P, x0, y0)
    img, ext = he_crop(P, box)
    tx = roi_transcripts(P, x0, y0)
    a = AFF[P]; s = a["hires_scalef"]
    gj = f"/scratch4/adeshpa6/PDAC_Long/{P}/data/segmented_outputs/cell_segmentations.geojson"
    vp = geoutil.polys_in_window(gj, box[0] / s, box[1] / s, box[2] / s, box[3] / s, pad=120)
    cols5 = ["entity_id", "x", "y", "cell_type"]
    cpre = pd.read_parquet(S.SPAT / f"{P}_pre_spatial.parquet", columns=cols5)
    cpw = pd.read_parquet(S.SPAT / f"{P}_post_whole_spatial.parquet", columns=cols5)
    cpp = pd.read_parquet(S.SPAT / f"{P}_post_partial_spatial.parquet", columns=cols5)
    cut = lambda d: d[(d.x >= x0) & (d.x < x0 + SZ) & (d.y >= y0) & (d.y < y0 + SZ)]
    cpre, cpw, cpp = cut(cpre), cut(cpw), cut(cpp)
    tmap = {str(e): c for e, c in zip(cpre.entity_id, cpre.cell_type)}
    wmap = {str(e): c for e, c in zip(cpw.entity_id, cpw.cell_type)}
    pmap = {str(e): c for e, c in zip(cpp.entity_id, cpp.cell_type)}

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.04})
    frame(axs[0], img, ext, box); axs[0].set_xlabel("H&E", fontsize=7.5, labelpad=2)
    ax = axs[1]; frame(ax, img, ext, box)
    verts, cols = [], []
    for cid, c in vp.items():
        ct = tmap.get(f"cellid_{cid:09d}-1")
        verts.append(c * s); cols.append(S.CT.get(ct, "#FFFFFF00") if ct else "#FFFFFF00")
    ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors="#3A3A3A",
                                     linewidths=0.22, alpha=0.78, zorder=3))
    ax.set_xlabel(f"10X polygons\nT = {int((cpre.cell_type=='T cell').sum())}",
                  fontsize=7.5, labelpad=2)
    ax = axs[2]; frame(ax, img, ext, box)
    wv, wc = hull_polys(tx[(tx.etype == "cell") & (tx.tracer_id != "-1")], wmap, P)
    ax.add_collection(PolyCollection(wv, facecolors=wc, edgecolors="#3A3A3A",
                                     linewidths=0.22, alpha=0.78, zorder=3))
    ax.set_xlabel(f"TRACER whole$^*$\nT = {int((cpw.cell_type=='T cell').sum())}",
                  fontsize=7.5, labelpad=2)
    ax = axs[3]; frame(ax, img, ext, box)
    ax.add_collection(PolyCollection(wv, facecolors=wc, edgecolors="#3A3A3A",
                                     linewidths=0.22, alpha=0.62, zorder=3))
    pv, _ = hull_polys(tx[(tx.etype == "partial") & (tx.tracer_id != "-1")], pmap, P)
    ax.add_collection(PolyCollection(pv, facecolors="none", edgecolors="#000000",
                                     linewidths=0.24, alpha=0.6, zorder=4))
    ax.set_xlabel(f"+ partial$^*$\n+{int((cpp.cell_type=='T cell').sum())} T",
                  fontsize=7.5, labelpad=2)
    S.save(fig, tag)
    # re-emit the corrected panel under its original panels_v3 name
    for ext_, dpi in (("pdf", 600), ("png", 400), ("svg", 600)):
        fig.savefig(S.PDAC6 / "panels_v3" / f"{v3name}.{ext_}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {P}: 10X polys {len(vp)}, ROI tx {len(tx):,}, "
          f"T {int((cpre.cell_type=='T cell').sum())} -> "
          f"{int((cpw.cell_type=='T cell').sum())} (+{int((cpp.cell_type=='T cell').sum())})",
          flush=True)
    del img, tx, vp, cpre, cpw, cpp; gc.collect()

fig, ax = plt.subplots(figsize=(5.4, 0.42)); ax.axis("off")
keys = ["T cell", "Macrophage cell", "B cell", "Ductal cell type 2",
        "Fibroblast cell", "Endothelial cell", "Acinar cell"]
ax.legend(handles=[Line2D([], [], marker="s", ls="", color=S.CT[k], ms=6,
                          label=S.CT_SHORT[k]) for k in keys] +
                  [Line2D([], [], marker="s", ls="", mfc="none", mec="#000000", ms=6,
                          label="partial")],
          ncol=8, loc="center", handlelength=1.0, handletextpad=0.35, columnspacing=1.0)
ax.text(0.5, -0.55, "$^*$TRACER footprints are inferred from each entity's transcripts",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="#555555")
S.save(fig, "M15c_legend_celltype")
print("done")
