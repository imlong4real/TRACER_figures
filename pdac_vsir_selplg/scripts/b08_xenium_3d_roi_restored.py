#!/usr/bin/env python3
"""M3 (restored) - the preferred Xenium 3-D ROI rendering.

Imports the ORIGINAL scripts/reproducibility/fig2/build_3d_roi.py and reuses its
geometry and rendering verbatim: `pick_roi`, `cell_mesh` (Open3D convex hull ->
Loop subdivision x1 -> Laplacian smoothing x8, lambda 0.5), `shade`
(light [0.3,0.4,1.0]), `make_cells`, `finalize` and `render` (elev=34, azim=-62,
box_aspect (1,1,0.6), z exaggeration 1.7x, the same per-class alpha/edge rules
and the same dark canvas + cell-type palette).

Two deliberate differences:
  1. the legend entry reads "new immune-tumour proximity (<25 um)" instead of
     "new immune-tumour contact" - the highlight rule is a centroid-to-centroid
     xy distance < 25 um, not a membrane contact;
  2. transcripts are streamed row-group-wise instead of `pd.read_parquet` on the
     whole 398 MB table, which needs several GB and OOMs in the shared 5 GB
     login cgroup. The ROI selection is identical (same qv >= 20 and same window
     test, applied to the same globally row-aligned tables).

Type is Arimo (metric Arial) to match the rest of the campaign; geometry,
camera, lighting, transparency and colours are unchanged.
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc

FIG2DIR = Path(__file__).resolve().parents[1] / "helpers"
sys.path.insert(0, str(FIG2DIR))
import fig2_style as FS
import build_3d_roi as B          # the preferred rendering code, reused as-is

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as PS

for _f in Path("/home/lyuan13/scr4_adeshpa6/tracer_campaign/fonts").glob("Arimo-*.ttf"):
    _fm.fontManager.addfont(str(_f))

DSET = Path("/home/lyuan13/scr4_adeshpa6/TRACER/datasets/pancreas_cancer_xenium_10x")
TRANS = DSET / "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs/transcripts.parquet"
PART = DSET / "pdac_io_partition_sequential.parquet"
FIG2 = DSET / "processed/fig2"
W = B.W
PROX_UM = 25.0                    # the threshold build_3d_roi actually uses


def roi_stream(x0, y0):
    """Same rows `pd.read_parquet(...)[m]` would give, one row group at a time."""
    pf = pq.ParquetFile(TRANS)
    keep, offs = [], []
    off = 0
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["x_location", "y_location", "z_location",
                                          "cell_id", "qv"])
        x = t.column("x_location").to_numpy(); y = t.column("y_location").to_numpy()
        m = ((x >= x0) & (x < x0 + W) & (y >= y0) & (y < y0 + W) &
             (t.column("qv").to_numpy() >= 20))
        if m.any():
            keep.append(pd.DataFrame({
                "x_location": x[m], "y_location": y[m],
                "z_location": t.column("z_location").to_numpy()[m],
                "cell_id": t.column("cell_id").to_pandas()[m].astype(str).to_numpy()}))
            offs.append(np.flatnonzero(m) + off)
        off += t.num_rows
        del t
    sub = pd.concat(keep, ignore_index=True)
    idx = np.concatenate(offs)
    # TRACER label for exactly those global row positions
    lab = np.empty(len(idx), dtype=object)
    pl = pq.ParquetFile(PART)
    off = 0
    for i in range(pl.metadata.num_row_groups):
        n = pl.metadata.row_group(i).num_rows
        sel = np.flatnonzero((idx >= off) & (idx < off + n))
        if len(sel):
            col = pl.read_row_group(i, columns=["label"]).column("label").to_pandas()
            lab[sel] = col.values[idx[sel] - off]
            del col
        off += n
    sub["label"] = lab
    return sub


def main():
    at = sc.read_h5ad(FIG2 / "tracer_annotated.h5ad")
    ao = sc.read_h5ad(FIG2 / "original_annotated.h5ad")
    score, x0, y0, npart, nduct = B.pick_roi(at)
    print(f"ROI x0={x0:.0f} y0={y0:.0f} W={W:.0f} | partial-immune={npart} ductal={nduct}",
          flush=True)

    sub = roi_stream(x0, y0)
    sub["x"] = sub.x_location - x0
    sub["y"] = sub.y_location - y0
    sub["z"] = sub.z_location
    print(f"  ROI transcripts: {len(sub):,}", flush=True)

    ct_o = ao.obs["cell_type"].to_dict()
    ct_t = at.obs["cell_type"].to_dict()
    cls_t = at.obs["entity_class"].to_dict()
    orig_cells = B.finalize(B.make_cells(sub, "cell_id", ct_o, "original"))
    trac = sub[sub.label.isin(set(cls_t))]
    trac_cells = B.finalize(B.make_cells(trac, "label", ct_t, "tracer", cls_t))
    del at, ao; gc.collect()

    diag = pd.concat([B.diag_frame(orig_cells), B.diag_frame(trac_cells)], ignore_index=True)
    diag.to_csv(PS.TAB / "xenium_3d_roi_cells.tsv", sep="\t", index=False)
    for name, cc in [("original", orig_cells), ("tracer", trac_cells)]:
        print(f"  {name}: {len(cc)} cells | hull {sum(not c['fallback'] for c in cc)} | "
              f"fallback {sum(c['fallback'] for c in cc)} | "
              f"partial {sum(c['partial'] for c in cc)}", flush=True)

    draw_o = [c for c in orig_cells if c["rendered"]]
    draw_t = [c for c in trac_cells if c["rendered"]]

    highlight = []
    duct = [c for c in draw_t if c["cell_type"].startswith("Ductal") and not c["partial"]]
    for c in draw_t:
        if c["partial"] and c["cell_type"] in B.IMMUNE and duct:
            d = min(duct, key=lambda q: np.linalg.norm(q["centroid"][:2] - c["centroid"][:2]))
            if np.linalg.norm(d["centroid"][:2] - c["centroid"][:2]) < PROX_UM:
                highlight.append((c["centroid"], d["centroid"]))
    print(f"  proximity links (<{PROX_UM:.0f} um): {len(highlight)}", flush=True)

    FS.use_dark()
    rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arimo", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    zscale = 1.7
    fig = plt.figure(figsize=(15.5, 7.6), facecolor=FS.BG)
    axA = fig.add_subplot(121, projection="3d"); axA.set_box_aspect((1, 1, 0.6))
    axB = fig.add_subplot(122, projection="3d"); axB.set_box_aspect((1, 1, 0.6))
    B.render(axA, draw_o, zscale, "a   Original 10x segmentation")
    B.render(axB, draw_t, zscale, "b   TRACER (complete + reconstructed partial)", highlight)
    for ax in (axA, axB):
        ax.set_xlim(0, W); ax.set_ylim(0, W); ax.set_zlim(0, sub.z.max() * zscale)

    leg = [Patch(facecolor=FS.CELLTYPE_COLORS[t], label=t.replace(" cell", ""))
           for t in ["Ductal cell type 2", "Macrophage cell", "T cell", "B cell",
                     "Endothelial cell", "Fibroblast cell", "Stellate cell"]]
    leg += [Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
                   markeredgecolor="none", alpha=0.4, markersize=8,
                   label="sparse partial (fallback)"),
            Line2D([0], [0], color="w", lw=1.3, ls=(0, (2, 2)),
                   label=f"new immune–tumour proximity (<{PROX_UM:.0f} µm)")]
    fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=11,
               labelcolor=FS.INK, bbox_to_anchor=(0.5, 0.005),
               handlelength=1.6, handletextpad=0.7, columnspacing=2.0)
    # info line sits ABOVE both panel titles: at loc="left" the panel-b title
    # starts near x=0.55 and collided with a centred line at the same height.
    fig.text(0.5, 0.988, f"{W:.0f} µm ROI · "
             f"{sum(c['partial'] for c in draw_t)} reconstructed partial cells · "
             f"z exaggerated {zscale:.1f}×",
             ha="center", va="top", color=FS.INK_SOFT, fontsize=11)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.14, wspace=0.0)
    for ext, dpi in (("pdf", 600), ("png", 450), ("svg", 600)):
        fig.savefig(PS.PANEL_MAIN / f"M3_xenium_3d_roi.{ext}", dpi=dpi,
                    facecolor=FS.BG, bbox_inches="tight")
    plt.close(fig)
    print("  wrote main/M3_xenium_3d_roi.{pdf,png,svg}", flush=True)


if __name__ == "__main__":
    main()
