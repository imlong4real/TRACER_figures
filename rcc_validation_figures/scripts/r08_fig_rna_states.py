#!/usr/bin/env python
"""r08 - Supp RCC S7: transcriptional coherence of 10x, TRACER whole, TRACER
partial and TRACER whole+partial entities (reads r07 outputs only).

a  one shared UMAP (fitted once, every arm projected), coloured by transferred
   cell type; identical axes in all four panels
b  canonical-marker dot plot: same genes, same type order, same CP10K+log1p
   normalisation, same colour and size scales in all four arms
c  per type x arm: entities, low-confidence fraction, strict marker support,
   and kNN label coherence (with whole cells thinned to partial depth as the
   depth-matched reference).  Prespecified flags are outlined, not removed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from r07_rna_states import CANONICAL, TYPES

FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
TAB, OBJ, PAN = FINAL / "tables", FINAL / "objects/r07", FINAL / "panels"
ARMS = ["10x", "TRACER whole", "TRACER partial", "TRACER whole+partial"]
SHORT = {"10x": "10x", "TRACER whole": "Whole", "TRACER partial": "Partial",
         "TRACER whole+partial": "Whole+partial",
         "TRACER whole, thinned to partial depth": "Whole, thinned"}
OUT = PAN / "SuppRCC_S7_rna_states.png"


def heat(ax, M, fmt, cmap, vmin, vmax, flags=None, cols=None, show_y=True):
    im = ax.imshow(M.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if pd.isna(v):
                continue
            rgb = im.cmap(im.norm(v))[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=4.4,
                    color="white" if lum < 0.5 else "#111111")
            if flags is not None and bool(flags.iat[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       ec="#111111", lw=0.9))
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([SHORT.get(c, c) for c in (cols or M.columns)],
                       rotation=40, ha="right", fontsize=5.0)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index if show_y else [], fontsize=5.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def main() -> int:
    F.use_style(7.0)
    U = pd.read_parquet(OBJ / "r07_umap.parquet")
    D = pd.read_csv(TAB / "r07_dotplot.csv")
    C = pd.read_csv(TAB / "r07_arm_counts.csv")
    FL = pd.read_csv(TAB / "r07_flags.csv")
    K = pd.read_csv(TAB / "r07_knn_coherence.csv")
    ref_small = set(FL[FL.flag_reference_small].cell_type)

    fig = plt.figure(figsize=(7.2, 8.9))
    gA = GridSpec(1, 4, figure=fig, left=0.10, right=0.985, top=0.985,
                  bottom=0.80, wspace=0.06)
    gB = GridSpec(1, 5, figure=fig, left=0.10, right=0.985, top=0.715,
                  bottom=0.335, wspace=0.10, width_ratios=[1, 1, 1, 1, 0.34])
    gC = GridSpec(1, 4, figure=fig, left=0.10, right=0.985, top=0.262,
                  bottom=0.075, wspace=0.12)

    # ---- a: shared-embedding UMAPs ---------------------------------------
    lim_x = np.percentile(U.umap1, [0.2, 99.8]); lim_y = np.percentile(U.umap2, [0.2, 99.8])
    pad = 0.04 * np.array([np.ptp(lim_x), np.ptp(lim_y)])
    for i, arm in enumerate(ARMS):
        ax = fig.add_subplot(gA[0, i])
        d = U[U.arm == arm].sample(frac=1.0, random_state=0)
        ax.scatter(d.umap1, d.umap2, s=0.25, lw=0, rasterized=True,
                   c=d.label.map(F.CT_COLORS).fillna("#BBBBBB"))
        ax.set_xlim(lim_x[0] - pad[0], lim_x[1] + pad[0])
        ax.set_ylim(lim_y[0] - pad[1], lim_y[1] + pad[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("auto")
        for sp in ax.spines.values():
            sp.set_color("#BBBBBB"); sp.set_linewidth(0.5)
        n = int(C[(C.arm == arm) & (C.cell_type == "ALL")].n.iloc[0])
        ax.set_xlabel(f"{SHORT[arm]}  (n = {n:,})", fontsize=6.0, labelpad=2)
        if i == 0:
            ax.set_ylabel("Shared UMAP", fontsize=6.0)
            F.panel(ax, "a", dx=-0.16, dy=1.0)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2,
                               color=F.CT_COLORS[t],
                               label=t + ("†" if t in ref_small else ""))
                        for t in TYPES],
               loc="upper center", bbox_to_anchor=(0.54, 0.785), ncol=8,
               fontsize=5.4, frameon=False, columnspacing=0.9,
               handletextpad=0.2)

    # ---- b: canonical-marker dot plot ------------------------------------
    genes = [g for t in TYPES for g in CANONICAL[t]]
    vmax = float(np.nanpercentile(D.mean_expr, 99))
    cmap = plt.get_cmap("viridis")
    for i, arm in enumerate(ARMS):
        ax = fig.add_subplot(gB[0, i])
        d = D[D.arm == arm]
        fl = FL[FL.arm == arm].set_index("cell_type").flagged
        for j, t in enumerate(TYPES):
            q = d[d.cell_type == t].set_index("gene").reindex(genes)
            y = np.arange(len(genes))
            ax.scatter(np.full(len(genes), j), y, s=q.frac_expr.fillna(0) * 26,
                       c=q.mean_expr.fillna(0), cmap=cmap, vmin=0, vmax=vmax,
                       lw=0)
        # expected blocks
        for j, t in enumerate(TYPES):
            ax.add_patch(Rectangle((j - 0.5, 4 * j - 0.5), 1, 4, fill=False,
                                   ec="#888888", lw=0.5))
        ax.set_xlim(-0.6, len(TYPES) - 0.4); ax.set_ylim(len(genes) - 0.5, -0.5)
        ax.set_xticks(range(len(TYPES)))
        ax.set_xticklabels([t + ("†" if bool(fl.get(t, False)) else "")
                            for t in TYPES], rotation=90, fontsize=5.0)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes if i == 0 else [], fontsize=4.8)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(SHORT[arm], fontsize=6.0, labelpad=2)
        if i == 0:
            F.panel(ax, "b", dx=-0.42, dy=1.12)
    lg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gB[0, 4], hspace=0.5,
                                 height_ratios=[1, 1])
    axl = fig.add_subplot(lg[0, 0]); axl.axis("off")
    axl.legend(handles=[Line2D([], [], marker="o", ls="", color="#555555",
                               ms=np.sqrt(f * 26), label=f"{int(f*100)} %")
                        for f in (0.1, 0.3, 0.6, 1.0)],
               title="Expressing", title_fontsize=5.2, fontsize=5.0,
               loc="center left", frameon=False, labelspacing=0.9)
    axc = fig.add_subplot(lg[1, 0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=axc.inset_axes([0.05, 0.0, 0.18, 0.9]))
    axc.axis("off")
    cb.set_label("Mean log CP10K", fontsize=5.2); cb.ax.tick_params(labelsize=4.8)

    # ---- c: per type x arm support ----------------------------------------
    Ct = C[C.cell_type != "ALL"]
    piv = lambda col, src=Ct: (src.pivot(index="cell_type", columns="arm",
                                          values=col).reindex(index=TYPES,
                                                              columns=ARMS))
    flg = lambda col: (FL.pivot(index="cell_type", columns="arm", values=col)
                       .reindex(index=TYPES, columns=ARMS).fillna(False))
    ax = fig.add_subplot(gC[0, 0])
    heat(ax, np.log10(piv("n").clip(lower=1)), lambda v: f"{10**v:,.0f}"
         if 10**v < 1000 else f"{10**v/1000:.0f}k", "Greys", 1, 6,
         flags=flg("flag_n_lt_200"))
    F.panel(ax, "c", dx=-0.62, dy=1.0)
    ax = fig.add_subplot(gC[0, 1])
    heat(ax, piv("low_conf_frac"), lambda v: f"{v:.2f}", "Reds", 0, 0.5,
         flags=flg("flag_low_conf_gt_0.25"), show_y=False)
    ax = fig.add_subplot(gC[0, 2])
    heat(ax, piv("marker_supported_strict"), lambda v: f"{v:.2f}", "Blues",
         0, 1, flags=flg("flag_marker_support_lt_0.6"), show_y=False)
    ax = fig.add_subplot(gC[0, 3])
    kcols = ARMS + ["TRACER whole, thinned to partial depth"]
    KM = (K.pivot(index="cell_type", columns="arm", values="knn_same_label")
          .reindex(index=TYPES, columns=kcols))
    heat(ax, KM, lambda v: f"{v:.2f}", "Purples", 0, 1, show_y=False,
         cols=kcols)
    # short, direct labels under each heat map
    for k, lab in enumerate(["Entities", "Low confidence (<0.4)",
                             "Marker-supported", "kNN same label (k=15)"]):
        fig.axes[-4 + k].set_xlabel(lab, fontsize=5.8, labelpad=2)
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    F.save(fig, OUT)                      # PNG at 400 dpi + PDF
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
