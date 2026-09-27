#!/usr/bin/env python3
"""V02 - marker-validation dot plots, heatmaps, coherence and atlas sensitivity.

Main panels carry five comparison columns, in a fixed order:

    scRNA reference (cPMI, CA001063 excluded) | pre-TRACER | post-TRACER whole
    | post-TRACER partial | post-TRACER whole+partial

Whole and partial are separate populations in the underlying data; the
whole+partial column is a DESCRIPTIVE cell-weighted pooling of those two,
derived in v01, and is labelled as such. A sixth group - the full atlas with
CA001063 restored - is used only for the atlas-sensitivity panel (S15) and never
replaces the cPMI reference.

Identical cell-type order and marker order in every panel; identical colour
limits so the z-scales are directly comparable. Dot area is the detection
fraction on a common absolute scale (S12 gives the within-group rescaled
variant); colour is expression z-scored per gene across cell types; cell counts
are printed in the y tick labels of every panel.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
sys.path.insert(0, "/scratch4/adeshpa6/tracer_campaign/final/scripts/pdac")
import pstyle as S

ROOT = Path("/scratch4/adeshpa6/tracer_campaign/final/panels/pdac_supp/marker_validation")
TAB, OUT = ROOT / "tables", ROOT.parent
A = pd.read_csv(TAB / "marker_expression_summary.tsv", sep="\t")
M = json.load(open(TAB / "marker_definition.json"))
GT = pd.read_csv(TAB / "group_totals.tsv", sep="\t").set_index("modality")
CT, GENES, MK = M["ct_order"], M["gene_order"], M["markers"]

MAIN = ["scRNA_cPMI", "pre", "post_whole", "post_partial", "post_all"]
ALL = MAIN + ["scRNA_full"]
MLAB = {"scRNA_cPMI": "scRNA reference\n(cPMI: CA001063 excl.)",
        "scRNA_full": "scRNA full atlas\n(CA001063 restored)",
        "pre": "pre-TRACER (10X)", "post_whole": "post-TRACER whole",
        "post_partial": "post-TRACER partial",
        "post_all": "post-TRACER whole+partial\n(descriptive pooling)"}
SHORT = {"Ductal cell type 1": "Ductal 1", "Ductal cell type 2": "Ductal 2",
         "Acinar cell": "Acinar", "Endocrine cell": "Endocrine", "T cell": "T",
         "B cell": "B", "Macrophage cell": "Macrophage",
         "Endothelial cell": "Endothelial", "Fibroblast cell": "Fibroblast",
         "Stellate cell": "Stellate"}
CMAP, VMIN, VMAX = "RdBu_r", -2.0, 2.0
SMIN, SMAX = 1.0, 46.0
WEAK_SPEC, WEAK_DET = 1.5, 0.05


def kfmt(n):
    n = float(n)
    return f"{n/1e6:.1f}M" if n >= 1e6 else (f"{n/1e3:.0f}k" if n >= 1e4 else f"{n:,.0f}")


def mats(mod):
    d = A[A.modality == mod]
    e = d.pivot(index="gene", columns="cell_type", values="expr").reindex(GENES)[CT]
    t = d.pivot(index="gene", columns="cell_type", values="detect").reindex(GENES)[CT]
    z = e.sub(e.mean(1), axis=0).div(e.std(1).replace(0, np.nan), axis=0)
    return z.fillna(0), t.fillna(0)


Z = {m: mats(m)[0] for m in ALL}
D = {m: mats(m)[1] for m in ALL}
NC = {m: A[A.modality == m].groupby("cell_type").n_cells.first().reindex(CT) for m in ALL}

# ------------------------------------------------------ coherence table ------
rows = []
for m in ALL:
    z, dt = Z[m], D[m]
    for t in CT:
        gs = MK[t]
        own = float(z.loc[gs, t].mean())
        oth = {o: float(z.loc[gs, o].mean()) for o in CT if o != t}
        conf = max(oth, key=oth.get)
        rows.append(dict(modality=m, cell_type=t, n_markers=len(gs), own_z=own,
                         best_other_z=oth[conf], specificity=own - oth[conf],
                         most_confusable=conf,
                         mean_detection=float(dt.loc[gs, t].mean()),
                         n_cells=int(NC[m][t])))
C = pd.DataFrame(rows)
C["weak_specificity"] = C.specificity < WEAK_SPEC
C["weak_detection"] = C.mean_detection < WEAK_DET
C["flag"] = np.where(C.weak_specificity & C.weak_detection, "ambiguous + sparse",
             np.where(C.weak_specificity, "ambiguous",
             np.where(C.weak_detection, "sparse", "ok")))
C.to_csv(TAB / "label_coherence.tsv", sep="\t", index=False)


def glab(m):
    r = GT.loc[m]
    return f"{MLAB[m]}\nn={int(r.n_cells):,} · median {r.median_n_tx:,.0f} tx"


def block_edges():
    e, c = [], 0
    for t in CT:
        c += len(MK[t]); e.append(c)
    return e[:-1]


def gene_axis(ax, show):
    ax.set_xlim(-0.6, len(GENES) - 0.4)
    ax.set_xticks(range(len(GENES)))
    ax.set_xticklabels(GENES if show else [], rotation=90, fontsize=5.0)
    for e in block_edges():
        ax.axvline(e - 0.5, color="#BBBBBB", lw=0.5)


def ct_axis(ax, mod):
    ax.set_ylim(-0.6, len(CT) - 0.4)
    ax.set_yticks(range(len(CT)))
    labs, cols = [], []
    for t in CT:
        r = C[(C.modality == mod) & (C.cell_type == t)].iloc[0]
        labs.append(f"{SHORT[t]}{' *' if r.flag != 'ok' else ''}  {kfmt(r.n_cells)}")
        cols.append("#B00020" if r.flag != "ok" else "#000000")
    ax.set_yticklabels(labs, fontsize=5.9)
    for tk, c in zip(ax.get_yticklabels(), cols):
        tk.set_color(c)
    ax.invert_yaxis()


def draw_dots(ax, mod, rescale=False):
    z, dt = Z[mod], D[mod]
    v = dt.to_numpy().astype(float)
    if rescale and v.max() > 0:
        v = v / v.max()
    xs, ys, ss, cs = [], [], [], []
    for gi in range(len(GENES)):
        for ti in range(len(CT)):
            xs.append(gi); ys.append(ti)
            ss.append(SMIN + (SMAX - SMIN) * np.sqrt(np.clip(v[gi, ti], 0, 1)))
            cs.append(z.iloc[gi, ti])
    ax.scatter(xs, ys, s=ss, c=cs, cmap=CMAP, vmin=VMIN, vmax=VMAX,
               linewidths=0.25, edgecolors="#555555")


def marker_bar(ax):
    ax.set_xlim(-0.6, len(GENES) - 0.4); ax.set_ylim(0, 1)
    c0 = 0
    for t in CT:
        n = len(MK[t])
        ax.add_patch(plt.Rectangle((c0 - 0.5, 0.15), n, 0.7,
                                   color=S.CT.get(t, "#999999"), lw=0))
        ax.text(c0 - 0.5 + n / 2, 0.5, SHORT[t], ha="center", va="center",
                fontsize=5.4, color="white", fontweight="bold")
        c0 += n
    ax.axis("off")


def legends(fig, rescale, ytop=0.58):
    cax = fig.add_axes([0.915, ytop, 0.010, 0.17])
    fig.colorbar(ScalarMappable(Normalize(VMIN, VMAX), CMAP), cax=cax)
    cax.set_ylabel("expression z-score\n(across cell types)", fontsize=6.2)
    cax.tick_params(labelsize=5.8)
    lv = [0.02, 0.1, 0.3, 0.6, 1.0] if not rescale else [0.05, 0.25, 0.5, 0.75, 1.0]
    h = [Line2D([], [], marker="o", ls="", markerfacecolor="#CCCCCC",
                markeredgecolor="#555555", markeredgewidth=0.25,
                markersize=np.sqrt(SMIN + (SMAX - SMIN) * np.sqrt(x)), label=f"{x:.0%}")
         for x in lv]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.905, ytop - 0.24),
               frameon=False, fontsize=6.0, labelspacing=0.8,
               title="detection" + ("\n(rescaled)" if rescale else ""),
               title_fontsize=6.2, handletextpad=0.6)


def build_dotplot(name, mods, rescale=False, h=9.2):
    fig = plt.figure(figsize=(11.0, h))
    gs = fig.add_gridspec(len(mods) + 1, 1, height_ratios=[0.26] + [1] * len(mods),
                          hspace=0.14)
    marker_bar(fig.add_subplot(gs[0]))
    for i, mod in enumerate(mods):
        ax = fig.add_subplot(gs[i + 1])
        draw_dots(ax, mod, rescale)
        gene_axis(ax, show=(i == len(mods) - 1))
        ct_axis(ax, mod)
        ax.set_ylabel(glab(mod), fontsize=6.3)
        ax.tick_params(length=1.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    legends(fig, rescale)
    for ext, dpi in (("pdf", 600), ("png", 450)):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig); print(f"  wrote {name}", flush=True)


build_dotplot("S11_marker_validation_dotplot", MAIN)
build_dotplot("S12_marker_validation_dotplot_rescaled", MAIN, rescale=True)

# ------------------------------------------------------------- heatmap -------
fig, axes = plt.subplots(len(MAIN), 1, figsize=(11.0, 7.4), sharex=True)
for ax, mod in zip(axes, MAIN):
    im = ax.imshow(Z[mod].to_numpy().T, aspect="auto", cmap=CMAP,
                   vmin=VMIN, vmax=VMAX, interpolation="nearest")
    ct_axis(ax, mod); ax.set_ylim(len(CT) - 0.5, -0.5)
    gene_axis(ax, show=(mod == MAIN[-1]))
    ax.set_ylabel(glab(mod), fontsize=6.3); ax.tick_params(length=1.6)
cax = fig.add_axes([0.92, 0.40, 0.010, 0.20])
fig.colorbar(im, cax=cax); cax.set_ylabel("expression z-score", fontsize=6.2)
cax.tick_params(labelsize=5.8)
for ext, dpi in (("pdf", 600), ("png", 450)):
    fig.savefig(OUT / f"S13_marker_validation_heatmap.{ext}", dpi=dpi, bbox_inches="tight")
plt.close(fig); print("  wrote S13_marker_validation_heatmap", flush=True)

# -------------------------------------------------- coherence summary --------
MC = {"scRNA_cPMI": "#111111", "pre": "#9E9E9E", "post_whole": "#0072B2",
      "post_partial": "#D55E00", "post_all": "#CC79A7"}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7))
x = np.arange(len(CT)); w = 0.16
for ax, (col, lab, logy, thr) in zip(
        axes, [("specificity", "Marker specificity\n(own − best other, z)", False, WEAK_SPEC),
               ("mean_detection", "Mean detection of\nown markers", True, WEAK_DET)]):
    for i, m in enumerate(MAIN):
        v = C[C.modality == m].set_index("cell_type").reindex(CT)[col]
        ax.bar(x + (i - 2) * w, v, w, color=MC[m],
               label=MLAB[m].split("\n")[0] if col == "specificity" else None)
    ax.axhline(thr, color="#B00020", lw=0.9, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in CT], rotation=45,
                                         ha="right", fontsize=6.2)
    ax.set_ylabel(lab)
    if logy:
        ax.set_yscale("log")
axes[0].legend(ncol=3, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, 1.30),
               handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
fig.subplots_adjust(wspace=0.42)
for ext, dpi in (("pdf", 600), ("png", 450)):
    fig.savefig(OUT / f"S14_marker_coherence_summary.{ext}", dpi=dpi, bbox_inches="tight")
plt.close(fig); print("  wrote S14_marker_coherence_summary", flush=True)

# ------------------------------------------- S15 atlas sensitivity -----------
COMP = pd.read_csv(TAB / "atlas_composition.tsv", sep="\t").set_index("cell_type").reindex(CT)
build_dotplot("S15a_atlas_sensitivity_dotplot", ["scRNA_cPMI", "scRNA_full"], h=4.4)

fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.7))
ax = axes[0]
for i, m in enumerate(["scRNA_cPMI", "scRNA_full"]):
    v = C[C.modality == m].set_index("cell_type").reindex(CT).specificity
    ax.bar(x + (i - 0.5) * 0.36, v, 0.36,
           color="#111111" if m == "scRNA_cPMI" else "#E69F00",
           label=MLAB[m].split("\n")[0])
ax.axhline(WEAK_SPEC, color="#B00020", lw=0.9, ls=(0, (3, 2)))
ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in CT], rotation=45,
                                     ha="right", fontsize=6.2)
ax.set_ylabel("Marker specificity\n(own − best other, z)")
ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, 1.22),
          ncol=2, handlelength=1.0, handletextpad=0.4)
ax = axes[1]
dspec = (C[C.modality == "scRNA_full"].set_index("cell_type").specificity
         - C[C.modality == "scRNA_cPMI"].set_index("cell_type").specificity).reindex(CT)
ax.barh(np.arange(len(CT))[::-1], dspec,
        color=["#009E73" if v > 0 else "#D55E00" for v in dspec])
ax.axvline(0, color="#666666", lw=0.8)
ax.set_yticks(np.arange(len(CT))[::-1]); ax.set_yticklabels([SHORT[t] for t in CT],
                                                            fontsize=6.2)
ax.set_xlabel("Δ specificity\n(full atlas − cPMI reference)")
ax = axes[2]
ax.barh(np.arange(len(CT))[::-1], COMP.frac_from_CA001063 * 100, color="#56B4E9")
ax.set_yticks(np.arange(len(CT))[::-1]); ax.set_yticklabels([SHORT[t] for t in CT],
                                                            fontsize=6.2)
ax.set_xlabel("% of atlas cells of this type\ncontributed by CA001063")
fig.subplots_adjust(wspace=0.55)
for ext, dpi in (("pdf", 600), ("png", 450)):
    fig.savefig(OUT / f"S15b_atlas_sensitivity_summary.{ext}", dpi=dpi, bbox_inches="tight")
plt.close(fig); print("  wrote S15b_atlas_sensitivity_summary", flush=True)

pd.set_option("display.width", 250)
print("\n=== coherence ===")
print(C[["modality", "cell_type", "own_z", "specificity", "most_confusable",
         "mean_detection", "n_cells", "flag"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
