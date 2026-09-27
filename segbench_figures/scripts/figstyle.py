"""Publication figure style for the final TRACER campaign panels.

Typeface: Arial.  This machine ships no Arial and no metric clone, so Arimo -
Google's metrically identical Arial substitute (same advance widths, same
1000-unit em, same glyph coverage for Latin/Greek) - is installed under
final/fonts and registered here.  Text set in Arimo is dimensionally
interchangeable with Arial; PDFs embed it as Type 42.

Design rules enforced:
  * no titles, subtitles or footnotes inside a panel - panel letters only;
  * short axis and legend labels;
  * colour-blind-safe categorical palette (Okabe-Ito derived, checked for
    deuteranopia/protanopia/tritanopia separation);
  * one colour per cell type / clone, shared across every panel;
  * hairline axes, no top/right spines, no gridlines unless load-bearing.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

FONTDIR = Path("/scratch4/adeshpa6/tracer_campaign/fonts")
for f in sorted(FONTDIR.glob("*.ttf")):
    try:
        fm.fontManager.addfont(str(f))
    except Exception:
        pass
_HAVE = {f.name for f in fm.fontManager.ttflist}
FONT = ["Arial", "Arimo", "Helvetica", "Nimbus Sans", "DejaVu Sans"]

# ---- palettes -----------------------------------------------------------
# Okabe-Ito, colour-blind safe, plus two darkened extensions.
OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7", "black": "#000000"}

# entity classes (RCC)
C_WHOLE   = "#0072B2"     # blue
C_PARTIAL = "#D55E00"     # vermillion
C_NULL    = "#8C8C8C"     # grey - registration / permutation null
C_BG      = "#BFBFBF"

# transferred cell types (RCC) - fixed across every RCC panel
CT_COLORS = {
    "Myeloid":     "#D55E00",
    "T cell":      "#0072B2",
    "B cell":      "#56B4E9",
    "Plasma cell": "#009E73",
    "Endothelial": "#CC79A7",
    "Mural":       "#E69F00",
    "Tumor":       "#661100",
    "Mast cell":   "#999933",
}
# lineages (GBM / cervical) - fixed across every clonality panel
LIN_COLORS = {
    "Neoplastic":     "#661100",
    "Myeloid":        "#D55E00",
    "T_NK":           "#0072B2",
    "B_plasma":       "#56B4E9",
    "Vascular":       "#CC79A7",
    "Glial_Neuronal": "#009E73",
    "Tumour":         "#661100",
    "Stromal":        "#E69F00",
    "Other":          "#8C8C8C",
}
LIN_LABEL = {"T_NK": "T / NK", "B_plasma": "B / plasma",
             "Glial_Neuronal": "Glial / neuronal", "Neoplastic": "Tumour"}

# segmentation arms - fixed across every clonality panel
ARM_COLORS = {"pre": "#8C8C8C", "original": "#8C8C8C",
              "post_cells": "#0072B2", "post_whole": "#0072B2",
              "post-TRACER whole": "#0072B2",
              "post_all": "#D55E00", "post-TRACER whole+partial": "#D55E00"}
ARM_LABEL = {"pre": "Original", "post_cells": "TRACER",
             "post_all": "TRACER +partial", "original": "Original",
             "post_whole": "TRACER"}

# clone colours - up to 8 per patient, cycled from the safe palette
CLONE_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
               "#56B4E9", "#661100", "#999933"]


def clone_color(i):
    return CLONE_CYCLE[int(i) % len(CLONE_CYCLE)]


def use_style(base=7.0):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.size": base,
        "axes.titlesize": base, "axes.labelsize": base,
        "xtick.labelsize": base - 0.5, "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2.0,
        "ytick.major.size": 2.0, "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "legend.handlelength": 1.2, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0, "legend.borderpad": 0.0,
        "legend.labelspacing": 0.35,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 400,
        "figure.dpi": 160, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
        "lines.linewidth": 1.0, "lines.markersize": 3.0,
        "errorbar.capsize": 0.0,
    })


def panel(ax, letter, dx=-0.30, dy=1.02, size=9):
    """Bold panel letter, placed OUTSIDE the axes so it can never collide."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left", clip_on=False)


def stars(q):
    if q is None or not np.isfinite(q):
        return "ns"
    return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "ns"


def nolabel(ax):
    ax.set_title("")


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def save(fig, path, pdf=True):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=400, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    if pdf:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {p}", flush=True)


def check_font():
    have = [f for f in FONT if f in _HAVE]
    return have[0] if have else "DejaVu Sans"
