"""Shared style + paths for the PDAC two-layer figure (Xenium introducer + VisiumHD cohort).

Arial via Arimo (metric-identical). High-contrast Okabe-Ito-derived palette,
colour-blind safe. Light canvas. No in-panel titles anywhere: every panel is
saved bare so it can be placed and captioned in the manuscript.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
from matplotlib import rcParams

for _f in Path("/home/lyuan13/scr4_adeshpa6/tracer_campaign/fonts").glob("Arimo-*.ttf"):
    _fm.fontManager.addfont(str(_f))
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arimo", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 200, "legend.frameon": False,
})

ROOT = Path("/scratch4/adeshpa6/tracer_campaign/final")
PANEL_MAIN = ROOT / "panels/pdac_main"
PANEL_SUPP = ROOT / "panels/pdac_supp"
TAB = ROOT / "tables/pdac"
PDAC6 = Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
SPAT = PDAC6 / "audit_v2/spatial"
XEN = Path("/scratch4/adeshpa6/TRACER/datasets/pancreas_cancer_xenium_10x/processed/fig2")

PTS = ["HC01", "HC03", "HC04", "HC05", "HC07", "HC08"]
RESP = {"HC05": 1, "HC07": 1, "HC08": 1, "HC01": 0, "HC03": 0, "HC04": 0}
ARMS = ["pre", "post_whole", "post_all"]
ARMLAB = {"pre": "10X", "post_whole": "whole", "post_all": "whole+partial"}
TUMOR = ["Ductal cell type 1", "Ductal cell type 2"]

# ---- high-contrast, colour-blind-safe cell-type palette (Okabe-Ito derived) ----
CT = {
    "T cell":             "#0072B2",   # blue
    "Macrophage cell":    "#D55E00",   # vermillion
    "B cell":             "#56B4E9",   # sky
    "Ductal cell type 1": "#E69F00",   # amber
    "Ductal cell type 2": "#E69F00",
    "Endothelial cell":   "#009E73",   # green
    "Fibroblast cell":    "#BFBFBF",   # grey
    "Stellate cell":      "#9E9E9E",
    "Acinar cell":        "#F0E442",   # yellow
    "Endocrine cell":     "#CC79A7",   # magenta
}
CT_SHORT = {
    "T cell": "T", "Macrophage cell": "Mac", "B cell": "B",
    "Ductal cell type 1": "Ductal 1", "Ductal cell type 2": "Ductal 2",
    "Endothelial cell": "Endo", "Fibroblast cell": "Fibro",
    "Stellate cell": "Stellate", "Acinar cell": "Acinar",
    "Endocrine cell": "Endocrine",
}
# ---- checkpoint-axis roles (kept identical in every panel) ----
VISTA_C = "#D55E00"    # VISTA+ TAM   (vermillion)
PSGL1_C = "#0072B2"    # PSGL-1+ T    (blue)
LINK_C  = "#8E44AD"    # ligand-receptor edge (purple, reads on both)
CTX_C   = "#E3E3E3"    # background tissue
NECTIN2_C = "#CC79A7"
TIGIT_C   = "#000000"

ARMC = {"pre": "#9E9E9E", "post_whole": "#0072B2", "post_all": "#D55E00"}
RC = {1: "#009E73", 0: "#D55E00"}          # responder / non-responder
ENTC = {"cell": "#0072B2", "partial": "#E69F00"}

CATS = ["visible pre", "strengthened post", "enabled post-whole",
        "enabled by partials", "unsupported"]
CATC = ["#F0E442", "#009E73", "#0072B2", "#56B4E9", "#EFEFEF"]


def save(fig, name, supp=False):
    d = PANEL_SUPP if supp else PANEL_MAIN
    d.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 450), ("svg", 600)):
        fig.savefig(d / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {'supp/' if supp else 'main/'}{name}.{{pdf,png,svg}}", flush=True)


def load_arm(P, arm, columns=None):
    """One arm's entity table. Pass `columns` to stay inside the 5 GB login cgroup -
    these parquets carry 36 columns and post_all can be >500k rows."""
    import pandas as pd
    rd = lambda f: pd.read_parquet(SPAT / f, columns=columns)
    if arm == "pre":
        return rd(f"{P}_pre_spatial.parquet")
    if arm == "post_whole":
        return rd(f"{P}_post_whole_spatial.parquet")
    return pd.concat([rd(f"{P}_post_whole_spatial.parquet"),
                      rd(f"{P}_post_partial_spatial.parquet")], ignore_index=True)
