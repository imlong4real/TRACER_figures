"""SegBench v2 main figure: 6 platform columns x 4 rows, plus per-panel exports.

    A  spatial landscape (RCTD dominant type)      one column per platform
    B  ridgeplots of per-cell RCTD entropy         fixed method slots
    C  ridgeplots of per-cell RCTD max weight      fixed method slots
    D  cell number by RCTD call                    stacked horizontal bars

``build()`` is reused by the supplement for the q25/q50 ROIs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
import figlib as fl
from figlib import fs
from labels import CT_COLOR
from settings import MAIN_ORDER, PANELS_MAIN, SETTINGS

W = 7.2                      # 183 mm, Nature double column
LEFT, KEY_W, GAP_C = 0.10, 0.66, 0.09
# A column with its own comparison set (Visium HD) gets its own labelled key,
# in an extra gap on its left, so no row reads against the wrong method name.
KEY2_W = 0.60
# ln(10) = 2.30 is the ceiling for the widest reference (10 types)
ENT_XLIM, MW_XLIM = (0.0, 2.3), (0.0, 1.0)
OVERLAP = 1.55


def _fmt_um(a, b):
    return f"{a:,.0f}–{b:,.0f}"


def build(columns, out_stem, rows_h=None, letters="ABCD"):
    fs.use_style(6.5)
    ncol = len(columns)
    own_key = [fl.methods_for(s) != fl.METHODS for s in columns]
    col_w = (W - LEFT - KEY_W - 0.05 - GAP_C * (ncol - 1) - KEY2_W * sum(own_key)) / ncol
    # vertical budget (inches, top -> bottom)
    H = dict(head=0.42, map=col_w, strip=0.30, ctleg=0.58, g1=0.04,
             ridge=1.62, gx=0.30, bars=1.30, gleg=0.30, callleg=0.14)
    total = (H["head"] + H["map"] + H["strip"] + H["ctleg"] + H["g1"]
             + 2 * (H["ridge"] + H["gx"]) + H["bars"] + H["gleg"] + H["callleg"] + 0.05)
    fig = plt.figure(figsize=(W, total))

    def ax_at(x, top, w, h, **kw):
        return fig.add_axes([x / W, 1 - (top + h) / total, w / W, h / total], **kw)

    xs, x = [], LEFT + KEY_W
    for i, ok in enumerate(own_key):
        x += KEY2_W if ok else 0.0
        xs.append(x)
        x += col_w + GAP_C
    t_map = H["head"]
    t_strip = t_map + H["map"]
    t_leg = t_strip + H["strip"]
    t_r1 = t_leg + H["ctleg"] + H["g1"]
    t_r2 = t_r1 + H["ridge"] + H["gx"]
    t_bar = t_r2 + H["ridge"] + H["gx"]
    t_cl = t_bar + H["bars"] + H["gleg"]

    chosen = {}
    for x, s in zip(xs, columns):
        ds, plat, roi, _, _ = SETTINGS[s]
        p, t = fl.HEADER[ds]
        fig.text((x + col_w / 2) / W, 1 - 0.035 / total, p, ha="center", va="top",
                 fontsize=7, fontweight="bold")
        fig.text((x + col_w / 2) / W, 1 - 0.155 / total, t, ha="center", va="top", fontsize=6)
        fig.text((x + col_w / 2) / W, 1 - 0.255 / total, fl.ROI_LABEL[roi], ha="center",
                 va="top", fontsize=5.3, color="#666666")
        # A: landscape
        axm = ax_at(x, t_map, col_w, H["map"])
        m, types = fl.draw_spatial(axm, s, method="TRACER")
        chosen[s] = m
        # strip: locator + coordinates / segmentation used
        box = fl.roi_box(s)
        if box:
            loc = ax_at(x, t_strip + 0.03, 0.30, 0.24)
            fl.draw_locator(loc, s)
            x0, x1, y0, y1 = box
            txt = f"x {_fmt_um(x0, x1)} µm\ny {_fmt_um(y0, y1)} µm"
            fig.text((x + 0.34) / W, 1 - (t_strip + 0.03) / total, txt, ha="left",
                     va="top", fontsize=4.9, linespacing=1.15)
        else:
            pass                                   # whole tissue: nothing to annotate
        # cell-type legend
        axl = ax_at(x, t_leg, col_w, H["ctleg"]); axl.axis("off")
        hs = [Line2D([], [], marker="o", ls="", ms=3.2, mec="none", mfc=CT_COLOR[c], label=c)
              for c in types]
        axl.legend(handles=hs, loc="upper left", ncol=2, fontsize=4.9, handletextpad=0.15,
                   columnspacing=0.5, labelspacing=0.25, borderaxespad=0.0,
                   bbox_to_anchor=(-0.02, 1.0))
        # B, C: ridges
        a1 = ax_at(x, t_r1, col_w, H["ridge"])
        ms, NS = fl.methods_for(s), len(fl.METHODS)
        fl.draw_ridges(a1, s, "entropy", ENT_XLIM, (0.0, None), overlap=OVERLAP,
                       methods=ms, nslots=NS)
        a1.set_xlabel("RCTD entropy", fontsize=5.8, labelpad=1.5)
        a1.set_xticks([0, 1, 2]); a1.set_xticklabels(["0", "1", "2"])
        a2 = ax_at(x, t_r2, col_w, H["ridge"])
        fl.draw_ridges(a2, s, "max_weight", MW_XLIM, (0.0, 1.0), overlap=OVERLAP,
                       methods=ms, nslots=NS)
        a2.set_xlabel("RCTD max weight", fontsize=5.8, labelpad=1.5)
        a2.set_xticks([0, 0.5, 1.0]); a2.set_xticklabels(["0", "0.5", "1"])
        # D: bars
        tot = fl.cells(s).groupby("method").size().max()
        scale = 1e3
        a3 = ax_at(x, t_bar, col_w, H["bars"])
        fl.draw_bars(a3, s, scale=scale, methods=ms, nslots=NS)
        a3.set_xlabel("Cells (×10³)", fontsize=5.8, labelpad=1.5)
        for a in (a1, a2, a3):
            a.tick_params(axis="x", labelsize=5.3, pad=1.5)
        if ms != fl.METHODS:                      # this column's own key
            for top, h, bars in ((t_r1, H["ridge"], False), (t_r2, H["ridge"], False),
                                 (t_bar, H["bars"], True)):
                k = ax_at(x - KEY2_W + 0.02, top, KEY2_W - 0.06, h)
                fl.method_key(k, overlap=OVERLAP, bars=bars, methods=ms, nslots=NS,
                              fontsize=5.4)

    # shared method key (left column), one per quantitative row
    for top, h, bars in ((t_r1, H["ridge"], False), (t_r2, H["ridge"], False),
                         (t_bar, H["bars"], True)):
        k = ax_at(LEFT, top, KEY_W - 0.04, h)
        fl.method_key(k, overlap=OVERLAP, bars=bars)
    # Row A is TRACER output on every platform, not a method comparison
    fig.text((LEFT + KEY_W - 0.06) / W, 1 - (t_map + H["map"] / 2) / total,
             "TRACER\ncell-type maps\nacross platforms\n(whole + partial)", ha="right",
             va="center", fontsize=6.2, linespacing=1.25)
    # row letters
    for letter, top in zip(letters, (0.0, t_r1 - 0.06, t_r2 - 0.06, t_bar - 0.06)):
        fig.text(0.02 / W, 1 - top / total, letter, fontsize=9, fontweight="bold",
                 va="top", ha="left")
    # legends: Original-median reference line and RCTD call categories
    handles = [Patch(fc=fl.CALL_COLOR[c], ec="none", label=fl.CALL_LABEL[c]) for c in fl.CALLS]
    handles.append(Line2D([], [], color="#5A5A5A", lw=0.7, ls=(0, (2.2, 1.4)),
                          label="Original median (B, C)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=5.6,
               bbox_to_anchor=((LEFT + KEY_W + (W - LEFT - KEY_W) / 2) / W,
                               1 - t_cl / total),
               handlelength=1.3, columnspacing=1.2, frameon=False)
    out = Path(out_stem)
    fs.save(fig, out.with_suffix(".png"))
    return chosen


def export_panels(columns, outdir):
    """Each panel on its own, same drawers and sizes as the assembled figure."""
    fs.use_style(6.5)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for s in columns:
        f, a = plt.subplots(figsize=(1.6, 1.6)); fl.draw_spatial(a, s, method="TRACER")
        fs.save(f, outdir / f"A_spatial__{s}.png")
        for tag, metric, xl, bd in (("B_entropy", "entropy", ENT_XLIM, (0.0, None)),
                                    ("C_maxweight", "max_weight", MW_XLIM, (0.0, 1.0))):
            f = plt.figure(figsize=(2.3, 1.9))
            ms = fl.methods_for(s)
            k = f.add_axes([0.02, 0.14, 0.36, 0.84]); fl.method_key(k, overlap=OVERLAP, methods=ms)
            a = f.add_axes([0.40, 0.14, 0.57, 0.84])
            fl.draw_ridges(a, s, metric, xl, bd, overlap=OVERLAP, methods=ms)
            a.set_xlabel("RCTD entropy" if metric == "entropy" else "RCTD max weight")
            fs.save(f, outdir / f"{tag}__{s}.png")
        f = plt.figure(figsize=(2.3, 1.5))
        ms = fl.methods_for(s)
        k = f.add_axes([0.02, 0.18, 0.36, 0.80]); fl.method_key(k, bars=True, methods=ms)
        a = f.add_axes([0.40, 0.18, 0.57, 0.80]); fl.draw_bars(a, s, methods=ms)
        a.set_xlabel("Cells (×10³)")
        fs.save(f, outdir / f"D_calls__{s}.png")


if __name__ == "__main__":
    chosen = build(MAIN_ORDER, PANELS_MAIN / "segbench_main_v2")
    print("landscape segmentation per column:", chosen)
    if "--panels" in sys.argv:
        export_panels(MAIN_ORDER, PANELS_MAIN / "panels_v2")
