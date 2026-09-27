#!/usr/bin/env python
"""r06 - RCC supplementary panels.

S1  estimand audit: the additive pedestal, the previous threshold and its
    uncontrolled cell-free false-positive rate, the fixed-FPR replacement, and
    the dynamic range each estimand actually delivers
S2  all 27 markers plus the 4 autofluorescence channels: positivity and
    background-subtracted intensity, whole vs partial vs registration null
S3  specificity matrices - whole cells, and partial cells scored only on
    spillover-free pixels
S4  ROI selection audit: every candidate window against the prespecified rule,
    with the selected windows marked
S5  the two runner-up ROIs, same layout as the main figure
S6  footprint support: distance to the nearest 10x cell, footprint size, and how
    the matched contrast depends on both
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from rcc_defs import CANON, TYPES, MARKERS, AF, ALLCH, DISPLAY

FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ, TAB, PAN = FINAL / "objects", FINAL / "tables", FINAL / "panels"
OLDBG = Path("/scratch4/adeshpa6/xenium_rcc/pr8/figdata/marker_background.json")
OLDPUN = Path("/scratch4/adeshpa6/xenium_rcc/rcc_analysis/validation/"
              "puncta_fraction.csv")
SPILL_UM = 2.0


def s1(thr, DIST):
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.75, wspace=0.30,
                  left=0.085, right=0.985, top=0.95, bottom=0.16)
    old = json.loads(OLDBG.read_text()) if OLDBG.exists() else {}
    mk = [m for m in MARKERS if m in thr["markers"]]
    x = np.arange(len(mk))

    ax = fig.add_subplot(gs[0, 0])
    ped = [thr["markers"][m]["pedestal"] for m in mk]
    p99 = [DIST[(DIST.marker == m) & (DIST.entity_class == "whole")]
           .mean_int.iloc[0] if len(DIST[(DIST.marker == m)]) else np.nan
           for m in mk]
    ax.bar(x, ped, 0.8, color="#B0B0B0", lw=0, label="Pedestal (counts)")
    ax.bar(x, [thr["markers"][m]["thr_used"] for m in mk], 0.8, bottom=ped,
           color=F.C_PARTIAL, lw=0, label="Threshold above pedestal")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90,
                                         fontsize=4.8)
    ax.set_ylabel("Intensity (counts)")
    ax.legend(fontsize=5.6, loc="upper left")
    F.panel(ax, "a", dx=-0.13)

    ax = fig.add_subplot(gs[0, 1])
    new_fpr = [thr["markers"][m]["cellfree_frac_pos"] for m in mk]
    ax.plot(x, new_fpr, "o-", color=F.C_PARTIAL, ms=2.4, lw=0.8,
            label="Fixed-quantile (this work)")
    if OLDPUN.exists():
        op = pd.read_csv(OLDPUN).set_index("marker").neg_punctafrac
        ov = [op.get(m, np.nan) for m in mk]
        ax.plot(x, ov, "s", color=F.C_NULL, ms=2.8,
                label="Previous  bg + 2·MAD")
    ax.axhline(0.01, color="#444444", ls="--", lw=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90,
                                         fontsize=4.8)
    ax.set_ylabel("Cell-free false-positive rate")
    ax.legend(fontsize=5.6, loc="lower left")
    F.panel(ax, "b", dx=-0.13)

    ax = fig.add_subplot(gs[1, 0])
    w = DIST[DIST.entity_class == "whole"].set_index("marker").reindex(mk)
    p = DIST[DIST.entity_class == "partial"].set_index("marker").reindex(mk)
    ax.plot(x, 1 + w.mean_int / np.array(ped), "o-", color=F.C_WHOLE, ms=2.2,
            lw=0.7, label="Whole")
    ax.plot(x, 1 + p.mean_int / np.array(ped), "o-", color=F.C_PARTIAL, ms=2.2,
            lw=0.7, label="Partial")
    ax.axhline(1.0, color="#444444", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90,
                                         fontsize=4.8)
    ax.set_ylabel("Previous estimand\nmean / pedestal")
    ax.legend(fontsize=5.6)
    F.panel(ax, "c", dx=-0.13)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x, w.mean_int, "o-", color=F.C_WHOLE, ms=2.2, lw=0.7, label="Whole")
    ax.plot(x, p.mean_int, "o-", color=F.C_PARTIAL, ms=2.2, lw=0.7,
            label="Partial")
    ax.axhline(0, color="#444444", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90,
                                         fontsize=4.8)
    ax.set_ylabel("This work\nintensity − pedestal (counts)")
    ax.legend(fontsize=5.6)
    F.panel(ax, "d", dx=-0.13)
    F.save(fig, PAN / "SuppRCC_S1_estimand_audit.png")


def s2(DIST):
    mk = [m for m in MARKERS] + AF
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2),
                             gridspec_kw=dict(hspace=0.62))
    x = np.arange(len(mk))
    w = DIST[DIST.entity_class == "whole"].set_index("marker").reindex(mk)
    p = DIST[DIST.entity_class == "partial"].set_index("marker").reindex(mk)
    ax = axes[0]
    ax.bar(x - 0.22, w.mean_pos, 0.42, color=F.C_WHOLE, lw=0, label="Whole")
    ax.bar(x + 0.22, p.mean_pos, 0.42, color=F.C_PARTIAL, lw=0, label="Partial")
    ax.plot(x, p.null_pos, "_", color="#111111", ms=7, mew=1.0,
            label="Registration null")
    for i in range(len(mk)):
        if np.isfinite(p.ci_lo.iloc[i]):
            ax.plot([x[i] + 0.22] * 2, [p.ci_lo.iloc[i], p.ci_hi.iloc[i]], "-",
                    color="#333333", lw=0.6)
    ax.set_ylabel("Fraction of pixels positive")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90,
                                         fontsize=5.0)
    ax.axvline(len(MARKERS) - 0.5, color="#111111", lw=0.8,
               ymin=0.0, ymax=0.92)
    ax.legend(fontsize=5.8, ncol=3, loc="upper left")
    F.panel(ax, "a", dx=-0.075)
    ax = axes[1]
    ax.bar(x - 0.22, w.mean_int, 0.42, color=F.C_WHOLE, lw=0)
    ax.bar(x + 0.22, p.mean_int, 0.42, color=F.C_PARTIAL, lw=0)
    ax.set_ylabel("Intensity − pedestal (counts)")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90,
                                         fontsize=5.0)
    ax.axvline(len(MARKERS) - 0.5, color="#111111", lw=0.8, ymin=0.0, ymax=0.95)
    F.panel(ax, "b", dx=-0.075)
    F.save(fig, PAN / "SuppRCC_S2_all_markers.png")


def _matrix(ax, SP, cls, col, qcol, title_markers):
    ct = [t for t in TYPES if len(SP[(SP.entity_class == cls)
                                     & (SP.cell_type == t)])]
    M = np.full((len(ct), len(title_markers)), np.nan)
    Q = np.full_like(M, np.nan)
    s = SP[SP.entity_class == cls]
    for i, t in enumerate(ct):
        for j, m in enumerate(title_markers):
            r = s[(s.cell_type == t) & (s.marker == m)]
            if len(r):
                M[i, j] = r[col].iloc[0]; Q[i, j] = r[qcol].iloc[0]
    vm = float(np.nanpercentile(np.abs(M), 98)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto",
                   interpolation="nearest")
    for i, t in enumerate(ct):
        j = title_markers.index(CANON[t])
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               ec="#111111", lw=1.0))
    for i in range(len(ct)):
        for j in range(len(title_markers)):
            if F.stars(Q[i, j]) != "ns" and abs(M[i, j]) > 0.6:
                ax.text(j, i, "•", ha="center", va="center", fontsize=5.2)
    ax.set_xticks(range(len(title_markers)))
    ax.set_xticklabels([DISPLAY[m] for m in title_markers], rotation=90,
                       fontsize=5.0)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct, fontsize=5.8)
    ax.axvline(len(MARKERS) - 0.5, color="#111111", lw=0.8)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(length=0)
    return im


def s3(SP):
    mk = [m for m in MARKERS] + AF
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0),
                             gridspec_kw=dict(hspace=0.62))
    im = _matrix(axes[0], SP, "whole", "rel_log2", "q_bh", mk)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.018, pad=0.006)
    cb.set_label("Specificity, whole cells\n(row-centred log₂)", fontsize=5.4)
    cb.ax.tick_params(labelsize=5)
    F.panel(axes[0], "a", dx=-0.075)
    im = _matrix(axes[1], SP, "partial", "rel_log2_far", "q_bh", mk)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.018, pad=0.006)
    cb.set_label("Specificity, partial cells\n>2 µm only", fontsize=5.4)
    cb.ax.tick_params(labelsize=5)
    F.panel(axes[1], "b", dx=-0.075)
    F.save(fig, PAN / "SuppRCC_S3_specificity_whole_and_spillfree.png")


def s4():
    C = pd.read_csv(TAB / "r04_roi_candidates.csv")
    sel = json.loads((TAB / "r04_roi_selected.json").read_text())
    names = {(round(m["x0"], 1), round(m["y0"], 1)): m["name"]
             for m in sel["selected"]}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1),
                            gridspec_kw=dict(wspace=0.55))
    ax = axes[0]
    sc = ax.scatter(C.residual_frac, C.n_supported, c=C.capture_ratio,
                    cmap="viridis", s=12, lw=0)
    for _, r in C.iterrows():
        nm = names.get((round(r.x0, 1), round(r.y0, 1)))
        if nm:
            ax.plot(r.residual_frac, r.n_supported, "o", mfc="none",
                    mec="#D55E00", ms=8, mew=1.1)
    ax.set_xlabel("Residual area fraction"); ax.set_ylabel("Supported partials")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03, shrink=0.85)
    cb.set_label("Capture ratio", fontsize=5.6); cb.ax.tick_params(labelsize=5)
    F.panel(ax, "a", dx=-0.30)
    ax = axes[1]
    ax.hist(C.capture_ratio, bins=18, color="#BBBBBB", lw=0)
    for m in sel["selected"]:
        ax.axvline(m["capture_ratio"], color="#D55E00", lw=1.0)
    ax.axvline(1.0, color="#111111", ls="--", lw=0.7)
    ax.set_xlabel("Capture ratio"); ax.set_ylabel("Windows")
    F.panel(ax, "b", dx=-0.30)
    ax = axes[2]
    ax.scatter(C.n_types, C.capture_ratio, c="#8C8C8C", s=10, lw=0)
    for m in sel["selected"]:
        ax.plot(m["n_types"], m["capture_ratio"], "o", color="#D55E00", ms=5)
    ax.axhline(1.0, color="#111111", ls="--", lw=0.7)
    ax.set_xlabel("Distinct lineages"); ax.set_ylabel("Capture ratio")
    F.panel(ax, "c", dx=-0.30)
    F.save(fig, PAN / "SuppRCC_S4_roi_selection.png")


def s5(thr):
    """The runner-up ROIs, drawn by the same block renderer as Fig 1a-e."""
    from r05_fig_main import roi_block
    from matplotlib.gridspec import GridSpec
    sel = json.loads((TAB / "r04_roi_selected.json").read_text())["selected"]
    rest = [m["name"] for m in sel][1:]
    if not rest:
        return
    n = len(rest)
    fig = plt.figure(figsize=(7.2, 4.05 * n))
    letters = "abcdefghijklmno"
    for k, nm in enumerate(rest):
        top = 0.99 - k * (0.985 / n)
        bot = top - (0.985 / n) + 0.055
        g = GridSpec(1, 1, figure=fig, left=0.075, right=0.975, top=top,
                     bottom=bot)
        roi_block(fig, g[0, 0], thr, name=nm,
                  letters=letters[5 * k:5 * k + 5], scalebar=True)

    F.save(fig, PAN / "SuppRCC_S5_additional_rois.png")


def s6():
    E = pd.read_parquet(OBJ / "rcc_entity_summary.parquet")
    E["entity_class"] = E.entity_class.astype(str)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05),
                            gridspec_kw=dict(wspace=0.52))
    ax = axes[0]
    for cls, col in (("whole", F.C_WHOLE), ("partial", F.C_PARTIAL)):
        v = E[E.entity_class == cls].dist_um_mean.dropna()
        ax.hist(v, bins=np.linspace(0, 12, 40), histtype="step", lw=1.0,
                color=col, density=True, label=cls.capitalize())
    ax.axvline(SPILL_UM, color="#111111", ls="--", lw=0.7)
    ax.set_xlabel("Distance to nearest\n10x cell (µm)"); ax.set_ylabel("Density")
    ax.legend(fontsize=5.6)
    F.panel(ax, "a", dx=-0.42)
    ax = axes[1]
    for cls, col in (("whole", F.C_WHOLE), ("partial", F.C_PARTIAL)):
        v = E[E.entity_class == cls].n_px.dropna()
        ax.hist(np.log10(v.clip(1)), bins=40, histtype="step", lw=1.0,
                color=col, density=True)
    ax.set_xlabel("log₁₀ footprint pixels"); ax.set_ylabel("Density")
    F.panel(ax, "b", dx=-0.42)
    # the MEAN contrast is used here: positivity over a small footprint is
    # granular, so a binned median collapses onto 0 and hides the trend
    P = E[E.entity_class == "partial"]
    ax = axes[2]
    edges = [0, 1, 2, 3, 4, 6, 8, 100]
    b = pd.cut(P.dist_um_mean, edges)
    g = P.groupby(b, observed=True).matched_delta.agg(["mean", "sem", "count"])
    xs = np.arange(len(g))
    ax.errorbar(xs, g["mean"], yerr=1.96 * g["sem"], fmt="o-",
                color=F.C_PARTIAL, ms=3, lw=0.9, elinewidth=0.7)
    ax.axhline(0, color="#111111", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{edges[i]}–{edges[i+1]}" if edges[i+1] < 100
                        else f">{edges[i]}" for i in range(len(g))],
                       rotation=90, fontsize=5.0)
    ax.set_xlabel("Distance to nearest 10x cell (µm)")
    ax.set_ylabel("Matched − mismatched")
    F.panel(ax, "c", dx=-0.42)
    ax = axes[3]
    qs = pd.qcut(P.n_px, 6, duplicates="drop")
    g = P.groupby(qs, observed=True).matched_delta.agg(["mean", "sem"])
    med = P.groupby(qs, observed=True).n_px.median()
    xs = np.arange(len(g))
    ax.errorbar(xs, g["mean"], yerr=1.96 * g["sem"], fmt="o-",
                color=F.C_PARTIAL, ms=3, lw=0.9, elinewidth=0.7)
    ax.axhline(0, color="#111111", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(v)}" for v in med.values], fontsize=5.2)
    ax.set_xlabel("Footprint pixels (bin median)")
    ax.set_ylabel("Matched − mismatched")
    F.panel(ax, "d", dx=-0.42)
    F.save(fig, PAN / "SuppRCC_S6_support.png")


def main() -> int:
    F.use_style(7.0)
    PAN.mkdir(parents=True, exist_ok=True)
    thr = json.loads((OBJ / "rcc_marker_thresholds.json").read_text())
    DIST = pd.read_csv(TAB / "r03_marker_distributions.csv")
    SP = pd.read_csv(TAB / "r03b_specificity_centred.csv")
    s1(thr, DIST); s2(DIST); s3(SP)
    if (TAB / "r04_roi_candidates.csv").exists():
        s4(); s5(thr)
    s6()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
