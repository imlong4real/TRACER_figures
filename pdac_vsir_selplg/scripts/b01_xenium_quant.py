#!/usr/bin/env python3
"""Xenium introducer - quantitative panels.
M2 transcript-fate accounting | M5 admixture cleanup | M6 VSIG4 attribution
M7 immune inventory recovery  | S1 VSIG4->T proximity density audit (negative)
No in-panel titles: every panel is bare for manuscript placement.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

FA = pd.read_csv(S.TAB / "xenium_transcript_fate.tsv", sep="\t").set_index("quantity")
AT = pd.read_csv(S.TAB / "xenium_vsig4_attribution.tsv", sep="\t")
IV = pd.read_csv(S.TAB / "xenium_inventory.tsv", sep="\t")
PR = pd.read_csv(S.TAB / "xenium_vsig4_proximity_audit.tsv", sep="\t")
CN = pd.read_csv(S.XEN / "contamination_by_celltype.csv")

# ---------------------------------------------------------------- M2 fate ---
lab = ["Kept in place", "Moved to another cell", "Built into partial cells",
       "Rescued from unassigned"]
key = ["kept_in_place", "redistributed", "out_partial", "recovered_from_unassigned"]
col = ["#D6D6D6", "#D55E00", "#E69F00", "#0072B2"]
pct = [FA.loc[k, "pct_of_total"] for k in key]
n = [FA.loc[k, "n"] for k in key]
fig, ax = plt.subplots(figsize=(3.5, 1.75))
y = np.arange(len(lab))[::-1]
ax.barh(y, pct, color=col, height=0.66, edgecolor="white", linewidth=0.6)
for yi, p, c in zip(y, pct, n):
    ax.text(p + 1.2, yi, f"{p:.1f}%  ({c/1e6:.2f}M)", va="center", ha="left", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(lab)
ax.set_xlabel("Transcripts (% of 20.7 M)")
ax.set_xlim(0, 78); ax.set_ylim(-0.6, len(lab) - 0.4)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
S.save(fig, "M2_xenium_transcript_fate_accounting")

# ------------------------------------------------------- M5 admixture -------
CN = CN.sort_values("delta", ascending=True)
CN["short"] = CN.cell_type.map(S.CT_SHORT).fillna(CN.cell_type)
imm = CN.cell_type.isin(["T cell", "Macrophage cell", "B cell", "Endothelial cell"])
fig, ax = plt.subplots(figsize=(3.3, 2.5))
y = np.arange(len(CN))
for yi, (_, r) in zip(y, CN.iterrows()):
    hl = r.cell_type in ("T cell", "Macrophage cell", "B cell", "Endothelial cell")
    ax.plot([r.orig, r.complete], [yi, yi], color="#D55E00" if hl else "#C8C8C8",
            lw=2.0 if hl else 1.2, zorder=1, solid_capstyle="round")
    ax.scatter([r.orig], [yi], s=24, color="#9E9E9E", zorder=3, linewidths=0)
    ax.scatter([r.complete], [yi], s=24, color="#0072B2", zorder=3, linewidths=0,
               alpha=1.0 if hl else 0.45)
ax.set_yticks(y)
ax.set_yticklabels([f"$\\bf{{{s}}}$" if h else s
                    for s, h in zip(CN.short, imm)], fontsize=7)
ax.set_xlabel("Cross-lineage admixture (median)")
ax.set_xlim(-0.02, 0.45); ax.set_ylim(-0.7, len(CN) - 0.3)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#9E9E9E", label="10X"),
                   plt.Line2D([], [], marker="o", ls="", color="#0072B2", label="TRACER")],
          loc="lower right", handletextpad=0.2, borderpad=0.2)
S.save(fig, "M5_xenium_admixture_cleanup")

# ------------------------------------------------- M6 VSIG4 attribution -----
order = ["Macrophage cell", "Fibroblast cell", "Ductal cell type 2", "Stellate cell",
         "B cell", "T cell", "Endothelial cell", "Acinar cell",
         "Ductal cell type 1", "Endocrine cell"]
oth = "#DCDCDC"
fig, ax = plt.subplots(figsize=(3.5, 1.35))
for yi, arm in enumerate(["tracer", "original"]):
    r = AT[AT.arm == arm].iloc[0]
    left = 0.0
    for t in order:
        v = float(r[t]) * 100
        if v <= 0:
            continue
        c = S.CT["Macrophage cell"] if t == "Macrophage cell" else oth
        ax.barh(yi, v, left=left, color=c, height=0.6,
                edgecolor="white", linewidth=0.7)
        if t == "Macrophage cell":
            ax.text(left + v / 2, yi, f"{v:.0f}%", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        left += v
ax.set_yticks([0, 1]); ax.set_yticklabels(["TRACER", "10X"])
ax.set_xlabel("VSIG4$^+$ cells assigned to each type (%)")
ax.set_xlim(0, 100); ax.set_ylim(-0.55, 1.55)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(facecolor=S.CT["Macrophage cell"], label="Macrophage"),
                   Patch(facecolor=oth, label="other types")],
          loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2,
          handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
S.save(fig, "M6_xenium_vsig4_specificity")

# ------------------------------------------------- M7 immune recovery -------
keep = ["T cell", "Macrophage cell", "B cell", "Endothelial cell"]
q = IV[IV.cell_type.isin(keep)].set_index("cell_type").loc[keep].reset_index()
fig, ax = plt.subplots(figsize=(3.0, 2.0))
x = np.arange(len(q)); w = 0.36
ax.bar(x - w / 2, q.original / 1e3, w, color="#9E9E9E", label="10X")
ax.bar(x + w / 2, q.tracer_complete / 1e3, w, color=S.ENTC["cell"], label="whole")
ax.bar(x + w / 2, q.tracer_partial / 1e3, w, bottom=q.tracer_complete / 1e3,
       color=S.ENTC["partial"], label="partial")
for xi, f in zip(x, q.fold):
    tot = (q.tracer_total.iloc[xi]) / 1e3
    ax.text(xi + w / 2, tot + 0.45, f"{f:.2f}×", ha="center", fontsize=7,
            fontweight="bold", color="#333333")
ax.set_xticks(x); ax.set_xticklabels([S.CT_SHORT[c] for c in q.cell_type])
ax.set_ylabel("Confidently typed cells (×10$^3$)")
ax.set_ylim(0, max(q.tracer_total / 1e3) * 1.22)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
          handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
S.save(fig, "M7_xenium_immune_recovery")

# ------------------------------- S1 proximity density audit (negative) ------
o = PR[PR.arm == "original"].iloc[0]; t = PR[PR.arm == "tracer"].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.15))
ax = axes[0]
x = np.arange(2); w = 0.34
ax.bar(x - w / 2, [o.med_raw, t.med_raw], w, color=S.CT["Macrophage cell"],
       label="VSIG4$^+$ TAM")
ax.bar(x + w / 2, [o.med_background, t.med_background], w, color="#C8C8C8",
       label="all cells")
for xi, (a, b) in enumerate([(o.med_raw, o.med_background), (t.med_raw, t.med_background)]):
    ax.text(xi - w / 2, a + 5, f"{a:.0f}", ha="center", fontsize=7)
    ax.text(xi + w / 2, b + 5, f"{b:.0f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(["10X", "TRACER"])
ax.set_ylabel("Distance to nearest T cell (µm)")
ax.set_ylim(0, 340)
ax.legend(loc="upper right", handlelength=1.0, handletextpad=0.4)
ax = axes[1]
for xi, r in enumerate([o, t]):
    ax.errorbar(xi, r.adjusted, yerr=[[r.adjusted - r.adj_lo], [r.adj_hi - r.adjusted]],
                fmt="o", ms=6, color="#0072B2", capsize=3, lw=1.3,
                markeredgecolor="white", markeredgewidth=0.7)
ax.axhline(1.0, color="#B00020", lw=0.9, ls=(0, (3, 2)))
ax.text(1.40, 0.95, "no enrichment", color="#B00020", fontsize=6.5,
        va="top", ha="right")
ax.set_xticks([0, 1]); ax.set_xticklabels(["10X", "TRACER"])
ax.set_xlim(-0.5, 1.45); ax.set_ylim(0, 1.15)
ax.set_ylabel("Density-adjusted proximity\n(VSIG4$^+$ TAM / all cells)")
fig.subplots_adjust(wspace=0.55)
S.save(fig, "S1_xenium_vsig4_proximity_density_audit", supp=True)
print("done")
