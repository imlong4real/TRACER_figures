#!/usr/bin/env python3
"""M11 (rebuilt) lineage identity of all four checkpoint populations
M11b reference similarity + myeloid/T posterior
S10 GAL9-TIM3 under the full VISTA-PSGL1 framework (forest, radius, confidence)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

ID = pd.read_csv(S.TAB / "axis_population_identity.tsv", sep="\t")
GT = pd.read_csv(S.TAB / "gal9_tim3_topology.tsv", sep="\t")
GM = pd.read_csv(S.TAB / "gal9_tim3_cohort_meta.tsv", sep="\t")
LAB = {"VSIR+ myeloid": "VISTA$^+$ TAM", "SELPLG+ T": "PSGL-1$^+$ T",
       "LGALS9+ myeloid": "GAL9$^+$ TAM", "HAVCR2+ T": "TIM-3$^+$ T"}
ORDER = ["VSIR+ myeloid", "SELPLG+ T", "LGALS9+ myeloid", "HAVCR2+ T"]
DEPTH = [("ratio_raw", "raw", "#9E9E9E"),
         ("ratio_depth", "depth-matched", "#0072B2"),
         ("ratio_rare", "rarefied", "#D55E00")]

# ------------------------------------------------- M11 identity ---------------
fig, ax = plt.subplots(figsize=(4.0, 2.5))
yy = np.arange(len(ORDER))[::-1].astype(float)
for y, pop in zip(yy, ORDER):
    q = ID[ID.population == pop]
    for k, (col, lab, c) in enumerate(DEPTH):
        v = q[col].dropna().to_numpy()
        off = (1 - k) * 0.24
        ax.scatter(v, np.full(len(v), y + off), s=15, color=c, alpha=0.75,
                   lw=0.3, edgecolors="white", zorder=3)
        ax.plot([np.median(v)] * 2, [y + off - 0.10, y + off + 0.10],
                color=c, lw=2.0, zorder=4)
    n = int((q.ratio_depth > 1).sum())
    ax.text(78, y, f"{n}/6", fontsize=8, va="center", ha="left", fontweight="bold",
            color="#009E73" if n >= 5 else "#B00020")
ax.axvline(1.0, color="#B00020", lw=1.0, ls=(0, (3, 2)), zorder=1)
ax.set_xscale("log")
ax.set_xticks([1, 3, 10, 30]); ax.set_xticklabels(["1", "3", "10", "30"])
ax.minorticks_off()
ax.set_xlim(0.75, 110)
ax.set_yticks(yy); ax.set_yticklabels([LAB[p] for p in ORDER])
ax.set_ylim(-0.55, len(ORDER) - 0.45)
ax.set_xlabel("Own-lineage module vs other-lineage entities")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=5, label=l)
                   for _, l, c in DEPTH],
          ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15),
          handletextpad=0.3, columnspacing=1.1)
S.save(fig, "M11_hd_lineage_identity")

# --------------------------------------- M11b reference similarity ------------
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1))
ax = axes[0]
x = np.arange(len(ORDER))
for i, pop in enumerate(ORDER):
    q = ID[ID.population == pop]
    ax.scatter(np.full(len(q), i) - 0.13, q.cos_case, s=16, color="#D55E00",
               lw=0.3, edgecolors="white", zorder=3)
    ax.scatter(np.full(len(q), i) + 0.13, q.cos_ctrl, s=16, color="#9E9E9E",
               lw=0.3, edgecolors="white", zorder=3)
    for a, b in zip(q.cos_case, q.cos_ctrl):
        ax.plot([i - 0.13, i + 0.13], [a, b], color="#CCCCCC", lw=0.5, zorder=1)
ax.set_xticks(x); ax.set_xticklabels([LAB[p] for p in ORDER], rotation=30, ha="right")
ax.set_ylabel("Cosine to reference\nlineage centroid")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#D55E00", ms=5, label="population"),
                   Line2D([], [], marker="o", ls="", color="#9E9E9E", ms=5, label="other lineages")],
          loc="upper left", handletextpad=0.3, columnspacing=0.8)
ax.set_ylim(0, max(ID.cos_case.max() * 1.45, 0.2))
ax = axes[1]
mye = ID[ID.lineage == "myeloid"]
ax.scatter(mye.own_pos_ctrl, mye.own_pos_case, s=22, color="#D55E00", lw=0.4,
           edgecolors="white", label="myeloid", zorder=3)
tc = ID[ID.lineage == "T"]
ax.scatter(tc.own_pos_ctrl, tc.own_pos_case, s=22, color="#0072B2", lw=0.4,
           edgecolors="white", marker="^", label="T", zorder=3)
lim = max(ID.own_pos_case.max(), ID.own_pos_ctrl.max()) * 1.12
ax.plot([0, lim], [0, lim], color="#999999", lw=0.8, ls=(0, (3, 2)))
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Control (other lineages)")
ax.set_ylabel("Checkpoint$^+$ population")
ax.set_title("")
ax.legend(loc="lower right", handletextpad=0.3)
fig.subplots_adjust(wspace=0.45)
S.save(fig, "M11b_hd_reference_similarity")

# ------------------------------------------------- S10 GAL9-TIM3 --------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
ax = axes[0]
q = GT[(GT.axis == "GAL9-TIM3") & (GT.R == 50) & (GT.arm == "post_all") &
       (GT.conf_q == 0.0) & (GT.status == "ok")].set_index("patient").reindex(S.PTS)
y = np.arange(len(S.PTS))[::-1].astype(float)
for yi, (_, r) in zip(y, q.iterrows()):
    if pd.isna(r.enrich_mixed_edges):
        continue
    lo, hi = r.null_q025 / r.null_mixed_edges, r.null_q975 / r.null_mixed_edges
    ax.plot([lo, hi], [yi, yi], color="#D2D2D2", lw=5.0, solid_capstyle="butt", zorder=1)
    ax.scatter([r.enrich_mixed_edges], [yi], s=34, color=S.RC[r.response],
               edgecolors="white", linewidths=0.7, zorder=3)
    if r.p_mixed_edges < 0.05:
        ax.text(r.enrich_mixed_edges + 0.2, yi + 0.1, "*", fontsize=9, va="center")
m = GM[(GM.axis == "GAL9-TIM3") & (GM.arm == "post_all")].iloc[0]
ax.plot([m.fold_lo, m.fold_hi], [-1.2, -1.2], color="#111111", lw=1.6, zorder=3)
ax.scatter([m.fold], [-1.2], marker="D", s=38, color="#111111", zorder=4)
ax.axvline(1.0, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=0)
ax.set_yticks(list(y) + [-1.2]); ax.set_yticklabels(list(S.PTS) + ["cohort"])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(0, 7.2); ax.set_ylim(-1.9, len(S.PTS) - 0.4)
ax.set_xlabel("Mixed-edge enrichment")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.axhline(-0.6, color="#DDDDDD", lw=0.7)

ax = axes[1]
piv = (GT[(GT.axis == "GAL9-TIM3") & (GT.arm == "post_all") & (GT.conf_q == 0.0) &
          (GT.status == "ok")].pivot_table(index="patient", columns="R",
                                           values="enrich_mixed_edges").reindex(S.PTS))
for p in S.PTS:
    if p in piv.index:
        ax.plot(piv.columns, piv.loc[p], marker="o", ms=3, lw=1.0, color=S.RC[S.RESP[p]],
                ls="-" if S.RESP[p] else (0, (3, 1.6)))
ax.axhline(1.0, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_xlabel("Radius (µm)"); ax.set_ylabel("Mixed-edge enrichment")
ax.set_xticks([20, 30, 50, 100])

ax = axes[2]
pc = (GT[(GT.axis == "GAL9-TIM3") & (GT.arm == "post_all") & (GT.R == 50) &
         (GT.status == "ok")].pivot_table(index="patient", columns="conf_q",
                                          values="enrich_mixed_edges").reindex(S.PTS))
for p in S.PTS:
    if p in pc.index:
        ax.plot(range(len(pc.columns)), pc.loc[p], marker="o", ms=3, lw=1.0,
                color=S.RC[S.RESP[p]], ls="-" if S.RESP[p] else (0, (3, 1.6)))
ax.axhline(1.0, color="#666666", lw=0.8, ls=(0, (3, 2)))
ax.set_xticks(range(len(pc.columns)))
ax.set_xticklabels([f"{c:.0%}" for c in pc.columns])
ax.set_xlabel("Label-confidence floor (quantile)")
ax.set_ylabel("Mixed-edge enrichment")
fig.subplots_adjust(wspace=0.45)
S.save(fig, "S10_hd_gal9_tim3_framework", supp=True)
print("done")
