#!/usr/bin/env python3
"""VisiumHD cohort - VISTA(VSIR)-PSGL1(SELPLG) centrepiece, quantitative panels.

M8  checkpoint axes that become measurable, per patient and arm
M9  per-patient mixed-edge enrichment vs the within-cell-type null + cohort effect
M10 whole-cell vs partial-cell contribution
M11 independent lineage-marker validation of BOTH populations (depth-matched)
M14 domain structure: components, size, continuity
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

TP = pd.read_csv(S.TAB / "vista_psgl1_topology.tsv", sep="\t")
MT = pd.read_csv(S.TAB / "vista_psgl1_cohort_meta.tsv", sep="\t")
CB = pd.read_csv(S.TAB / "vista_psgl1_partial_contribution.tsv", sep="\t")
LV = pd.read_csv(S.TAB / "lineage_depth_matched.tsv", sep="\t")
NQ = pd.read_csv(S.TAB / "vista_psgl1_null_quantiles.tsv", sep="\t")
AX = pd.read_csv(S.PDAC6 / "audit_v3/axis_evaluability.tsv", sep="\t")

# ------------------------------------------------- M8 axes become measurable -
piv = (AX[AX.evaluable] .groupby(["patient", "arm"]).size()
       .unstack(fill_value=0).reindex(S.PTS).reindex(columns=S.ARMS, fill_value=0))
fig, ax = plt.subplots(figsize=(3.3, 2.0))
x = np.arange(len(S.PTS)); w = 0.27
for i, arm in enumerate(S.ARMS):
    ax.bar(x + (i - 1) * w, piv[arm].values, w, color=S.ARMC[arm], label=S.ARMLAB[arm])
ax.set_xticks(x)
ax.set_xticklabels([f"{p}\n{'R' if S.RESP[p] else 'NR'}" for p in S.PTS])
ax.set_ylabel("Checkpoint axes detectable")
ax.set_ylim(0, max(piv.values.max() + 2.2, 4))
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17),
          handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
S.save(fig, "M8_hd_axes_detectable")

# ----------------------------------------- M9 enrichment forest + cohort -----
q = TP[(TP.R == 50) & (TP.arm == "post_all") & (TP.status == "ok")].copy()
q = q.set_index("patient").reindex(S.PTS).reset_index()
nq = NQ[NQ.arm == "post_all"].set_index("patient").reindex(S.PTS).reset_index()
m = MT[MT.arm == "post_all"].iloc[0]
fig, ax = plt.subplots(figsize=(3.5, 2.35))
y = np.arange(len(S.PTS))[::-1].astype(float)
for yi, (_, r), (_, n) in zip(y, q.iterrows(), nq.iterrows()):
    nullmean = r.null_mixed_edges
    lo, hi = n.null_q025 / nullmean, n.null_q975 / nullmean
    ax.plot([lo, hi], [yi, yi], color="#D2D2D2", lw=5.0, solid_capstyle="butt", zorder=1)
    ax.scatter([r.enrich_mixed_edges], [yi], s=36, zorder=3,
               color=S.RC[S.RESP[r.patient]], edgecolors="white", linewidths=0.7)
    star = "***" if r.p_mixed_edges < 0.001 else "**" if r.p_mixed_edges < 0.01 else \
           "*" if r.p_mixed_edges < 0.05 else ""
    if star:
        ax.text(r.enrich_mixed_edges + 0.16, yi + 0.10, star, fontsize=8, va="center")
ax.axvline(1.0, color="#666666", lw=0.8, ls=(0, (3, 2)), zorder=0)
ycoh = -1.25
ax.plot([m.fold_lo, m.fold_hi], [ycoh, ycoh], color="#111111", lw=1.6, zorder=3)
ax.scatter([m.fold], [ycoh], marker="D", s=42, color="#111111", zorder=4)
ax.set_yticks(list(y) + [ycoh])
ax.set_yticklabels(list(S.PTS) + ["cohort"])
for t, p in zip(ax.get_yticklabels()[:-1], S.PTS):
    t.set_color(S.RC[S.RESP[p]])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlabel("Mixed-edge enrichment vs within-type null")
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xlim(0, 5.9); ax.set_ylim(ycoh - 0.75, len(S.PTS) - 0.35)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.axhline(-0.62, color="#DDDDDD", lw=0.7)
ax.text(m.fold_hi + 0.22, ycoh, f"{m.fold:.2f}× (95% CI {m.fold_lo:.2f}–{m.fold_hi:.2f})",
        fontsize=6.8, va="center", ha="left", fontweight="bold")
S.save(fig, "M9_hd_vista_enrichment_forest")

# ------------------------------------------- M10 whole vs partial ------------
CB = CB.set_index("patient").reindex(S.PTS).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.05))
ax = axes[0]
x = np.arange(len(S.PTS)); w = 0.36
ax.bar(x - w / 2, CB.lig_whole, w, color=S.ENTC["cell"], label="whole")
ax.bar(x - w / 2, CB.lig_partial, w, bottom=CB.lig_whole, color=S.ENTC["partial"], label="partial")
ax.bar(x + w / 2, CB.rec_whole, w, color=S.ENTC["cell"])
ax.bar(x + w / 2, CB.rec_partial, w, bottom=CB.rec_whole, color=S.ENTC["partial"])
ax.set_xticks(x); ax.set_xticklabels(S.PTS, rotation=45, ha="right")
ax.set_ylabel("Checkpoint$^+$ entities")
ax.set_ylim(0, 405)
ax.text(0.0, 1.015, "left bar VISTA$^+$ TAM · right bar PSGL-1$^+$ T", transform=ax.transAxes,
        fontsize=6.5, va="bottom", ha="left", color="#555555")
ax.legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.99),
          handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
ax = axes[1]
ax.bar(x, CB.frac_mixed_needing_partial * 100, 0.6,
       color=[S.RC[S.RESP[p]] for p in CB.patient])
ax.set_xticks(x); ax.set_xticklabels(S.PTS, rotation=45, ha="right")
ax.set_ylabel("Domain edges requiring\na partial cell (%)")
ax.set_ylim(0, 108)
fig.subplots_adjust(wspace=0.5)
S.save(fig, "M10_hd_vista_partial_contribution")

# ---------------------------------------------- M14 domain structure ---------
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.05))
ax = axes[0]
for i, arm in enumerate(["post_whole", "post_all"]):
    s = TP[(TP.R == 50) & (TP.arm == arm) & (TP.status == "ok")].set_index("patient").reindex(S.PTS)
    ax.bar(np.arange(len(S.PTS)) + (i - 0.5) * 0.38, s.n_mixed_comp.fillna(0), 0.38,
           color=S.ARMC[arm], label=S.ARMLAB[arm])
ax.set_xticks(np.arange(len(S.PTS))); ax.set_xticklabels(S.PTS, rotation=45, ha="right")
ax.set_ylabel("Checkpoint domains\n(mixed components)")
ax.set_ylim(0, 33)
ax.legend(ncol=2, loc="upper left", handlelength=1.0, handletextpad=0.4)
ax = axes[1]
s = TP[(TP.R == 50) & (TP.arm == "post_all") & (TP.status == "ok")].set_index("patient").reindex(S.PTS)
n = NQ[NQ.arm == "post_all"].set_index("patient").reindex(S.PTS)
x = np.arange(len(S.PTS))
ax.bar(x, s.largest_mixed.fillna(0), 0.6, color=[S.RC[S.RESP[p]] for p in S.PTS])
ax.set_xticks(x); ax.set_xticklabels(S.PTS, rotation=45, ha="right")
ax.set_ylabel("Largest domain\n(cells)")
fig.subplots_adjust(wspace=0.5)
S.save(fig, "M14_hd_domain_structure")

# legend chip: responder colour key
fig, ax = plt.subplots(figsize=(2.2, 0.32)); ax.axis("off")
ax.legend(handles=[Patch(facecolor=S.RC[1], label="responder"),
                   Patch(facecolor=S.RC[0], label="non-responder")],
          ncol=2, loc="center", handlelength=1.0, handletextpad=0.4, columnspacing=1.1)
S.save(fig, "M0_legend_response")
print("done")
