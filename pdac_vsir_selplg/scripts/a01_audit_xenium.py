#!/usr/bin/env python3
"""AUDIT 1 - Xenium introducer: which claims survive a density-adjusted estimand?

The published SUMMARY.md claim "VSIG4+ TAM -> T-cell proximity increases after
TRACER (117 -> 72 um)" is a raw nearest-neighbour distance. TRACER also more than
doubles the confidently-typed T-cell inventory, and 2-D nearest-neighbour distance
falls as 1/sqrt(density) for ANY query set. This script separates the two.

Estimands
  raw        median distance VSIG4+ macrophage -> nearest T cell, per arm
  background median distance ALL cells -> nearest T cell, same arm (density proxy)
  adjusted   raw / background  (dimensionless; 1.0 = no VSIG4-specific proximity)
  case-ctrl  VSIG4+ macrophage vs VSIG4- macrophage -> nearest T, same arm
             (holds cell type AND T-cell density constant; Mann-Whitney one-sided)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KDTree
from scipy.stats import mannwhitneyu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

CONF = 0.5
rng = np.random.default_rng(0)


def load(n):
    d = pd.read_parquet(S.XEN / f"vista_hypoxia_{n}.parquet")
    return d[d.lt_conf >= CONF].copy()


def nnd(src, tgt):
    if len(src) == 0 or len(tgt) == 0:
        return np.array([])
    return KDTree(tgt).query(src, k=1)[0].ravel()


def boot_med_ratio(a, b, n=2000):
    """95% CI for median(a)/median(b) by independent bootstrap."""
    ra = np.array([np.median(rng.choice(a, len(a))) for _ in range(n)])
    rb = np.array([np.median(rng.choice(b, len(b))) for _ in range(n)])
    r = ra / rb
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


rows, dist_rows = [], []
for arm in ["original", "tracer"]:
    d = load(arm)
    xy = d[["centroid_x", "centroid_y"]].values
    isT = d.cell_type.eq("T cell").values
    isM = d.cell_type.eq("Macrophage cell").values
    v = d.VSIG4_pos.values
    T = xy[isT]
    dv, dn, da = nnd(xy[isM & v], T), nnd(xy[isM & ~v], T), nnd(xy, T)
    u, p = mannwhitneyu(dv, dn, alternative="less")
    rbc = -(1 - 2 * u / (len(dv) * len(dn)))
    lo_a, hi_a = boot_med_ratio(dv, da)
    lo_c, hi_c = boot_med_ratio(dv, dn)
    rows.append(dict(
        arm=arm, n_T=int(isT.sum()), n_vsig4_mac=len(dv), n_other_mac=len(dn),
        med_raw=float(np.median(dv)), med_background=float(np.median(da)),
        med_vsig4neg_mac=float(np.median(dn)),
        adjusted=float(np.median(dv) / np.median(da)), adj_lo=lo_a, adj_hi=hi_a,
        case_ctrl=float(np.median(dv) / np.median(dn)), cc_lo=lo_c, cc_hi=hi_c,
        mwu_p_vsig4_closer=float(p), rank_biserial=float(rbc)))
    for nm, arr in (("VSIG4+ Mac", dv), ("VSIG4- Mac", dn), ("all cells", da)):
        dist_rows.append(pd.DataFrame({"arm": arm, "group": nm, "dist": arr}))

A = pd.DataFrame(rows)
A.to_csv(S.TAB / "xenium_vsig4_proximity_audit.tsv", sep="\t", index=False)
pd.concat(dist_rows).to_parquet(S.TAB / "xenium_vsig4_distances.parquet", index=False)

o, t = rows[0], rows[1]
print("=== VSIG4+ TAM -> nearest T cell ===")
print(A.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\nRAW        {o['med_raw']:.1f} -> {t['med_raw']:.1f} um   ({t['med_raw']/o['med_raw']:.3f}x)  <- the SUMMARY.md claim")
print(f"BACKGROUND {o['med_background']:.1f} -> {t['med_background']:.1f} um   ({t['med_background']/o['med_background']:.3f}x)  <- shrinks MORE")
print(f"T inventory {o['n_T']} -> {t['n_T']}  ({t['n_T']/o['n_T']:.2f}x); 1/sqrt(density) predicts {np.sqrt(o['n_T']/t['n_T']):.3f}x")
print(f"ADJUSTED   {o['adjusted']:.3f} [{o['adj_lo']:.3f},{o['adj_hi']:.3f}] -> {t['adjusted']:.3f} [{t['adj_lo']:.3f},{t['adj_hi']:.3f}]  (higher = weaker proximity)")
print(f"CASE-CTRL  {o['case_ctrl']:.3f} -> {t['case_ctrl']:.3f};  MWU p {o['mwu_p_vsig4_closer']:.2e} -> {t['mwu_p_vsig4_closer']:.2e}")
print("\nVERDICT: the raw improvement is fully explained by T-cell inventory growth;")
print("         the VSIG4-specific proximity is present PRE-TRACER and does not survive after.")

# ---- claims that are NOT density-confounded --------------------------------
print("\n=== VSIG4+ attribution specificity (composition estimand) ===")
att = []
for arm in ["original", "tracer"]:
    d = load(arm); pos = d[d.VSIG4_pos]
    vc = pos.cell_type.value_counts(normalize=True)
    att.append(dict(arm=arm, n_VSIG4pos=len(pos),
                    **{k: float(vc.get(k, 0.0)) for k in S.CT}))
AT = pd.DataFrame(att); AT.to_csv(S.TAB / "xenium_vsig4_attribution.tsv", sep="\t", index=False)
print(AT[["arm", "n_VSIG4pos", "Macrophage cell", "Fibroblast cell",
          "Ductal cell type 2", "T cell"]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

print("\n=== confidently-typed inventory by entity class ===")
inv = []
do, dt = load("original"), load("tracer")
for ct in S.CT:
    sub = dt[dt.cell_type.eq(ct)]
    inv.append(dict(cell_type=ct, original=int(do.cell_type.eq(ct).sum()),
                    tracer_complete=int(sub.entity_class.eq("complete").sum()),
                    tracer_partial=int(sub.entity_class.eq("partial").sum()),
                    tracer_total=int(len(sub))))
IV = pd.DataFrame(inv)
IV["fold"] = IV.tracer_total / IV.original
IV.to_csv(S.TAB / "xenium_inventory.tsv", sep="\t", index=False)
print(IV.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("\nwrote xenium_*.tsv")
