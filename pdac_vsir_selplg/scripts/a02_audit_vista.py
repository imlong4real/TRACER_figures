#!/usr/bin/env python3
"""AUDIT 2 - VisiumHD centrepiece: VISTA(VSIR)+ TAM / PSGL-1(SELPLG)+ T domains.

Fixes two things found in the audit of audit_v3/31_topology.py:

  (1) p was computed as mean(null >= obs), whose minimum is 0 and which is
      anti-conservative at the boundary. Replaced by the standard
      (1 + #{null >= obs}) / (1 + NPERM), and NPERM raised 200 -> 2000 so the
      resolution is 5e-4 rather than 5e-3.
  (2) the cohort claim was being read off six SEPARATE per-patient permutation
      tests ("3/6 significant"), which is far more conservative than the
      estimand actually being claimed. A cohort-level effect is added:
      mean log2 enrichment with a t-based 95% CI, an exact sign test on
      directional consistency, and Stouffer combination of the per-patient p.

Null (unchanged, and correct): marker status permuted WITHIN cell type, so
positions, cell-type map and the number of positive cells are all preserved.
Only which cells carry the marker is randomised - this holds TRACER's recovery
of extra cells constant and isolates spatial organisation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, t as tdist, binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

R_PRIMARY = 50
RADII = [20, 30, 50, 100]
NPERM = {50: 2000, 20: 500, 30: 500, 100: 500}
AXIS = ("VISTA-PSGL1", "VSIR", "TAM", "SELPLG", "T")


def ctx(d, w):
    if w == "TUMOR":
        return d.cell_type.isin(S.TUMOR).to_numpy()
    if w == "TAM":
        return (d.cell_type == "Macrophage cell").to_numpy()
    return (d.cell_type == "T cell").to_numpy()


def topo(xy, is_lig, R):
    n = len(xy)
    if n < 2:
        return dict(n_nodes=n, n_edges=0, n_mixed_edges=0, n_comp=n, n_mixed_comp=0,
                    frac_in_mixed=0.0, largest_mixed=0, largest_comp=int(n > 0))
    pairs = cKDTree(xy).query_pairs(R, output_type="ndarray")
    if len(pairs) == 0:
        return dict(n_nodes=n, n_edges=0, n_mixed_edges=0, n_comp=n, n_mixed_comp=0,
                    frac_in_mixed=0.0, largest_mixed=0, largest_comp=1)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    mixed_edges = int((is_lig[pairs[:, 0]] != is_lig[pairs[:, 1]]).sum())
    df = pd.DataFrame({"lab": lab, "lig": is_lig})
    gsz = df.groupby("lab").size()
    hb = df.groupby("lab").lig.nunique() == 2
    ml = set(hb[hb].index)
    return dict(n_nodes=n, n_edges=int(len(pairs)), n_mixed_edges=mixed_edges,
                n_comp=int(nc), n_mixed_comp=int(len(ml)),
                frac_in_mixed=float(np.isin(lab, list(ml)).mean()),
                largest_mixed=int(gsz[list(ml)].max()) if ml else 0,
                largest_comp=int(gsz.max()))


lab_ax, lg, lc, rg, rc = AXIS
rng = np.random.default_rng(11)
rows, contrib = [], []

for P in S.PTS:
    for arm in S.ARMS:
        d = S.load_arm(P, arm)
        am, tm = ctx(d, lc), ctx(d, rc)
        lm = (d[lg].to_numpy() > 0) & am
        rm = (d[rg].to_numpy() > 0) & tm
        lig, rec = d[lm], d[rm]
        base = dict(patient=P, response=S.RESP[P], arm=arm,
                    n_lig=int(lm.sum()), n_rec=int(rm.sum()))
        if lm.sum() < 5 or rm.sum() < 5:
            rows.append({**base, "R": R_PRIMARY, "status": "too_sparse"})
            print(f"[{P} {arm}] too_sparse  VISTA+TAM={lm.sum()} PSGL1+T={rm.sum()}", flush=True)
            continue
        xy = np.vstack([lig[["x", "y"]].to_numpy(), rec[["x", "y"]].to_numpy()])
        is_lig = np.r_[np.ones(lm.sum(), bool), np.zeros(rm.sum(), bool)]
        ai, ti = np.flatnonzero(am), np.flatnonzero(tm)
        XY = d[["x", "y"]].to_numpy()
        for R in RADII:
            o = topo(xy, is_lig, R)
            npm = NPERM[R]
            nm = np.empty(npm); nf = np.empty(npm); nl = np.empty(npm)
            for i in range(npm):
                xy2 = np.vstack([XY[rng.choice(ai, lm.sum(), replace=False)],
                                 XY[rng.choice(ti, rm.sum(), replace=False)]])
                oo = topo(xy2, is_lig, R)
                nm[i], nf[i], nl[i] = oo["n_mixed_edges"], oo["frac_in_mixed"], oo["largest_mixed"]
            r = {**base, "R": R, "status": "ok", "nperm": npm, **o,
                 "null_mixed_edges": float(nm.mean()), "null_mixed_edges_sd": float(nm.std()),
                 "p_mixed_edges": float((1 + (nm >= o["n_mixed_edges"]).sum()) / (1 + npm)),
                 "enrich_mixed_edges": float(o["n_mixed_edges"] / nm.mean()) if nm.mean() > 0 else np.nan,
                 "null_frac_in_mixed": float(nf.mean()),
                 "p_frac_in_mixed": float((1 + (nf >= o["frac_in_mixed"]).sum()) / (1 + npm)),
                 "null_largest_mixed": float(nl.mean()),
                 "p_largest_mixed": float((1 + (nl >= o["largest_mixed"]).sum()) / (1 + npm))}
            rows.append(r)
        # ---- whole vs partial contribution (post_all only) ----
        if arm == "post_all":
            et = np.r_[lig.etype.to_numpy(), rec.etype.to_numpy()]
            pairs = cKDTree(xy).query_pairs(R_PRIMARY, output_type="ndarray")
            mx = is_lig[pairs[:, 0]] != is_lig[pairs[:, 1]]
            mp = pairs[mx]
            invol = (et[mp[:, 0]] == "partial") | (et[mp[:, 1]] == "partial")
            contrib.append(dict(
                patient=P, response=S.RESP[P],
                lig_whole=int((lig.etype == "cell").sum()), lig_partial=int((lig.etype == "partial").sum()),
                rec_whole=int((rec.etype == "cell").sum()), rec_partial=int((rec.etype == "partial").sum()),
                n_mixed_edges=int(mx.sum()),
                mixed_edges_with_partial=int(invol.sum()),
                frac_mixed_needing_partial=float(invol.mean()) if mx.sum() else np.nan))
        print(f"[{P} {arm}] VISTA+TAM={lm.sum()} PSGL1+T={rm.sum()} done", flush=True)

T = pd.DataFrame(rows)
T.to_csv(S.TAB / "vista_psgl1_topology.tsv", sep="\t", index=False)
C = pd.DataFrame(contrib)
C.to_csv(S.TAB / "vista_psgl1_partial_contribution.tsv", sep="\t", index=False)

# ---------------- cohort-level meta-analysis ----------------
print("\n" + "=" * 78)
meta = []
for arm in ["post_whole", "post_all"]:
    q = T[(T.R == R_PRIMARY) & (T.arm == arm) & (T.status == "ok")].copy()
    q = q[q.enrich_mixed_edges.notna() & (q.enrich_mixed_edges > 0)]
    l2 = np.log2(q.enrich_mixed_edges.values)
    n = len(l2)
    m, se = l2.mean(), l2.std(ddof=1) / np.sqrt(n)
    ci = tdist.ppf(0.975, n - 1) * se
    npos = int((q.enrich_mixed_edges > 1).sum())
    sign_p = binomtest(npos, n, 0.5, alternative="two-sided").pvalue
    z = norm.isf(q.p_mixed_edges.values)
    stouffer_z = z.sum() / np.sqrt(n)
    meta.append(dict(arm=arm, n_patients=n,
                     mean_log2_enrich=m, ci_lo=m - ci, ci_hi=m + ci,
                     fold=2 ** m, fold_lo=2 ** (m - ci), fold_hi=2 ** (m + ci),
                     median_enrich=float(np.median(q.enrich_mixed_edges)),
                     n_enrich_gt1=npos, sign_test_p=sign_p,
                     stouffer_z=float(stouffer_z), stouffer_p=float(norm.sf(stouffer_z)),
                     n_indiv_p_lt_05=int((q.p_mixed_edges < 0.05).sum())))
    print(f"\n=== VISTA-PSGL1 cohort, arm={arm}, R={R_PRIMARY} um ===")
    print(q[["patient", "response", "n_lig", "n_rec", "n_edges", "n_mixed_edges",
             "null_mixed_edges", "enrich_mixed_edges", "p_mixed_edges",
             "n_mixed_comp", "largest_mixed", "frac_in_mixed"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"  mean log2 enrichment {m:+.3f} (95% CI {m-ci:+.3f},{m+ci:+.3f}) "
          f"-> fold {2**m:.2f}x ({2**(m-ci):.2f}-{2**(m+ci):.2f})")
    print(f"  directional consistency {npos}/{n} enrichment>1, sign test p={sign_p:.4f}")
    print(f"  Stouffer z={stouffer_z:.2f} p={norm.sf(stouffer_z):.2e};  "
          f"individually p<0.05 in {int((q.p_mixed_edges<0.05).sum())}/{n}")
pd.DataFrame(meta).to_csv(S.TAB / "vista_psgl1_cohort_meta.tsv", sep="\t", index=False)

print("\n=== whole vs partial contribution (post_all, R=50) ===")
print(C.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\nwrote vista_psgl1_*.tsv")
