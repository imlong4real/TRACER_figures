#!/usr/bin/env python3
"""AUDIT 9 - GAL9(LGALS9)-TIM3(HAVCR2) under the SAME final framework as VISTA-PSGL1.

NPERM 2,000 at the primary radius (1,000 elsewhere), within-cell-type marker
permutation, corrected p = (1+#{null>=obs})/(1+NPERM), patient-level effect with
a cohort t-CI, sign test and Stouffer combination, whole vs whole+partial,
radius sensitivity 20-100 um, and a label-confidence sensitivity sweep.
Independent lineage validation of both populations is done in a10.
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

RADII = [20, 30, 50, 100]
R_PRIMARY = 50
NPERM = {50: 2000, 20: 1000, 30: 1000, 100: 1000}
CONF_Q = [0.0, 0.25, 0.50]          # label-confidence sensitivity (quantiles)
AXES = {"GAL9-TIM3": ("LGALS9", "TAM", "HAVCR2", "T"),
        "VISTA-PSGL1": ("VSIR", "TAM", "SELPLG", "T")}
COLS = ["x", "y", "cell_type", "confidence", "LGALS9", "HAVCR2", "VSIR", "SELPLG"]


def topo(xy, is_lig, R):
    n = len(xy)
    if n < 2:
        return dict(n_nodes=n, n_edges=0, n_mixed_edges=0, n_comp=n, n_mixed_comp=0,
                    frac_in_mixed=0.0, largest_mixed=0)
    pairs = cKDTree(xy).query_pairs(R, output_type="ndarray")
    if len(pairs) == 0:
        return dict(n_nodes=n, n_edges=0, n_mixed_edges=0, n_comp=n, n_mixed_comp=0,
                    frac_in_mixed=0.0, largest_mixed=0)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    me = int((is_lig[pairs[:, 0]] != is_lig[pairs[:, 1]]).sum())
    df = pd.DataFrame({"lab": lab, "lig": is_lig})
    hb = df.groupby("lab").lig.nunique() == 2
    ml = set(hb[hb].index)
    gsz = df.groupby("lab").size()
    return dict(n_nodes=n, n_edges=int(len(pairs)), n_mixed_edges=me, n_comp=int(nc),
                n_mixed_comp=int(len(ml)),
                frac_in_mixed=float(np.isin(lab, list(ml)).mean()),
                largest_mixed=int(gsz[list(ml)].max()) if ml else 0)


rng = np.random.default_rng(23)
rows = []
for axis, (lg, lctx, rg, rctx) in AXES.items():
    for P in S.PTS:
        for arm in ["post_whole", "post_all"]:
            d0 = S.load_arm(P, arm, columns=COLS)
            for qi, q in enumerate(CONF_Q):
                thr = d0.confidence.quantile(q) if q > 0 else -np.inf
                d = d0[d0.confidence >= thr]
                am = (d.cell_type == "Macrophage cell").to_numpy()
                tm = (d.cell_type == "T cell").to_numpy()
                lm = (d[lg].to_numpy() > 0) & am
                rm = (d[rg].to_numpy() > 0) & tm
                base = dict(axis=axis, patient=P, response=S.RESP[P], arm=arm,
                            conf_q=q, conf_thr=float(thr if np.isfinite(thr) else 0),
                            n_lig=int(lm.sum()), n_rec=int(rm.sum()))
                if lm.sum() < 5 or rm.sum() < 5:
                    rows.append({**base, "R": R_PRIMARY, "status": "too_sparse"})
                    continue
                XY = d[["x", "y"]].to_numpy()
                xy = np.vstack([XY[lm], XY[rm]])
                is_lig = np.r_[np.ones(lm.sum(), bool), np.zeros(rm.sum(), bool)]
                ai, ti = np.flatnonzero(am), np.flatnonzero(tm)
                radii = RADII if q == 0.0 else [R_PRIMARY]
                for R in radii:
                    o = topo(xy, is_lig, R)
                    npm = NPERM[R] if q == 0.0 else 500
                    nm = np.empty(npm)
                    for i in range(npm):
                        xy2 = np.vstack([XY[rng.choice(ai, lm.sum(), replace=False)],
                                         XY[rng.choice(ti, rm.sum(), replace=False)]])
                        nm[i] = topo(xy2, is_lig, R)["n_mixed_edges"]
                    rows.append({**base, "R": R, "status": "ok", "nperm": npm, **o,
                                 "null_mixed_edges": float(nm.mean()),
                                 "null_q025": float(np.percentile(nm, 2.5)),
                                 "null_q975": float(np.percentile(nm, 97.5)),
                                 "p_mixed_edges": float((1 + (nm >= o["n_mixed_edges"]).sum()) / (1 + npm)),
                                 "enrich_mixed_edges": float(o["n_mixed_edges"] / nm.mean())
                                 if nm.mean() > 0 else np.nan})
                print(f"[{axis} {P} {arm} q={q}] lig={lm.sum()} rec={rm.sum()} done", flush=True)
            del d0
T = pd.DataFrame(rows)
T.to_csv(S.TAB / "gal9_tim3_topology.tsv", sep="\t", index=False)

meta = []
for axis in AXES:
    for arm in ["post_whole", "post_all"]:
        q = T[(T.axis == axis) & (T.R == R_PRIMARY) & (T.arm == arm) & (T.conf_q == 0.0) &
              (T.status == "ok") & T.enrich_mixed_edges.notna() & (T.enrich_mixed_edges > 0)]
        if len(q) < 2:
            continue
        l2 = np.log2(q.enrich_mixed_edges.values); n = len(l2)
        m, se = l2.mean(), l2.std(ddof=1) / np.sqrt(n)
        ci = tdist.ppf(0.975, n - 1) * se
        npos = int((q.enrich_mixed_edges > 1).sum())
        z = norm.isf(q.p_mixed_edges.values); sz = z.sum() / np.sqrt(n)
        meta.append(dict(axis=axis, arm=arm, n_patients=n, fold=2 ** m,
                         fold_lo=2 ** (m - ci), fold_hi=2 ** (m + ci),
                         median_enrich=float(np.median(q.enrich_mixed_edges)),
                         n_gt1=npos, sign_p=binomtest(npos, n, 0.5).pvalue,
                         stouffer_z=float(sz), stouffer_p=float(norm.sf(sz)),
                         n_p_lt_05=int((q.p_mixed_edges < 0.05).sum())))
M = pd.DataFrame(meta); M.to_csv(S.TAB / "gal9_tim3_cohort_meta.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print("\n=== cohort meta, R=50, full confidence ===")
print(M.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
for axis in AXES:
    q = T[(T.axis == axis) & (T.R == 50) & (T.arm == "post_all") & (T.conf_q == 0.0) &
          (T.status == "ok")]
    print(f"\n=== {axis}, post_all, R=50 ===")
    print(q[["patient", "response", "n_lig", "n_rec", "n_mixed_edges", "null_mixed_edges",
             "enrich_mixed_edges", "p_mixed_edges", "n_mixed_comp", "largest_mixed"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\n=== radius sensitivity (post_all, enrichment) ===")
print(T[(T.arm == "post_all") & (T.conf_q == 0.0) & (T.status == "ok")]
      .pivot_table(index=["axis", "patient"], columns="R", values="enrich_mixed_edges")
      .to_string(float_format=lambda v: f"{v:,.2f}"))
print("\n=== label-confidence sensitivity (post_all, R=50, enrichment) ===")
print(T[(T.arm == "post_all") & (T.R == 50) & (T.status == "ok")]
      .pivot_table(index=["axis", "patient"], columns="conf_q", values="enrich_mixed_edges")
      .to_string(float_format=lambda v: f"{v:,.2f}"))
