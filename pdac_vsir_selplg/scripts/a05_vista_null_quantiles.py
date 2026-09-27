#!/usr/bin/env python3
"""Store the FULL within-cell-type null distribution for VISTA-PSGL1 at R=50 um.

a02 kept only the null mean/sd. A count statistic's null is skewed, so the
forest panel needs real quantiles rather than a normal approximation.
Same RNG stream and same permutation scheme as a02.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

R = 50
NPERM = 2000


def topo_mixed(xy, is_lig, R):
    n = len(xy)
    if n < 2:
        return 0, 0.0, 0
    pairs = cKDTree(xy).query_pairs(R, output_type="ndarray")
    if len(pairs) == 0:
        return 0, 0.0, 0
    me = int((is_lig[pairs[:, 0]] != is_lig[pairs[:, 1]]).sum())
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    df = pd.DataFrame({"lab": lab, "lig": is_lig})
    hb = df.groupby("lab").lig.nunique() == 2
    ml = set(hb[hb].index)
    gsz = df.groupby("lab").size()
    return me, float(np.isin(lab, list(ml)).mean()), (int(gsz[list(ml)].max()) if ml else 0)


rng = np.random.default_rng(11)
rows, draws = [], []
for P in S.PTS:
    for arm in S.ARMS:
        d = S.load_arm(P, arm)
        am = (d.cell_type == "Macrophage cell").to_numpy()
        tm = (d.cell_type == "T cell").to_numpy()
        lm = (d.VSIR.to_numpy() > 0) & am
        rm = (d.SELPLG.to_numpy() > 0) & tm
        if lm.sum() < 5 or rm.sum() < 5:
            continue
        XY = d[["x", "y"]].to_numpy()
        xy = np.vstack([XY[lm], XY[rm]])
        is_lig = np.r_[np.ones(lm.sum(), bool), np.zeros(rm.sum(), bool)]
        obs_me, obs_fm, obs_lm_ = topo_mixed(xy, is_lig, R)
        ai, ti = np.flatnonzero(am), np.flatnonzero(tm)
        nm = np.empty(NPERM)
        for i in range(NPERM):
            xy2 = np.vstack([XY[rng.choice(ai, lm.sum(), replace=False)],
                             XY[rng.choice(ti, rm.sum(), replace=False)]])
            nm[i] = topo_mixed(xy2, is_lig, R)[0]
        rows.append(dict(patient=P, arm=arm, response=S.RESP[P],
                         obs_mixed_edges=obs_me, null_mean=float(nm.mean()),
                         null_q025=float(np.percentile(nm, 2.5)),
                         null_q250=float(np.percentile(nm, 25)),
                         null_q750=float(np.percentile(nm, 75)),
                         null_q975=float(np.percentile(nm, 97.5)),
                         enrich=float(obs_me / nm.mean()) if nm.mean() > 0 else np.nan,
                         p=float((1 + (nm >= obs_me).sum()) / (1 + NPERM))))
        draws.append(pd.DataFrame({"patient": P, "arm": arm, "null_mixed_edges": nm}))
        print(f"[{P} {arm}] obs={obs_me} null={nm.mean():.2f} done", flush=True)

Q = pd.DataFrame(rows)
Q.to_csv(S.TAB / "vista_psgl1_null_quantiles.tsv", sep="\t", index=False)
pd.concat(draws).to_parquet(S.TAB / "vista_psgl1_null_draws.parquet", index=False)
print("\n" + Q.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
