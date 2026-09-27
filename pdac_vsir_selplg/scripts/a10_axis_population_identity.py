#!/usr/bin/env python3
"""AUDIT 10 - reference-anchored lineage identity for ALL FOUR checkpoint populations.

Same framework as a08, applied to both axes:
  VSIR+ myeloid, SELPLG+ T   (VISTA-PSGL1)
  LGALS9+ myeloid, HAVCR2+ T (GAL9-TIM3)

Each population is tested for its OWN lineage identity against depth-matched
entities of OTHER lineages (not against the marker-negative cells of the same
lineage - that is a within-lineage state contrast, kept only as a sensitivity
control). Modules are the reference-derived stable ones from a07, with VSIR,
SELPLG and every checkpoint gene excluded.
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.special import gammaln
from scipy.stats import mannwhitneyu, binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

PDAC = Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
INTER = S.PDAC6 / "intermediate"
CHUNK = 8_000_000
D_RARE = 20
POPS = [("VSIR+ myeloid", "Macrophage cell", "VSIR", "myeloid"),
        ("SELPLG+ T", "T cell", "SELPLG", "T"),
        ("LGALS9+ myeloid", "Macrophage cell", "LGALS9", "myeloid"),
        ("HAVCR2+ T", "T cell", "HAVCR2", "T")]
REFNAME = {"myeloid": "Macrophage cell", "T": "T cell"}

MOD = pd.read_csv(S.TAB / "reference_lineage_modules.tsv", sep="\t")
MODULES = {k: list(v.gene) for k, v in MOD.groupby("lineage")}
CENT = pd.read_csv(S.TAB / "reference_centroids.tsv", sep="\t", index_col=0)
UNION = list(CENT.columns)


def stream(h5path, want_idx, n_cells):
    out = np.zeros((n_cells, len(want_idx)), np.int32)
    pos = {g: k for k, g in enumerate(want_idx)}
    wset = np.array(sorted(want_idx))
    with h5py.File(h5path, "r") as f:
        indptr = f["X/indptr"][:]; nnz = f["X/data"].shape[0]; s0 = 0
        while s0 < nnz:
            s1 = min(s0 + CHUNK, nnz)
            idx = f["X/indices"][s0:s1]; dat = f["X/data"][s0:s1]
            k = np.isin(idx, wset)
            if k.any():
                r = np.searchsorted(indptr, np.flatnonzero(k) + s0, side="right") - 1
                c = np.array([pos[g] for g in idx[k]])
                np.add.at(out, (r, c), dat[k])
            s0 = s1
    return out


def match_depth(a_tx, b_tx, k=5):
    if len(b_tx) == 0 or len(a_tx) == 0:
        return np.array([], int)
    a = np.log10(np.maximum(a_tx, 1)); b = np.log10(np.maximum(b_tx, 1))
    order = np.argsort(b); bs = b[order]
    used = np.zeros(len(b), bool); picked = []
    for v in a:
        j = np.searchsorted(bs, v)
        for _ in range(k):
            lo, hi = j - 1, j; best, bd = -1, np.inf
            while lo >= 0 or hi < len(bs):
                if lo >= 0 and not used[order[lo]] and abs(bs[lo] - v) < bd:
                    best, bd = lo, abs(bs[lo] - v)
                if hi < len(bs) and not used[order[hi]] and abs(bs[hi] - v) < bd:
                    best, bd = hi, abs(bs[hi] - v)
                if best >= 0:
                    break
                lo -= 1; hi += 1
            if best < 0:
                break
            used[order[best]] = True; picked.append(order[best])
    return np.array(picked, int)


def rare_prob(n, k, D):
    n = np.asarray(n, float); k = np.asarray(k, float)
    out = np.zeros(len(n)); ok = (k > 0) & (n >= D)
    full = ok & ((n - k) < D); calc = ok & ~full
    out[full] = 1.0
    if calc.any():
        nn, kk = n[calc], k[calc]
        out[calc] = 1.0 - np.exp(gammaln(nn - kk + 1) - gammaln(nn - kk - D + 1)
                                 - (gammaln(nn + 1) - gammaln(nn - D + 1)))
    return out


rows = []
for P in S.PTS:
    genes = np.load(INTER / P / "genes.npy", allow_pickle=True)
    gp = {g: i for i, g in enumerate(genes)}
    un = [g for g in UNION if g in gp]
    meta = pd.read_parquet(INTER / P / "arm_final_meta.parquet")
    M = stream(PDAC / P / "data/tracer_results/outputs/cell_by_gene_tracer.h5ad",
               [gp[g] for g in un], len(meta))
    modidx = {k: [un.index(g) for g in v if g in un] for k, v in MODULES.items()}
    d = pd.DataFrame({"entity_id": meta.entity_id.values,
                      "n_tx": meta.n_tx.values.astype(np.float32),
                      "_pos": np.arange(len(meta), dtype=np.int64)})
    for k, ii in modidx.items():
        d[f"{k}_k"] = M[:, ii].sum(1).astype(np.float32)
    del meta; gc.collect()
    sp = S.load_arm(P, "post_all",
                    columns=["entity_id", "cell_type", "VSIR", "SELPLG", "LGALS9", "HAVCR2"])
    d = sp.merge(d, on="entity_id", how="inner")
    del sp; gc.collect()
    tx = np.maximum(d.n_tx.to_numpy(float), 1)
    for k in MODULES:
        cnt = d[f"{k}_k"].to_numpy(float)
        d[f"{k}_cpm"] = cnt / tx * 1e4
        d[f"{k}_pos"] = cnt > 0
        d[f"{k}_rare"] = rare_prob(tx, cnt, D_RARE)
    sub_idx = [UNION.index(g) for g in un]
    REFV = {}
    for t in CENT.index:
        v = np.log1p(CENT.loc[t].to_numpy(float))[sub_idx]
        REFV[t] = v / max(np.linalg.norm(v), 1e-9)

    def refsim(pos, ntx):
        X = np.log1p(M[pos].astype(np.float32) / np.maximum(ntx, 1)[:, None] * 1e6)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return pd.DataFrame({t: X @ REFV[t] for t in REFV})

    for popname, ctype, gene, lineage in POPS:
        own = d.cell_type.eq(ctype).to_numpy()
        pos = d[gene].to_numpy() > 0
        case = d[own & pos]
        if len(case) < 5:
            continue
        other = d[~own]                                  # identity control
        same_neg = d[own & ~pos]                         # state control
        mi = match_depth(case.n_tx.values, other.n_tx.values, 5)
        ctl = other.iloc[mi]
        mi2 = match_depth(case.n_tx.values, same_neg.n_tx.values, 5)
        stl = same_neg.iloc[mi2] if len(mi2) else same_neg.iloc[[]]
        cc = refsim(case["_pos"].to_numpy(), case.n_tx.to_numpy(float))
        ct = refsim(ctl["_pos"].to_numpy(), ctl.n_tx.to_numpy(float))
        rn = REFNAME[lineage]
        rows.append(dict(
            patient=P, population=popname, lineage=lineage, n_case=len(case),
            n_ctrl=len(ctl), med_tx_case=float(np.median(case.n_tx)),
            med_tx_ctrl=float(np.median(ctl.n_tx)),
            own_pos_case=float(case[f"{lineage}_pos"].mean()),
            own_pos_ctrl=float(ctl[f"{lineage}_pos"].mean()),
            own_pos_ctrl_raw=float(other[f"{lineage}_pos"].mean()),
            own_rare_case=float(case[f"{lineage}_rare"].mean()),
            own_rare_ctrl=float(ctl[f"{lineage}_rare"].mean()),
            own_cpm_case=float(case[f"{lineage}_cpm"].mean()),
            own_cpm_ctrl=float(ctl[f"{lineage}_cpm"].mean()),
            cos_case=float(cc[rn].mean()), cos_ctrl=float(ct[rn].mean()),
            p_own_pos=float(mannwhitneyu(case[f"{lineage}_pos"], ctl[f"{lineage}_pos"],
                                         alternative="greater").pvalue),
            p_cos=float(mannwhitneyu(cc[rn], ct[rn], alternative="greater").pvalue),
            ratio_raw=float(case[f"{lineage}_pos"].mean() / other[f"{lineage}_pos"].mean())
            if other[f"{lineage}_pos"].mean() > 0 else np.nan,
            ratio_depth=float(case[f"{lineage}_pos"].mean() / ctl[f"{lineage}_pos"].mean())
            if ctl[f"{lineage}_pos"].mean() > 0 else np.nan,
            ratio_rare=float(case[f"{lineage}_rare"].mean() / ctl[f"{lineage}_rare"].mean())
            if ctl[f"{lineage}_rare"].mean() > 0 else np.nan,
            ratio_state=float(case[f"{lineage}_pos"].mean() / stl[f"{lineage}_pos"].mean())
            if len(stl) and stl[f"{lineage}_pos"].mean() > 0 else np.nan))
    del d, M; gc.collect()
    print(f"[{P}] done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(S.TAB / "axis_population_identity.tsv", sep="\t", index=False)
pd.set_option("display.width", 260)
print("\n=== identity vs depth-matched OTHER-lineage entities ===")
print(R[["patient", "population", "n_case", "own_pos_case", "own_pos_ctrl",
         "ratio_depth", "ratio_rare", "ratio_raw", "p_own_pos", "cos_case", "cos_ctrl",
         "p_cos", "ratio_state"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\n=== summary ===")
for pop in R.population.unique():
    q = R[R.population == pop]
    for col, lab in (("ratio_depth", "depth-matched"), ("ratio_rare", "rarefied"),
                     ("ratio_raw", "raw")):
        v = q[col].dropna(); n = int((v > 1).sum())
        print(f"  {pop:16s} {lab:14s} {n}/{len(v)} > 1, median {v.median():6.2f}, "
              f"sign p={binomtest(n,len(v),0.5).pvalue:.4f}" if len(v) else
              f"  {pop:16s} {lab:14s} n/a")
    ns = int((q.p_cos < 0.05).sum())
    print(f"  {pop:16s} reference cosine higher than control in "
          f"{int((q.cos_case>q.cos_ctrl).sum())}/{len(q)} (p<0.05 in {ns})")
