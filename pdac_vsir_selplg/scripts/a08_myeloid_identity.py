#!/usr/bin/env python3
"""AUDIT 8 - are VSIR+ entities genuinely MYELOID?

The previous test asked "are VSIR+ macrophages more macrophage-like than VSIR-
macrophages?" - a within-lineage state contrast that a VISTA-high macrophage
programme can legitimately fail. The identity question needs NON-myeloid
controls and a reference anchor.

Tests, all with reference-derived stable modules (a07; VSIR/SELPLG and every
checkpoint gene excluded):
  A  myeloid module vs depth-matched NON-myeloid entities   <- the identity test
  B  reference similarity: cosine of the entity profile to each Chijimatsu
     lineage centroid; is the argmax myeloid, and how does the myeloid cosine
     compare to the same non-myeloid controls
  C  myeloid posterior p_Macrophage_cell from the frozen classifier
  D  off-lineage T / B / epithelial modules
  E  within-lineage contrast vs VSIR- macrophages, kept as a SENSITIVITY control
     rather than a disqualification criterion

Depth is handled three ways: raw, depth-matched (1:5 on log10 n_tx) and
rarefied - the exact probability that a module count survives subsampling every
entity to a fixed depth D, 1 - C(n-k, D)/C(n, D), so depth is removed
analytically rather than by matching.
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
rng = np.random.default_rng(7)

MOD = pd.read_csv(S.TAB / "reference_lineage_modules.tsv", sep="\t")
MODULES = {k: list(v.gene) for k, v in MOD.groupby("lineage")}
CENT = pd.read_csv(S.TAB / "reference_centroids.tsv", sep="\t", index_col=0)
UNION = list(CENT.columns)
REF_L2 = {}
for t in CENT.index:
    v = np.log1p(CENT.loc[t].to_numpy(float))
    REF_L2[t] = v / max(np.linalg.norm(v), 1e-9)


def stream(h5path, want_idx, n_cells):
    out = np.zeros((n_cells, len(want_idx)), np.int32)
    pos = {g: k for k, g in enumerate(want_idx)}
    wset = np.array(sorted(want_idx))
    with h5py.File(h5path, "r") as f:
        indptr = f["X/indptr"][:]
        nnz = f["X/data"].shape[0]
        s0 = 0
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


def match_depth(case_tx, ctrl_tx, k=5):
    if len(ctrl_tx) == 0 or len(case_tx) == 0:
        return np.array([], int)
    a = np.log10(np.maximum(case_tx, 1)); b = np.log10(np.maximum(ctrl_tx, 1))
    order = np.argsort(b); bs = b[order]
    used = np.zeros(len(b), bool); picked = []
    for v in a:
        j = np.searchsorted(bs, v)
        for _ in range(k):
            lo, hi = j - 1, j
            best, bd = -1, np.inf
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
    """P(>=1 module count survives subsampling to depth D), exact hypergeometric."""
    n = np.asarray(n, float); k = np.asarray(k, float)
    out = np.zeros(len(n))
    ok = (k > 0) & (n >= D)
    full = ok & ((n - k) < D)
    calc = ok & ~full
    out[full] = 1.0
    if calc.any():
        nn, kk = n[calc], k[calc]
        lp0 = (gammaln(nn - kk + 1) - gammaln(nn - kk - D + 1)
               - (gammaln(nn + 1) - gammaln(nn - D + 1)))
        out[calc] = 1.0 - np.exp(lp0)
    return out


rows, per_cell = [], []
for P in S.PTS:
    genes = np.load(INTER / P / "genes.npy", allow_pickle=True)
    gp = {g: i for i, g in enumerate(genes)}
    un = [g for g in UNION if g in gp]
    M = stream(PDAC / P / "data/tracer_results/outputs/cell_by_gene_tracer.h5ad",
               [gp[g] for g in un],
               len(pd.read_parquet(INTER / P / "arm_final_meta.parquet")))
    meta = pd.read_parquet(INTER / P / "arm_final_meta.parquet")
    modcols = {k: [un.index(g) for g in v if g in un] for k, v in MODULES.items()}
    # collapse to module sums immediately; the full 94-gene matrix stays as a
    # positional array and is sliced only for the compared subsets (the login
    # cgroup cannot hold a float copy of 5x10^5 x 94)
    dfm = pd.DataFrame({"entity_id": meta.entity_id.values,
                        "n_tx": meta.n_tx.values.astype(np.float32),
                        "_pos": np.arange(len(meta), dtype=np.int64)})
    for k, ii in modcols.items():
        dfm[f"{k}_k"] = M[:, ii].sum(1).astype(np.float32)
    del meta; gc.collect()

    lab = pd.concat([pd.read_parquet(INTER / P / "labels_post_whole.parquet",
                                     columns=["entity_id", "cell_type", "p_Macrophage_cell",
                                              "confidence"]),
                     pd.read_parquet(INTER / P / "labels_post_partial.parquet",
                                     columns=["entity_id", "cell_type", "p_Macrophage_cell",
                                              "confidence"])], ignore_index=True)
    pd.options.mode.chained_assignment = None
    sp = S.load_arm(P, "post_all", columns=["entity_id", "cell_type", "VSIR", "etype"])
    d = sp.merge(dfm, on="entity_id", how="inner").merge(
        lab[["entity_id", "p_Macrophage_cell"]], on="entity_id", how="left")
    del dfm, sp, lab; gc.collect()

    tx = np.maximum(d.n_tx.to_numpy(float), 1)
    for k in MODULES:
        cnt = d[f"{k}_k"].to_numpy(float)
        d[f"{k}_cpm"] = cnt / tx * 1e4
        d[f"{k}_pos"] = cnt > 0
        d[f"{k}_rare"] = rare_prob(tx, cnt, D_RARE)
    sub_idx = [UNION.index(g) for g in un]
    REFV = {}
    for t in REF_L2:
        v = REF_L2[t][sub_idx]
        REFV[t] = v / max(np.linalg.norm(v), 1e-9)

    def refsim(pos, ntx):
        """cosine to each reference lineage centroid, for selected rows only"""
        X = np.log1p(M[pos].astype(np.float32) / np.maximum(ntx, 1)[:, None] * 1e6)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        C = pd.DataFrame({t: X @ REFV[t] for t in REFV})
        return C["Macrophage cell"].to_numpy(), C.idxmax(1).to_numpy()

    is_mac = d.cell_type.eq("Macrophage cell").to_numpy()
    vsir = d.VSIR.to_numpy() > 0
    case = d[is_mac & vsir]
    ctrl_nonmye = d[~is_mac]                       # identity control
    ctrl_vsirneg = d[is_mac & ~vsir]               # state control (sensitivity)
    if len(case) < 5:
        continue
    mi = match_depth(case.n_tx.values, ctrl_nonmye.n_tx.values, 5)
    mnm = ctrl_nonmye.iloc[mi]
    mi2 = match_depth(case.n_tx.values, ctrl_vsirneg.n_tx.values, 5)
    mvn = ctrl_vsirneg.iloc[mi2] if len(mi2) else ctrl_vsirneg.iloc[[]]
    for grp in (case, mnm, mvn):
        if len(grp):
            c, a = refsim(grp["_pos"].to_numpy(), grp.n_tx.to_numpy(float))
            grp["cos_myeloid"] = c; grp["ref_argmax"] = a
    nm_raw_s = ctrl_nonmye.sample(min(len(ctrl_nonmye), 20000), random_state=0)
    c, a = refsim(nm_raw_s["_pos"].to_numpy(), nm_raw_s.n_tx.to_numpy(float))
    nm_raw_s["cos_myeloid"] = c; nm_raw_s["ref_argmax"] = a

    def blk(tag, grp):
        return {f"{tag}_n": len(grp),
                f"{tag}_med_tx": float(np.median(grp.n_tx)) if len(grp) else np.nan,
                f"{tag}_mye_pos": float(grp.myeloid_pos.mean()) if len(grp) else np.nan,
                f"{tag}_mye_cpm": float(grp.myeloid_cpm.mean()) if len(grp) else np.nan,
                f"{tag}_mye_rare": float(grp.myeloid_rare.mean()) if len(grp) else np.nan,
                f"{tag}_cos_mye": float(grp.cos_myeloid.mean()) if len(grp) else np.nan,
                f"{tag}_p_mac": float(grp.p_Macrophage_cell.mean()) if len(grp) else np.nan,
                f"{tag}_ref_mac": float(grp.ref_argmax.eq("Macrophage cell").mean()) if len(grp) else np.nan}

    r = dict(patient=P, response=S.RESP[P])
    r.update(blk("case", case)); r.update(blk("nonmye", mnm))
    r.update(blk("nonmye_raw", nm_raw_s)); r.update(blk("vsirneg", mvn))
    for k in ["T", "B", "epithelial"]:
        r[f"case_{k}_cpm"] = float(case[f"{k}_cpm"].mean())
        r[f"nonmye_{k}_cpm"] = float(mnm[f"{k}_cpm"].mean()) if len(mnm) else np.nan
    r["p_mye_pos_vs_nonmye"] = float(mannwhitneyu(case.myeloid_pos, mnm.myeloid_pos,
                                                  alternative="greater").pvalue) if len(mnm) else np.nan
    r["p_cos_vs_nonmye"] = float(mannwhitneyu(case.cos_myeloid, mnm.cos_myeloid,
                                              alternative="greater").pvalue) if len(mnm) else np.nan
    r["ratio_raw"] = r["case_mye_pos"] / r["nonmye_raw_mye_pos"] if r["nonmye_raw_mye_pos"] else np.nan
    r["ratio_depthmatched"] = r["case_mye_pos"] / r["nonmye_mye_pos"] if r["nonmye_mye_pos"] else np.nan
    r["ratio_rarefied"] = r["case_mye_rare"] / r["nonmye_mye_rare"] if r["nonmye_mye_rare"] else np.nan
    r["ratio_vs_vsirneg"] = r["case_mye_pos"] / r["vsirneg_mye_pos"] if r.get("vsirneg_mye_pos") else np.nan
    rows.append(r)
    per_cell.append(pd.DataFrame({
        "patient": P, "group": (["VSIR+ myeloid"] * len(case) + ["non-myeloid"] * len(mnm)),
        "myeloid_cpm": np.r_[case.myeloid_cpm, mnm.myeloid_cpm],
        "cos_myeloid": np.r_[case.cos_myeloid, mnm.cos_myeloid],
        "p_mac": np.r_[case.p_Macrophage_cell, mnm.p_Macrophage_cell]}))
    print(f"[{P}] case {len(case)} | myeloid+ {r['case_mye_pos']:.3f} vs "
          f"non-myeloid {r['nonmye_mye_pos']:.3f} | cos {r['case_cos_mye']:.3f} vs "
          f"{r['nonmye_cos_mye']:.3f} | ref argmax myeloid {r['case_ref_mac']:.3f}",
          flush=True)
    del d, case, mnm, mvn, M, nm_raw_s; gc.collect()

R = pd.DataFrame(rows)
R.to_csv(S.TAB / "myeloid_identity.tsv", sep="\t", index=False)
pd.concat(per_cell).to_parquet(S.TAB / "myeloid_identity_cells.parquet", index=False)

pd.set_option("display.width", 250)
print("\n=== A. myeloid module: VSIR+ entities vs depth-matched NON-myeloid ===")
print(R[["patient", "case_n", "case_med_tx", "nonmye_med_tx", "case_mye_pos",
         "nonmye_mye_pos", "ratio_depthmatched", "p_mye_pos_vs_nonmye"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\n=== B. reference similarity and posterior ===")
print(R[["patient", "case_cos_mye", "nonmye_cos_mye", "p_cos_vs_nonmye",
         "case_ref_mac", "nonmye_ref_mac", "case_p_mac", "nonmye_p_mac"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\n=== C. depth treatments (ratio vs non-myeloid) and the state control ===")
print(R[["patient", "ratio_raw", "ratio_depthmatched", "ratio_rarefied",
         "ratio_vs_vsirneg"]].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print("\n=== D. off-lineage modules (mean CPM) ===")
print(R[["patient", "case_mye_cpm", "case_T_cpm", "case_B_cpm", "case_epithelial_cpm"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
for col, lab in [("ratio_depthmatched", "depth-matched"), ("ratio_rarefied", "rarefied"),
                 ("ratio_raw", "raw")]:
    v = R[col].dropna(); n = int((v > 1).sum())
    print(f"\n{lab}: myeloid module higher than non-myeloid controls in {n}/{len(v)} "
          f"patients, median ratio {v.median():.2f}, sign p={binomtest(n,len(v),0.5).pvalue:.4f}")
