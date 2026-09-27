#!/usr/bin/env python3
"""AUDIT 6 - depth-matched independent-lineage validation.

a04 showed VISTA+ TAMs carry ~4x the transcripts of the average macrophage
(median 122 vs 31). Raw CPM is not comparable across such different depths:
in a 20-transcript entity a SINGLE myeloid count is 500 CPM, so shallow cells
inflate mean CPM. Every comparison here is therefore against a depth-matched
control drawn from the same cell type in the same patient and arm
(1:5 nearest-neighbour matching on log10 n_tx, without replacement).

Markers are independent of the definitions: VSIR and SELPLG are excluded.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.stats import mannwhitneyu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

PDAC = Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
INTER = S.PDAC6 / "intermediate"
MYE = ["CD68", "CD163", "C1QA", "C1QB", "CSF1R", "AIF1", "MRC1", "MARCO", "ITGAM", "TREM2"]
TC = ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A", "LCK", "ZAP70", "THEMIS", "SKAP1"]
CHUNK = 8_000_000
rng = np.random.default_rng(3)


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
    """1:k nearest-neighbour match on log10 depth, without replacement."""
    if len(ctrl_tx) == 0 or len(case_tx) == 0:
        return np.array([], int)
    a = np.log10(np.maximum(case_tx, 1)); b = np.log10(np.maximum(ctrl_tx, 1))
    order = np.argsort(b); bs = b[order]
    used = np.zeros(len(b), bool); picked = []
    for v in a:
        j = np.searchsorted(bs, v)
        for _ in range(k):
            lo, hi = j - 1, j
            best = -1; bd = np.inf
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


rows, cells = [], []
for P in S.PTS:
    genes = np.load(INTER / P / "genes.npy", allow_pickle=True)
    gp = {g: i for i, g in enumerate(genes)}
    wn = [g for g in MYE + TC if g in gp]
    M = stream(PDAC / P / "data/tracer_results/outputs/cell_by_gene_tracer.h5ad",
               [gp[g] for g in wn], len(pd.read_parquet(INTER / P / "arm_final_meta.parquet")))
    meta = pd.read_parquet(INTER / P / "arm_final_meta.parquet")
    post = pd.DataFrame({"entity_id": meta.entity_id.values, "etype": meta.etype.values,
                         "n_tx": meta.n_tx.values})
    post["mye"] = M[:, [wn.index(g) for g in MYE if g in wn]].sum(1)
    post["tc"] = M[:, [wn.index(g) for g in TC if g in wn]].sum(1)
    Xp = sp.load_npz(INTER / P / "arm_pre_X.npz").tocsc()
    pm = pd.read_parquet(INTER / P / "arm_pre_meta.parquet")
    pre = pd.DataFrame({"entity_id": pm.entity_id.values, "etype": "cell", "n_tx": pm.n_tx.values})
    pre["mye"] = np.asarray(Xp[:, [gp[g] for g in MYE if g in gp]].sum(1)).ravel()
    pre["tc"] = np.asarray(Xp[:, [gp[g] for g in TC if g in gp]].sum(1)).ravel()
    del Xp, M

    for arm in S.ARMS:
        sd = S.load_arm(P, arm)[["entity_id", "cell_type", "VSIR", "SELPLG"]]
        d = sd.merge(pre if arm == "pre" else post, on="entity_id", how="inner")
        if arm == "post_whole":
            d = d[d.etype == "cell"]
        d = d.reset_index(drop=True)
        for pop, ct, gene, tag in [("VISTA+ TAM", "Macrophage cell", "VSIR", "mye"),
                                   ("PSGL1+ T", "T cell", "SELPLG", "tc")]:
            sub = d[d.cell_type == ct]
            case = sub[sub[gene] > 0]; ctrl = sub[sub[gene] == 0]
            if len(case) < 5 or len(ctrl) < 5:
                continue
            mi = match_depth(case.n_tx.values, ctrl.n_tx.values, k=5)
            mc = ctrl.iloc[mi]
            cpm = lambda s, c: s[c].to_numpy() / np.maximum(s.n_tx.to_numpy(), 1) * 1e4
            rec = dict(patient=P, arm=arm, population=pop, cell_type=ct,
                       n_case=len(case), n_ctrl_matched=len(mc),
                       med_tx_case=float(np.median(case.n_tx)),
                       med_tx_ctrl=float(np.median(mc.n_tx)),
                       own_cpm_case=float(np.mean(cpm(case, tag))),
                       own_cpm_ctrl=float(np.mean(cpm(mc, tag))),
                       off_cpm_case=float(np.mean(cpm(case, "tc" if tag == "mye" else "mye"))),
                       own_frac_case=float((case[tag] > 0).mean()),
                       own_frac_ctrl=float((mc[tag] > 0).mean()))
            rec["own_frac_ratio"] = rec["own_frac_case"] / rec["own_frac_ctrl"] if rec["own_frac_ctrl"] > 0 else np.nan
            if len(case) > 4 and len(mc) > 4:
                rec["p_own_frac"] = float(mannwhitneyu(case[tag] > 0, mc[tag] > 0,
                                                       alternative="greater").pvalue)
            rows.append(rec)
        if arm == "post_all":
            keep = d[d.cell_type.isin(["Macrophage cell", "T cell"])].copy()
            keep["patient"] = P
            cells.append(keep[["patient", "cell_type", "etype", "n_tx", "mye", "tc",
                               "VSIR", "SELPLG"]])
    del post, pre
    print(f"[{P}] done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(S.TAB / "lineage_depth_matched.tsv", sep="\t", index=False)
pd.concat(cells).to_parquet(S.TAB / "lineage_cells_post_all.parquet", index=False)
pd.set_option("display.width", 220)
print("\n=== depth-matched independent-lineage validation (post_all) ===")
print(R[R.arm == "post_all"][["patient", "population", "n_case", "n_ctrl_matched",
                              "med_tx_case", "med_tx_ctrl", "own_frac_case",
                              "own_frac_ctrl", "own_frac_ratio", "p_own_frac",
                              "off_cpm_case"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
