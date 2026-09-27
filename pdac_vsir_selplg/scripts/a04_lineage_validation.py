#!/usr/bin/env python3
"""AUDIT 4 - are VISTA+ "TAMs" and PSGL-1+ "T cells" really TAMs and T cells?

The two populations that carry the centrepiece claim are defined by
(frozen label transfer) x (VSIR>0 / SELPLG>0). If the label transfer were
drifting, the "domain" could be an artefact of mislabelled entities. This
validates both populations with lineage markers that are INDEPENDENT of the
genes used to define them - VSIR and SELPLG are excluded throughout.

Counts are streamed straight out of the TRACER CSR h5ad one chunk at a time
(the 5 GB login cgroup cannot hold the dense matrix), and out of the stored
pre-arm CSR npz for the 10X arm.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

PDAC = Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
INTER = S.PDAC6 / "intermediate"
MYE = ["CD68", "CD163", "C1QA", "C1QB", "CSF1R", "AIF1", "MRC1", "MARCO", "ITGAM", "TREM2"]
TCELL = ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A", "LCK", "ZAP70", "THEMIS", "SKAP1"]
OTHER = ["EPCAM", "KRT19", "PTPRC"]
WANT = MYE + TCELL + OTHER
CHUNK = 8_000_000


def stream_cols(h5path, want_idx, n_cells):
    """Sum of counts for the wanted gene columns, per cell, streamed from CSR."""
    out = np.zeros((n_cells, len(want_idx)), np.int32)
    pos_of = {g: k for k, g in enumerate(want_idx)}
    wset = np.array(sorted(want_idx))
    with h5py.File(h5path, "r") as f:
        indptr = f["X/indptr"][:]
        nnz = f["X/data"].shape[0]
        start = 0
        while start < nnz:
            stop = min(start + CHUNK, nnz)
            idx = f["X/indices"][start:stop]
            dat = f["X/data"][start:stop]
            keep = np.isin(idx, wset)
            if keep.any():
                gi = idx[keep]; dv = dat[keep]
                rows = np.searchsorted(indptr, np.flatnonzero(keep) + start, side="right") - 1
                cols = np.array([pos_of[g] for g in gi])
                np.add.at(out, (rows, cols), dv)
            start = stop
    return out


rows = []
for P in S.PTS:
    genes = np.load(INTER / P / "genes.npy", allow_pickle=True)
    gpos = {g: i for i, g in enumerate(genes)}
    want_idx = [gpos[g] for g in WANT if g in gpos]
    want_nm = [g for g in WANT if g in gpos]

    # ---- post arms (TRACER h5ad) ----
    h5 = PDAC / P / "data/tracer_results/outputs/cell_by_gene_tracer.h5ad"
    meta = pd.read_parquet(INTER / P / "arm_final_meta.parquet")
    M = stream_cols(h5, want_idx, len(meta))
    post = pd.DataFrame(M, columns=want_nm)
    post["entity_id"] = meta.entity_id.values
    post["etype"] = meta.etype.values
    post["n_tx"] = meta.n_tx.values

    # ---- pre arm (stored CSR) ----
    Xp = sp.load_npz(INTER / P / "arm_pre_X.npz").tocsc()
    pmeta = pd.read_parquet(INTER / P / "arm_pre_meta.parquet")
    pre = pd.DataFrame({g: np.asarray(Xp[:, gpos[g]].todense()).ravel() for g in want_nm})
    pre["entity_id"] = pmeta.entity_id.values
    pre["etype"] = "cell"
    pre["n_tx"] = pmeta.n_tx.values
    del Xp

    for arm in S.ARMS:
        sp_df = S.load_arm(P, arm)[["entity_id", "cell_type", "VSIR", "SELPLG"]]
        src = pre if arm == "pre" else post
        d = sp_df.merge(src, on="entity_id", how="inner")
        if arm == "post_whole":
            d = d[d.etype == "cell"]
        cpm = lambda sub, gl: (sub[gl].sum(1).to_numpy() / np.maximum(sub.n_tx.to_numpy(), 1) * 1e4)
        mye_g = [g for g in MYE if g in want_nm]
        tc_g = [g for g in TCELL if g in want_nm]
        pops = {
            "VISTA+ TAM": d[(d.cell_type == "Macrophage cell") & (d.VSIR > 0)],
            "all TAM": d[d.cell_type == "Macrophage cell"],
            "PSGL1+ T": d[(d.cell_type == "T cell") & (d.SELPLG > 0)],
            "all T": d[d.cell_type == "T cell"],
            "tumour": d[d.cell_type.isin(S.TUMOR)],
            "all cells": d,
        }
        for nm, sub in pops.items():
            if len(sub) == 0:
                continue
            rows.append(dict(patient=P, arm=arm, population=nm, n=len(sub),
                             myeloid_cpm=float(np.median(cpm(sub, mye_g))),
                             tcell_cpm=float(np.median(cpm(sub, tc_g))),
                             myeloid_mean=float(np.mean(cpm(sub, mye_g))),
                             tcell_mean=float(np.mean(cpm(sub, tc_g))),
                             frac_myeloid_pos=float((sub[mye_g].sum(1) > 0).mean()),
                             frac_tcell_pos=float((sub[tc_g].sum(1) > 0).mean()),
                             median_n_tx=float(np.median(sub.n_tx))))
        # per-gene detail for the dot plot (post_all only)
        if arm == "post_all":
            for nm, sub in pops.items():
                if len(sub) == 0:
                    continue
                for g in want_nm:
                    rows.append(dict(patient=P, arm=arm, population=nm, n=len(sub), gene=g,
                                     frac_pos=float((sub[g] > 0).mean()),
                                     cpm=float(np.mean(sub[g].to_numpy() /
                                                       np.maximum(sub.n_tx.to_numpy(), 1) * 1e4))))
    del post, pre, M
    print(f"[{P}] done", flush=True)

R = pd.DataFrame(rows)
R[R.gene.isna()].drop(columns=["gene", "frac_pos", "cpm"]).to_csv(
    S.TAB / "lineage_validation_summary.tsv", sep="\t", index=False)
R[R.gene.notna()][["patient", "arm", "population", "n", "gene", "frac_pos", "cpm"]].to_csv(
    S.TAB / "lineage_validation_bygene.tsv", sep="\t", index=False)
q = R[R.gene.isna() & (R.arm == "post_all")]
print("\n=== independent lineage markers (VSIR/SELPLG excluded), post_all ===")
print(q[["patient", "population", "n", "myeloid_cpm", "tcell_cpm",
         "frac_myeloid_pos", "frac_tcell_pos", "median_n_tx"]]
      .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
