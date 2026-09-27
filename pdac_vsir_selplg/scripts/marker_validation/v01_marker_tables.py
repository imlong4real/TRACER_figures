#!/usr/bin/env python3
"""V01 - canonical-marker expression tables for label-transfer validation.

Builds per (modality, cell_type, gene) mean expression and detection fraction for
an IDENTICAL canonical marker set across four columns:

  scRNA_cPMI     Chijimatsu PDAC atlas, study CA001063 EXCLUDED - the reference
                 actually used for the cPMI calculation. PRIMARY.
  scRNA_full     the same atlas with CA001063 restored - sensitivity only, and
                 never substituted for the cPMI reference
  pre            VisiumHD, 10X vendor segmentation
  post_whole     VisiumHD, post-TRACER whole cells
  post_partial   VisiumHD, post-TRACER PARTIAL entities only
  post_all       whole + partial, DERIVED descriptively from the two separate
                 populations (see below) - a summary column, not a re-analysis

Whole and partial entities are stored separately and are never pooled in the
underlying data, and are never filtered on marker coherence. The combined column
is derived exactly: mean expression and detection fraction are both cell-weighted
means, so pooling per patient with that patient's own cell counts reproduces the
pooled statistic without re-reading any matrix.

Expression = mean of log1p(CP10K) over cells of that type.
Detection   = fraction of cells of that type with a non-zero raw count.
VisiumHD values are computed per patient and then averaged with equal weight, so
no single patient dominates.
"""
from __future__ import annotations
import sys, gc, json
from pathlib import Path
import numpy as np, pandas as pd, h5py, scipy.sparse as sp

ROOT = Path("/scratch4/adeshpa6/tracer_campaign/final/panels/pdac_supp/marker_validation")
TAB = ROOT / "tables"
PDAC6 = Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
INTER = PDAC6 / "intermediate"
SRC = Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
REF = Path("/home/lyuan13/scr4_adeshpa6/TRACER/datasets/"
           "pancreas_cancer_snrna_chijimatsu_2022/pk_all_30k_stratified.h5ad")
PTS = ["HC01", "HC03", "HC04", "HC05", "HC07", "HC08"]
CHUNK = 8_000_000
EXCLUDE_STUDY = "CA001063"   # matches the cPMI scRNA reference

# ---- canonical markers; KRT18 dropped so the set is identical in every modality
MARKERS = {
    "Ductal cell type 1": ["CFTR", "MMP7", "TFF1", "TFF2", "AQP1", "SLC4A4", "ONECUT2"],
    "Ductal cell type 2": ["KRT19", "KRT7", "TSPAN8", "SOX9", "MUC1", "CEACAM6", "S100P"],
    "Acinar cell":        ["PRSS1", "CTRB1", "CTRB2", "CPA1", "CELA3A", "REG1A", "PNLIP", "CPB1"],
    "Endocrine cell":     ["CHGA", "CHGB", "INS", "GCG", "SST", "PPY", "SCG2", "PCSK1N"],
    "T cell":             ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A", "CD247", "THEMIS"],
    "B cell":             ["CD79A", "MS4A1", "CD79B", "BANK1", "IGHM", "VPREB3", "BLK"],
    "Macrophage cell":    ["CD68", "CD14", "AIF1", "LYZ", "C1QA", "C1QB", "TYROBP", "ITGAM", "FCER1G"],
    "Endothelial cell":   ["PECAM1", "VWF", "CDH5", "PLVAP", "CLDN5", "EGFL7", "RAMP2"],
    "Fibroblast cell":    ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FN1", "COL3A1"],
    "Stellate cell":      ["RGS5", "ACTA2", "PDGFRB", "NOTCH3", "MYL9", "DES", "NDUFA4L2"],
}
CT_ORDER = list(MARKERS)
GENE_ORDER = [g for t in CT_ORDER for g in MARKERS[t]]


DEPTH = []   # (modality, patient, cell_type, n_tx) for the depth report


def summarise(counts, ntx, labels, genes, types):
    """mean log1p(CP10K) and detection fraction per (type, gene)."""
    cpm = counts / np.maximum(ntx, 1)[:, None] * 1e4
    lg = np.log1p(cpm)
    rows = []
    for t in types:
        m = labels == t
        if m.sum() == 0:
            continue
        rows.append(pd.DataFrame({"cell_type": t, "gene": genes,
                                  "expr": lg[m].mean(0),
                                  "detect": (counts[m] > 0).mean(0),
                                  "n_cells": int(m.sum()),
                                  "median_n_tx": float(np.median(ntx[m]))}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stream_h5(path, gidx, ncells, xkey="X"):
    out = np.zeros((ncells, len(gidx)), np.float32)
    pos = {g: k for k, g in enumerate(gidx)}
    ws = np.array(sorted(gidx))
    with h5py.File(path, "r") as f:
        ip = f[f"{xkey}/indptr"][:]
        nnz = f[f"{xkey}/data"].shape[0]
        s0 = 0
        while s0 < nnz:
            s1 = min(s0 + CHUNK, nnz)
            ii = f[f"{xkey}/indices"][s0:s1]; dd = f[f"{xkey}/data"][s0:s1]
            k = np.isin(ii, ws)
            if k.any():
                r = np.searchsorted(ip, np.flatnonzero(k) + s0, side="right") - 1
                c = np.array([pos[g] for g in ii[k]])
                np.add.at(out, (r, c), dd[k])
            s0 = s1
    return out


frames = []

# ---------------------------------------------------------------- scRNA ------
with h5py.File(REF, "r") as f:
    rgenes = np.array([g.decode() if isinstance(g, bytes) else g
                       for g in f["var/_index"][:]])
    codes = f["obs/Cell_type/codes"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else c
                     for c in f["obs/Cell_type/categories"][:]])
    pcats = np.array([c.decode() if isinstance(c, bytes) else c
                      for c in f["obs/Project/categories"][:]])
    study = pcats[f["obs/Project/codes"][:]]
    keep_ref = study != EXCLUDE_STUDY
    n_ref = len(codes)
    ip = f["X/indptr"][:]
    tot = np.zeros(n_ref)
    s0 = 0; nnz = f["X/data"].shape[0]
    while s0 < nnz:
        s1 = min(s0 + CHUNK, nnz)
        dd = f["X/data"][s0:s1]
        r = np.searchsorted(ip, np.arange(s0, s1), side="right") - 1
        np.add.at(tot, r, dd)
        s0 = s1
gmap = {g: i for i, g in enumerate(rgenes)}
gi = [gmap[g] for g in GENE_ORDER]
Xr = stream_h5(REF, gi, n_ref)
lab = cats[codes]
print(f"[scRNA] study {EXCLUDE_STUDY}: {int((~keep_ref).sum()):,} cells; "
      f"{int(keep_ref.sum()):,} of {n_ref:,} retained for the cPMI reference", flush=True)
for mod, msk in (("scRNA_cPMI", keep_ref), ("scRNA_full", np.ones(n_ref, bool))):
    d = summarise(Xr[msk], tot[msk], lab[msk], np.array(GENE_ORDER), CT_ORDER)
    d["modality"] = mod; d["patient"] = "reference"
    frames.append(d)
    DEPTH.append(pd.DataFrame({"modality": mod, "patient": "reference",
                               "cell_type": lab[msk], "n_tx": tot[msk]}))
    print(f"[{mod}] {int(msk.sum()):,} cells, {d.cell_type.nunique()} types", flush=True)
# atlas composition, with and without the excluded study
comp = pd.DataFrame({"cell_type": CT_ORDER})
comp["n_cPMI"] = [int(((lab == t) & keep_ref).sum()) for t in CT_ORDER]
comp["n_full"] = [int((lab == t).sum()) for t in CT_ORDER]
comp["n_CA001063"] = comp.n_full - comp.n_cPMI
comp["frac_cPMI"] = comp.n_cPMI / comp.n_cPMI.sum()
comp["frac_full"] = comp.n_full / comp.n_full.sum()
comp["frac_from_CA001063"] = comp.n_CA001063 / comp.n_full
comp.to_csv(TAB / "atlas_composition.tsv", sep="\t", index=False)
print("\n=== atlas composition ===")
print(comp.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
del Xr; gc.collect()

# ------------------------------------------------------------- VisiumHD ------
for P in PTS:
    genes = np.array([str(g) for g in np.load(INTER / P / "genes.npy", allow_pickle=True)])
    gm = {g: i for i, g in enumerate(genes)}
    gi = [gm[g] for g in GENE_ORDER]

    lp = pd.read_parquet(INTER / P / "labels_pre.parquet",
                         columns=["matrix_index", "cell_type", "n_tx"])
    Xp = sp.load_npz(INTER / P / "arm_pre_X.npz").tocsc()
    Cp = np.asarray(Xp[:, gi].todense(), dtype=np.float32)
    del Xp; gc.collect()
    Cp = Cp[lp.matrix_index.to_numpy()]
    d = summarise(Cp, lp.n_tx.to_numpy(float), lp.cell_type.to_numpy(),
                  np.array(GENE_ORDER), CT_ORDER)
    d["modality"] = "pre"; d["patient"] = P
    frames.append(d)
    DEPTH.append(pd.DataFrame({"modality": "pre", "patient": P,
                               "cell_type": lp.cell_type.to_numpy(),
                               "n_tx": lp.n_tx.to_numpy(float)}))
    del Cp; gc.collect()

    meta = pd.read_parquet(INTER / P / "arm_final_meta.parquet")
    Cq = stream_h5(SRC / P / "data/tracer_results/outputs/cell_by_gene_tracer.h5ad",
                   gi, len(meta))
    for arm, files in (("post_whole", ["labels_post_whole.parquet"]),
                       ("post_partial", ["labels_post_partial.parquet"])):
        L = pd.concat([pd.read_parquet(INTER / P / f,
                                       columns=["matrix_index", "cell_type", "n_tx"])
                       for f in files], ignore_index=True)
        sub = Cq[L.matrix_index.to_numpy()]
        d = summarise(sub, L.n_tx.to_numpy(float), L.cell_type.to_numpy(),
                      np.array(GENE_ORDER), CT_ORDER)
        d["modality"] = arm; d["patient"] = P
        frames.append(d)
        DEPTH.append(pd.DataFrame({"modality": arm, "patient": P,
                                   "cell_type": L.cell_type.to_numpy(),
                                   "n_tx": L.n_tx.to_numpy(float)}))
        del sub; gc.collect()
    del Cq, meta; gc.collect()
    print(f"[{P}] pre + post_whole + post_partial done", flush=True)

R = pd.concat(frames, ignore_index=True)

# post_all: cell-weighted pooling of the two SEPARATE populations, per patient.
# expr and detect are both per-cell means, so this is exact, not an approximation.
w = R[R.modality.isin(["post_whole", "post_partial"])]
key = ["patient", "cell_type", "gene"]
comb = (w.assign(we=w.expr * w.n_cells, wd=w.detect * w.n_cells)
         .groupby(key)
         .agg(we=("we", "sum"), wd=("wd", "sum"), n_cells=("n_cells", "sum"),
              median_n_tx=("median_n_tx", "mean"), k=("modality", "nunique"))
         .reset_index())
comb = comb[comb.k == 2]                      # both populations present
comb["expr"] = comb.we / comb.n_cells
comb["detect"] = comb.wd / comb.n_cells
comb["modality"] = "post_all"
R = pd.concat([R, comb[["cell_type", "gene", "expr", "detect", "n_cells",
                        "median_n_tx", "modality", "patient"]]], ignore_index=True)
R.to_csv(TAB / "marker_expression_by_patient.tsv", sep="\t", index=False)

# equal-weight average across patients for the VisiumHD modalities
agg = (R.groupby(["modality", "cell_type", "gene"])
        .agg(expr=("expr", "mean"), detect=("detect", "mean"),
             n_cells=("n_cells", "sum"), n_patients=("patient", "nunique"))
        .reset_index())

# ---- cell counts and POOLED median depth per group, for the sparsity caption
DP = pd.concat(DEPTH, ignore_index=True)
DP = pd.concat([DP, DP[DP.modality.isin(["post_whole", "post_partial"])]
                     .assign(modality="post_all")], ignore_index=True)
grp = (DP.groupby(["modality", "cell_type"])
         .agg(n_cells=("n_tx", "size"), median_n_tx=("n_tx", "median"),
              q25_n_tx=("n_tx", lambda v: float(np.percentile(v, 25))),
              q75_n_tx=("n_tx", lambda v: float(np.percentile(v, 75))))
         .reset_index())
grp.to_csv(TAB / "group_depth_summary.tsv", sep="\t", index=False)
tot_grp = (DP.groupby("modality")
             .agg(n_cells=("n_tx", "size"), median_n_tx=("n_tx", "median"))
             .reset_index())
tot_grp.to_csv(TAB / "group_totals.tsv", sep="\t", index=False)
print("\n=== cells and median transcript depth per group ===")
print(tot_grp.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
agg.to_csv(TAB / "marker_expression_summary.tsv", sep="\t", index=False)
json.dump({"markers": MARKERS, "ct_order": CT_ORDER, "gene_order": GENE_ORDER,
           "n_genes": len(GENE_ORDER), "dropped": ["KRT18 (absent from VisiumHD panel)"],
           "excluded_study": EXCLUDE_STUDY,
           "primary_reference": "scRNA_cPMI",
           "modalities": ["scRNA_cPMI", "scRNA_full", "pre", "post_whole",
                          "post_partial", "post_all"]},
          open(TAB / "marker_definition.json", "w"), indent=2)
print(f"\nwrote {len(agg):,} summary rows over {agg.modality.nunique()} modalities, "
      f"{len(GENE_ORDER)} genes")
print(agg.groupby("modality").n_cells.sum().to_string())
