#!/usr/bin/env python3
"""AUDIT 7 - derive STABLE lineage modules from the Chijimatsu PDAC reference.

Replaces the hand-written myeloid list (CD68, CD163, C1QA/B, CSF1R, AIF1, MRC1,
MARCO, ITGAM, TREM2). That list is state-specific: MRC1/MARCO/TREM2 mark
particular TAM programmes, so a VISTA-high macrophage state can legitimately be
low for them, and the list then measures "which macrophage subset" rather than
"is this myeloid at all".

Selection criteria, per lineage, computed on the reference (30,000 cells,
`Cell_type`, 10 types):
  * present in the Visium HD gene universe (18,067 genes)
  * DETECTION >= 0.35 of reference cells of that lineage  -> broadly stable,
    not a subset marker
  * SPECIFICITY = mean CPM in lineage / max mean CPM across the other lineages,
    required >= 3
  * ranked by  log2(specificity) * detection
  * VSIR, SELPLG and every checkpoint gene in the Visium HD panel are excluded,
    so the module is independent of the checkpoint definitions

Also writes per-lineage reference centroids (CPM over the module union) for the
reference-similarity test in a08.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

REF = Path("/home/lyuan13/scr4_adeshpa6/TRACER/datasets/"
           "pancreas_cancer_snrna_chijimatsu_2022/pk_all_30k_stratified.h5ad")
UNIVERSE = S.PDAC6 / "intermediate/HC01/genes.npy"
CHUNK = 8_000_000
MIN_DET = 0.35
MIN_SPEC = 3.0
TOPN = 30
# excluded: the checkpoint definitions themselves and every checkpoint/hypoxia
# gene carried in the Visium HD feature table
EXCLUDE = {"VSIR", "SELPLG", "NECTIN2", "TIGIT", "PVR", "LGALS9", "HAVCR2",
           "CD274", "PDCD1", "CD5", "VSIG4", "IGSF11", "CD276", "LAG3", "CTLA4",
           "BTLA", "CD226", "TNFRSF14", "ADM", "CA9", "CA12", "HIF1A", "VEGFA",
           "NDRG1", "SLC2A1", "SLC16A1", "SLC16A3", "SLC9A1", "SLC4A4",
           "ENO1", "PGK1", "PDK1"}
LINEAGES = {
    "myeloid":    ["Macrophage cell"],
    "T":          ["T cell"],
    "B":          ["B cell"],
    "epithelial": ["Ductal cell type 1", "Ductal cell type 2", "Acinar cell"],
}

with h5py.File(REF, "r") as f:
    genes = np.array([g.decode() if isinstance(g, bytes) else g
                      for g in f["var/_index"][:]])
    ct_codes = f["obs/Cell_type/codes"][:]
    ct_cats = np.array([c.decode() if isinstance(c, bytes) else c
                        for c in f["obs/Cell_type/categories"][:]])
    indptr = f["X/indptr"][:]
    n_cells, n_genes = len(indptr) - 1, len(genes)
    types = ct_cats[ct_codes]
    uniq = list(ct_cats)
    tindex = {t: i for i, t in enumerate(uniq)}
    trow = np.array([tindex[t] for t in types])
    tot = np.zeros((len(uniq), n_genes))
    det = np.zeros((len(uniq), n_genes))
    depth = np.zeros(len(uniq))
    ncell = np.array([(trow == i).sum() for i in range(len(uniq))], float)
    nnz = f["X/data"].shape[0]
    s0 = 0
    while s0 < nnz:
        s1 = min(s0 + CHUNK, nnz)
        idx = f["X/indices"][s0:s1]
        dat = f["X/data"][s0:s1].astype(np.float64)
        rows = np.searchsorted(indptr, np.arange(s0, s1), side="right") - 1
        tr = trow[rows]
        np.add.at(tot, (tr, idx), dat)
        np.add.at(det, (tr, idx), 1.0)
        np.add.at(depth, tr, dat)
        s0 = s1
        print(f"  ref nnz {s1:,}/{nnz:,}", flush=True)

cpm = tot / np.maximum(depth[:, None], 1) * 1e6
detr = det / np.maximum(ncell[:, None], 1)
uni = set(np.load(UNIVERSE, allow_pickle=True).tolist())
in_uni = np.array([g in uni and g not in EXCLUDE for g in genes])
print(f"reference {n_cells:,} cells x {n_genes:,} genes; "
      f"{in_uni.sum():,} usable in the Visium HD universe", flush=True)

rows = []
modules = {}
for name, members in LINEAGES.items():
    mi = [tindex[m] for m in members]
    others = [i for i in range(len(uniq)) if i not in mi]
    own_cpm = cpm[mi].mean(0)
    own_det = detr[mi].mean(0)
    oth_cpm = cpm[others].max(0)
    spec = own_cpm / np.maximum(oth_cpm, 1e-9)
    ok = in_uni & (own_det >= MIN_DET) & (spec >= MIN_SPEC) & (own_cpm > 0)
    score = np.where(ok, np.log2(np.maximum(spec, 1e-9)) * own_det, -np.inf)
    top = np.argsort(score)[::-1][:TOPN]
    top = [i for i in top if np.isfinite(score[i])]
    modules[name] = [genes[i] for i in top]
    for i in top:
        rows.append(dict(lineage=name, gene=genes[i], detection=own_det[i],
                         cpm=own_cpm[i], max_other_cpm=oth_cpm[i],
                         specificity=spec[i], score=score[i]))
    print(f"[{name}] {len(modules[name])} genes: {', '.join(modules[name][:12])}"
          f"{' ...' if len(modules[name])>12 else ''}", flush=True)

M = pd.DataFrame(rows)
M.to_csv(S.TAB / "reference_lineage_modules.tsv", sep="\t", index=False)

# ---- how does the hand-written list compare? -------------------------------
MANUAL = ["CD68", "CD163", "C1QA", "C1QB", "CSF1R", "AIF1", "MRC1", "MARCO",
          "ITGAM", "TREM2"]
gi = {g: i for i, g in enumerate(genes)}
mi = [tindex["Macrophage cell"]]
others = [i for i in range(len(uniq)) if i not in mi]
own_det = detr[mi].mean(0); own_cpm = cpm[mi].mean(0); oth = cpm[others].max(0)
man = []
for g in MANUAL:
    if g not in gi:
        continue
    i = gi[g]
    man.append(dict(gene=g, detection=own_det[i], cpm=own_cpm[i],
                    specificity=own_cpm[i] / max(oth[i], 1e-9),
                    passes_stability=bool(own_det[i] >= MIN_DET),
                    in_reference_module=g in modules["myeloid"]))
MAN = pd.DataFrame(man)
MAN.to_csv(S.TAB / "manual_myeloid_list_audit.tsv", sep="\t", index=False)
print("\n=== audit of the hand-written myeloid list ===")
print(MAN.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print(f"  {int(MAN.passes_stability.sum())}/{len(MAN)} reach the {MIN_DET:.0%} "
      f"stability floor in reference macrophages")

# ---- reference centroids over the union of modules --------------------------
union = sorted({g for v in modules.values() for g in v})
ui = [gi[g] for g in union]
cent = pd.DataFrame(cpm[:, ui], index=uniq, columns=union)
cent.to_csv(S.TAB / "reference_centroids.tsv", sep="\t")
print(f"\nwrote reference_lineage_modules.tsv, reference_centroids.tsv "
      f"({len(union)} union genes)")
