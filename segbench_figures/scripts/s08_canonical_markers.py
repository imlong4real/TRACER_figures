"""Marker specificity on a fixed canonical lineage-marker panel.

Replaces the evaluator's data-driven marker set.  The evaluator
(``segbench.evaluate.marker_specificity``) picked, per cell type, the top 30
genes by fold change in the held-out scRNA reference pseudobulk.  That set is
chosen from the same scRNA distribution SPLIT purifies toward, so it can favour
methods that pull profiles toward the reference.  Here the genes are fixed in
advance from textbook lineage markers and never looked up in the data.

Estimand, per method and setting:
  for each scored cell type t (>= 50 held-out reference cells, the evaluator's
  universe; >= 5 spatial cells labelled t by RCTD; >= 2 panel markers present):
    per marker g:  log2( (mean CPM_g in t + 1) / (mean over other types of their
                   mean CPM_g + 1) )          # other types weighted equally
    type score  =  median over t's markers
  method score  =  median over types, reported only if >= 2 types are scored
                   (MERFISH's 241-gene panel carries no canonical enterocyte or
                   goblet marker, so it is NA rather than a one-type score)
TRACER is scored whole, partial and pooled; pooled is the comparison that does
not benefit from TRACER routing ambiguous transcripts into partials.
Counts and labels are the ones the evaluator used: each method's
``rctd_input.h5ad`` and its per-cell RCTD ``dominant_celltype``.  A true log2 is
used; the evaluator's "log2FC" is a natural-log difference of log1p(CPM).

    s08_canonical_markers.py <setting>
"""
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from settings import OBJ, RUNS_DIR, SETTINGS
from labels import harmonize

# Harmonized type -> canonical markers.  Human symbols; the mouse ileum panel is
# given in mouse symbols.  Each list is disjoint from every other type that can
# co-occur in the same dataset, so a marker never counts for two lineages.
# Each list carries standard alternatives because some platform panels omit
# the usual first choice (Xenium 5K has no KRT8/18/19 or COL1A1/2); the lists
# were fixed on panel coverage alone, before any method was scored, and are
# identical for every platform.  A type is scored only with >= 2 markers
# present in that method's own output - Segger's gene filter removes most of
# them, which is reported as NA rather than zero.
HUMAN = {
    "Tumor":         ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT5", "KRT17",
                      "CLDN4", "CLDN7", "ELF3"],
    "Ciliated":      ["FOXJ1", "CAPS", "TPPP3", "PIFO"],
    "T/NK":          ["CD3E", "CD3D", "CD2", "NKG7"],
    "B":             ["MS4A1", "CD19", "BANK1", "CD22"],
    "Plasma":        ["MZB1", "JCHAIN", "TNFRSF17"],
    "Lymphoid":      ["CD3E", "CD2", "MS4A1"],
    "Myeloid":       ["CD68", "CD14", "CD163", "LYZ", "CSF1R"],
    "Neutrophil":    ["CSF3R", "FCGR3B", "CXCR2"],
    "Mast":          ["TPSAB1", "CPA3", "KIT", "MS4A2", "HDC"],
    "Fibroblast":    ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Stroma":        ["COL1A1", "DCN", "PDGFRB", "ACTA2"],
    "Smooth muscle": ["ACTA2", "MYH11", "DES", "CNN1", "TAGLN"],
    "Endothelial":   ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "PT":            ["LRP2", "CUBN", "SLC34A1"],
    "TAL":           ["UMOD", "SLC12A1"],
    "PC":            ["AQP2", "SCNN1G"],
    "IC":            ["ATP6V1B1", "FOXI1"],
    "Podocyte":      ["NPHS1", "NPHS2", "PODXL"],
}
MOUSE = {
    "Stem/TA":    ["Lgr5", "Olfm4", "Ascl2", "Mki67"],
    "Enterocyte": ["Fabp1", "Fabp2", "Alpi", "Apoa4"],
    "Goblet":     ["Muc2", "Clca1", "Tff3", "Spink4"],
}
KEY = {"baysor": "Baysor", "proseg": "ProSeg", "split": "SPLIT", "segger": "Segger",
       "celladmix": "CellAdmix", "bin2cell": "Bin2Cell", "tracer_noseg": "TRACER no-seg"}


def score(X, genes, labels, types, panel):
    """labels aligned to X rows (harmonized type or None)."""
    gidx = {g: i for i, g in enumerate(genes)}
    X = sp.csr_matrix(X, dtype=np.float64)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1.0
    cpm = sp.diags(1e4 / lib) @ X
    means = {}
    for t in types:
        idx = np.flatnonzero(labels == t)
        means[t] = np.asarray(cpm[idx].mean(0)).ravel()
    per_type, used = {}, {}
    for t in types:
        mk = [g for g in panel.get(t, []) if g in gidx]
        rest = [u for u in types if u != t]
        if len(mk) < 2 or not rest:
            continue
        cols = [gidx[g] for g in mk]
        inn = means[t][cols]
        out = np.mean([means[u][cols] for u in rest], axis=0)
        per_type[t] = float(np.median(np.log2((inn + 1) / (out + 1))))
        used[t] = mk
    return per_type, used


def main(setting):
    ds = SETTINGS[setting][0]
    panel = MOUSE if ds == "merfish_mouse_ileum" else HUMAN
    bio = pd.read_csv(OBJ / f"tables/bio/bio__{setting}.tsv", sep="\t")
    kept = {harmonize(setting, t) for t in bio.kept_celltypes.iloc[0].split(",")}
    inputs = [("Original", OBJ / "original" / setting / "original_rctd_input.h5ad",
               OBJ / "original" / setting / "rctd/rctd_cell_assignments_post.tsv", None)]
    for mdir in sorted((RUNS_DIR / setting / "methods").iterdir()):
        r = mdir / "rctd"
        if not (r / "rctd_input.h5ad").exists():
            continue
        if mdir.name == "tracer_seg":
            for st, lab in (("whole", "TRACER whole"), ("partial", "TRACER partial"),
                            (None, "TRACER pooled")):
                inputs.append((lab, r / "rctd_input.h5ad", r / "rctd_cell_assignments_post.tsv", st))
        else:
            inputs.append((KEY[mdir.name], r / "rctd_input.h5ad", r / "rctd_cell_assignments_post.tsv", None))
    rows, used_all = [], {}
    for method, h5, tsv, stratum in inputs:
        a = ad.read_h5ad(h5)
        if stratum is not None:
            a = a[a.obs["whole_partial_status"].astype(str).to_numpy() == stratum].copy()
        X = a.layers["counts"] if "counts" in a.layers else a.X
        lab = pd.read_csv(tsv, sep="\t", usecols=["cell_id", "dominant_celltype"])
        lab["cell_id"] = lab["cell_id"].astype(str)
        lab = lab.set_index("cell_id")["dominant_celltype"].reindex(a.obs_names.astype(str))
        lab = np.array([harmonize(setting, x) if isinstance(x, str) else None for x in lab],
                       dtype=object)
        types = [t for t in sorted(kept) if (lab == t).sum() >= 5]
        per_type, used = score(X, list(map(str, a.var_names)), lab, types, panel)
        used_all.update(used)
        rows.append({"setting": setting, "method": method,
                     "canonical_marker_log2fc": float(np.median(list(per_type.values())))
                     if len(per_type) >= 2 else np.nan,
                     "n_types_scored": len(per_type),
                     **{f"type__{t}": v for t, v in per_type.items()}})
        del a, X
    out = pd.DataFrame(rows)
    d = OBJ / "tables/canonical"; d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"canonical__{setting}.tsv", sep="\t", index=False)
    (d / f"markers_used__{setting}.json").write_text(json.dumps(used_all, indent=1))
    print(out[["method", "canonical_marker_log2fc", "n_types_scored"]].round(3).to_string(index=False))
    print("markers present:", {t: len(v) for t, v in used_all.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
