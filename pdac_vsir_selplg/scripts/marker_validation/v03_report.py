#!/usr/bin/env python3
"""V03 - written summary of the marker validation (regenerates the report .md)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/scratch4/adeshpa6/tracer_campaign/final/panels/pdac_supp/marker_validation")
OUT = R.parent
C = pd.read_csv(R / "tables/label_coherence.tsv", sep="\t")
G = pd.read_csv(R / "tables/group_totals.tsv", sep="\t").set_index("modality")
COMP = pd.read_csv(R / "tables/atlas_composition.tsv", sep="\t").set_index("cell_type")
M = json.load(open(R / "tables/marker_definition.json"))
CT = M["ct_order"]
S = {"Ductal cell type 1": "Ductal 1", "Ductal cell type 2": "Ductal 2",
     "Acinar cell": "Acinar", "Endocrine cell": "Endocrine", "T cell": "T",
     "B cell": "B", "Macrophage cell": "Macrophage", "Endothelial cell": "Endothelial",
     "Fibroblast cell": "Fibroblast", "Stellate cell": "Stellate"}
ML = {"scRNA_cPMI": "scRNA reference (cPMI)", "scRNA_full": "scRNA full atlas",
      "pre": "pre-TRACER (10X)", "post_whole": "post-TRACER whole",
      "post_partial": "post-TRACER partial", "post_all": "post-TRACER whole+partial"}
MAIN = ["scRNA_cPMI", "pre", "post_whole", "post_partial", "post_all"]
ATL = ["scRNA_cPMI", "scRNA_full"]


def pv(col, mods):
    return C.pivot(index="cell_type", columns="modality", values=col)[mods].reindex(CT)


spec, det, nc, flag = (pv(c, MAIN + ["scRNA_full"]) for c in
                       ("specificity", "mean_detection", "n_cells", "flag"))
L = ["# Marker validation of the PDAC label transfer (v3)\n",
     "Panels `S11`-`S15` in `panels/pdac_supp/`; tables and scripts in "
     "`panels/pdac_supp/marker_validation/`. Earlier versions are in "
     "`archive_20260924_pre_CA001063_exclusion/` (all 30,000 atlas cells, 4 columns) "
     "and `archive_20260924b_3col/` (cPMI reference, 3 columns).\n",
     "## Design\n",
     "Five comparison columns, identical marker order, cell-type order and colour "
     "limits throughout:\n",
     "| column | contents |", "|---|---|",
     "| scRNA reference (cPMI) | Chijimatsu atlas, study CA001063 excluded - **the "
     "reference actually used for cPMI, and the primary analysis here** |",
     "| pre-TRACER | VisiumHD, 10X vendor segmentation |",
     "| post-TRACER whole | whole cells only |",
     "| post-TRACER partial | partial entities only |",
     "| post-TRACER whole+partial | **descriptive** cell-weighted pooling of the two |\n",
     "Whole and partial are stored as separate populations everywhere in the source "
     "tables and are never filtered on marker coherence. Because mean expression and "
     "detection fraction are both per-cell means, the combined column is derived "
     "exactly by pooling with each patient's own cell counts - it is a summary of the "
     "two populations, not a separate re-analysis.\n",
     "The full atlas with CA001063 restored is reported as a **sensitivity analysis "
     "only** (S15). The cPMI reference is unchanged.\n",
     "## Group sizes and sequencing depth\n",
     "| group | cells | median transcripts / cell |", "|---|---:|---:|"]
for m in ["scRNA_cPMI", "scRNA_full", "pre", "post_whole", "post_partial", "post_all"]:
    L.append(f"| {ML[m]} | {int(G.loc[m,'n_cells']):,} | {G.loc[m,'median_n_tx']:,.0f} |")
L += ["",
      f"Partial entities carry **{G.loc['post_partial','median_n_tx']:,.0f}** transcripts "
      f"at the median against **{G.loc['post_whole','median_n_tx']:,.0f}** for whole cells "
      f"({G.loc['post_whole','median_n_tx']/G.loc['post_partial','median_n_tx']:.1f}x) and "
      f"**{G.loc['scRNA_cPMI','median_n_tx']:,.0f}** for the atlas "
      f"({G.loc['scRNA_cPMI','median_n_tx']/G.loc['post_partial','median_n_tx']:.0f}x). "
      "Partial depth is uniform across cell types (median 12-20 tx, "
      "`group_depth_summary.tsv`), so detection differences between cell types within "
      "the partial column are not a depth artefact.\n",
      "## Verdict\n",
      "**The transferred labels are biologically coherent in every group, including "
      "partial entities.** Mean marker specificity (own - best-other z) over the 10 types:\n",
      "| " + " | ".join(ML[m] for m in MAIN) + " |", "|" + "---:|" * len(MAIN),
      "| " + " | ".join(f"{spec[m].mean():.2f}" for m in MAIN) + " |\n",
      "Partial entities score **highest of any group** on marker specificity despite "
      "being the sparsest. Sparsity costs them detection, not identity.\n",
      "## 1. Atlas sensitivity: restoring CA001063\n",
      "CA001063 is effectively a ductal-1 / endothelial study. Its cells are "
      "**not** spread evenly over the taxonomy:\n",
      "| cell type | cells (cPMI ref) | cells (full atlas) | % of type from CA001063 | "
      "specificity cPMI | specificity full | Δ |",
      "|---|---:|---:|---:|---:|---:|---:|"]
for t in CT:
    L.append(f"| {S[t]} | {int(COMP.loc[t,'n_cPMI']):,} | {int(COMP.loc[t,'n_full']):,} | "
             f"{COMP.loc[t,'frac_from_CA001063']*100:.1f}% | {spec.loc[t,'scRNA_cPMI']:.2f} | "
             f"{spec.loc[t,'scRNA_full']:.2f} | {spec.loc[t,'scRNA_full']-spec.loc[t,'scRNA_cPMI']:+.2f} |")
L += ["",
      "**Representation changes a lot; marker specificity mostly does not.** "
      f"{COMP.loc['Ductal cell type 1','frac_from_CA001063']*100:.0f}% of ductal-1 and "
      f"{COMP.loc['Endothelial cell','frac_from_CA001063']*100:.0f}% of endothelial atlas "
      "cells come from CA001063, and restoring it raises ductal-1 from "
      f"{COMP.loc['Ductal cell type 1','frac_cPMI']*100:.1f}% to "
      f"{COMP.loc['Ductal cell type 1','frac_full']*100:.1f}% of the atlas and endothelial "
      f"from {COMP.loc['Endothelial cell','frac_cPMI']*100:.1f}% to "
      f"{COMP.loc['Endothelial cell','frac_full']*100:.1f}%. Yet **endothelial specificity "
      f"barely moves** ({spec.loc['Endothelial cell','scRNA_cPMI']:.2f} -> "
      f"{spec.loc['Endothelial cell','scRNA_full']:.2f}) - a two-thirds change in "
      "composition with essentially no effect on how cleanly endothelial markers separate.\n",
      "**The ductal subtypes are the exception, and neither atlas version resolves them.** "
      f"Restoring CA001063 moves specificity between the two subtypes - ductal-1 "
      f"{spec.loc['Ductal cell type 1','scRNA_cPMI']:.2f} -> "
      f"{spec.loc['Ductal cell type 1','scRNA_full']:.2f}, ductal-2 "
      f"{spec.loc['Ductal cell type 2','scRNA_cPMI']:.2f} -> "
      f"{spec.loc['Ductal cell type 2','scRNA_full']:.2f} - but **both stay below the 1.5 "
      "ambiguity threshold in both versions**, and each remains the other's most "
      "confusable partner. Which subtype looks cleaner is an artefact of which study "
      "dominates the atlas, not a biological fact.\n",
      "The cPMI reference is unchanged by this analysis and remains the primary reference.\n",
      "## 2. Per-cell-type marker specificity (own - best other, z)\n",
      "| cell type | " + " | ".join(ML[m] for m in MAIN) + " | most confusable (partials) |",
      "|---|" + "---:|" * len(MAIN) + "---|"]
for t in CT:
    conf = C[(C.modality == "post_partial") & (C.cell_type == t)].most_confusable.iloc[0]
    L.append(f"| {S[t]} | " + " | ".join(f"{spec.loc[t,m]:.2f}" for m in MAIN) +
             f" | {S.get(conf, conf)} |")
L += ["", "## 3. Mean detection fraction of own markers\n",
      "| cell type | " + " | ".join(ML[m] for m in MAIN) + " |",
      "|---|" + "---:|" * len(MAIN)]
for t in CT:
    L.append(f"| {S[t]} | " + " | ".join(f"{det.loc[t,m]:.3f}" for m in MAIN) + " |")
L += ["", "## 4. Cells per type per group\n",
      "| cell type | " + " | ".join(ML[m] for m in MAIN) + " |",
      "|---|" + "---:|" * len(MAIN)]
for t in CT:
    L.append(f"| {S[t]} | " + " | ".join(f"{int(nc.loc[t,m]):,}" for m in MAIN) + " |")
L += ["", "## 5. Flags\n",
      "Thresholds fixed in advance: **ambiguous** if specificity < 1.5, **sparse** if "
      "mean detection of own markers < 0.05. Flagged types carry a red `*` in every "
      "panel and are never dropped.\n",
      "| cell type | " + " | ".join(ML[m] for m in MAIN) + " |",
      "|---|" + "---|" * len(MAIN)]
for t in CT:
    if (flag.loc[t, MAIN] != "ok").any():
        L.append(f"| {S[t]} | " + " | ".join(str(flag.loc[t, m]) for m in MAIN) + " |")
L += ["",
      "**Ductal 1 / Ductal 2 - ambiguous in the reference itself, in both atlas "
      "versions.** Downstream claims that separate the two ductal subtypes are not "
      "supported by the label transfer. Claims treating ductal/tumour as a single "
      "compartment are unaffected. They are shown, flagged, and kept in every panel.\n",
      "**Stellate** is ambiguous with fibroblast in the whole-cell arm "
      f"({spec.loc['Stellate cell','post_whole']:.2f}) and recovers in partials "
      f"({spec.loc['Stellate cell','post_partial']:.2f}). Consider a merged stromal "
      "compartment for stellate-specific questions.\n",
      "**Immune types are sparse, not incoherent** - T, B and macrophage hold "
      "specificity 2.6-3.0 in partials, among the cleanest of any group, on detection "
      "of 0.5-2.4%. Per-cell immune marker calls on partials are unreliable; aggregate "
      "measures over many entities are sound.\n",
      "## Files\n", "| file | contents |", "|---|---|",
      "| `S11_marker_validation_dotplot.{pdf,png}` | five-column dot plot, common detection scale |",
      "| `S12_marker_validation_dotplot_rescaled.{pdf,png}` | same, detection rescaled within group |",
      "| `S13_marker_validation_heatmap.{pdf,png}` | heatmap, same five columns |",
      "| `S14_marker_coherence_summary.{pdf,png}` | specificity and detection, five groups |",
      "| `S15a_atlas_sensitivity_dotplot.{pdf,png}` | cPMI reference vs full atlas, dot plot |",
      "| `S15b_atlas_sensitivity_summary.{pdf,png}` | specificity, Δ specificity, CA001063 contribution |",
      "| `tables/atlas_composition.tsv` | cells and fractions per type, both atlas versions |",
      "| `tables/group_totals.tsv`, `tables/group_depth_summary.tsv` | cells and depth per group (x cell type) |",
      "| `tables/marker_expression_summary.tsv` | expression + detection per group/type/gene |",
      "| `tables/marker_expression_by_patient.tsv` | same, per patient, whole and partial separate |",
      "| `tables/label_coherence.tsv` | specificity, confusable partner, flags |",
      "| `tables/marker_definition.json` | frozen marker set, excluded study, primary reference |",
      "| `scripts/v01,v02,v03*.py` | generation scripts |"]
(OUT / "S11_S15_marker_validation_REPORT.md").write_text("\n".join(L))
old = OUT / "S11_S14_marker_validation_REPORT.md"
if old.exists():
    old.unlink()
print("\n".join(L[:60]))
