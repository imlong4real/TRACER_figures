#!/usr/bin/env python
"""r03b - derived statistics for the RCC figure, from the r03 tables.

Three things the raw r03 tables need before they can be plotted honestly.

1. The toroidal-shift null is a REGISTRATION null, not a brightness null.  Every
   entity footprint lies on tissue while a shifted footprint can land anywhere,
   so even the autofluorescence channels come out "enriched" (log2 0.7-1.4).
   Absolute enrichment against that null therefore cannot be read as marker
   specificity.  Row-centring each cell type's profile - subtracting its own
   median log2 enrichment across all 31 channels - removes the per-cell-type
   brightness term and leaves the specificity, which is the same quantity the
   within-entity contrast measures at the level of single entities.

2. Positivity is granular for small footprints, so a median contrast is often
   exactly 0 and understates a real distribution shift.  Cliff's delta against
   the per-entity identity-permutation draw is used as the primary effect size;
   the median and its CI are kept as the secondary, on-scale summary.

3. The spillover comparison needs a paired test that granularity cannot flatten:
   the sign test on entities whose two rates differ.

Outputs: r03b_specificity_centred.csv, r03b_effect_sizes.csv,
         r03b_spillover_tests.csv, r03b_headline.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcutil import mklog, bh, boot_ci, trim
from rcc_defs import CANON, TYPES, MARKERS, AF, ALLCH

FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ, TAB = FINAL / "objects", FINAL / "tables"
log = mklog("r03b")


def main() -> int:
    S = pd.read_csv(TAB / "r03_specificity_matrix.csv")
    for c, out in (("log2_enrich", "rel_log2"),
                   ("log2_enrich_far", "rel_log2_far")):
        S[out] = S[c] - S.groupby(["entity_class", "cell_type"])[c] \
            .transform("median")
    S.to_csv(TAB / "r03b_specificity_centred.csv", index=False)
    diag = S[S.is_canonical].pivot_table(index="cell_type",
                                         columns="entity_class",
                                         values="rel_log2")
    afb = (S[S.is_af].groupby(["entity_class", "cell_type"]).rel_log2.median()
           .unstack("entity_class"))
    log("row-centred diagonal (partial): " +
        str(diag["partial"].round(2).to_dict()))
    log("autofluorescence baseline (partial): " +
        str(afb["partial"].round(2).to_dict()))

    # ---- effect sizes, whole vs partial concordance -----------------------
    C = pd.read_csv(TAB / "r03_matched_marker_contrast.csv")
    q = C[(C.view == "all_pixels") & (C.subset == "all")
          & (C.cell_type != "ALL")]
    piv = q.pivot_table(index="cell_type", columns="entity_class",
                        values="cliffs_delta_vs_null")
    piv = piv.join(diag.rename(columns={"partial": "rel_partial",
                                        "whole": "rel_whole"}))
    piv["af_baseline_partial"] = afb["partial"]
    piv["diag_above_af_partial"] = piv.rel_partial - piv.af_baseline_partial
    rho, pv = st.spearmanr(piv["whole"], piv["partial"])
    piv.reset_index().to_csv(TAB / "r03b_effect_sizes.csv", index=False)
    log(f"whole vs partial Cliff's delta across 7 lineages: "
        f"Spearman rho={rho:.3f} p={pv:.4f}")

    # ---- spillover paired tests ------------------------------------------
    D = pd.read_parquet(OBJ / "rcc_entity_summary.parquet")
    D["cell_type"] = D.cell_type.astype(str)
    D["entity_class"] = D.entity_class.astype(str)
    rows = []
    src = OBJ / "rcc_percell_protein.parquet"
    for t in TYPES:
        m = CANON[t]
        near = pd.read_parquet(src, columns=[f"{m}_pos_near"]).iloc[:, 0] \
            .to_numpy(np.float64)
        far = pd.read_parquet(src, columns=[f"{m}_pos_far"]).iloc[:, 0] \
            .to_numpy(np.float64)
        base = pd.read_parquet(src, columns=["entity_class", "cell_type",
                                             "n_px"])
        sel_t = (base.cell_type.astype(str) == t) & (base.n_px >= 8)
        for cls in ("partial", "whole"):
            idx = np.flatnonzero(sel_t
                                 & (base.entity_class.astype(str) == cls))
            a, b = near[idx], far[idx]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if len(a) < 30:
                continue
            d = b - a
            nz = d[d != 0]
            n_up = int((nz > 0).sum())
            p_sign = (st.binomtest(n_up, len(nz), 0.5).pvalue
                      if len(nz) else np.nan)
            lo, hi = boot_ci(d, np.mean, n=1000)
            rows.append(dict(cell_type=t, marker=m, entity_class=cls,
                             n_paired=int(len(a)),
                             mean_within_2um=float(a.mean()),
                             mean_beyond_2um=float(b.mean()),
                             mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi,
                             n_discordant=int(len(nz)), n_beyond_higher=n_up,
                             frac_beyond_higher=float(n_up / max(len(nz), 1)),
                             p_sign=p_sign))
        del near, far, base
        trim()
    SPT = pd.DataFrame(rows)
    SPT["q_bh"] = bh(SPT.p_sign.to_numpy())
    SPT.to_csv(TAB / "r03b_spillover_tests.csv", index=False)
    log("spillover: partial cells with higher positivity beyond 2 um in " +
        str(int((SPT[(SPT.entity_class == 'partial')].mean_diff > 0).sum()))
        + f"/{int((SPT.entity_class=='partial').sum())} lineages")

    allr = C[(C.view == "all_pixels") & (C.subset == "all")
             & (C.cell_type == "ALL") & (C.entity_class == "partial")].iloc[0]
    spf = C[(C.view == "spillover_free") & (C.subset == "all")
            & (C.cell_type == "ALL") & (C.entity_class == "partial")]
    dis = C[(C.view == "all_pixels") & (C.subset == "discordant_neighbour")
            & (C.cell_type == "ALL") & (C.entity_class == "partial")]
    (TAB / "r03b_headline.json").write_text(json.dumps({
        "n_partial_scored": int(allr.n),
        "primary_effect_size": "Cliff's delta, own-lineage vs random other "
                               "lineage canonical marker, within entity",
        "partial_all_pixels_cliffs_delta": float(allr.cliffs_delta_vs_null),
        "partial_spillover_free_cliffs_delta":
            float(spf.cliffs_delta_vs_null.iloc[0]) if len(spf) else None,
        "partial_discordant_neighbour_cliffs_delta":
            float(dis.cliffs_delta_vs_null.iloc[0]) if len(dis) else None,
        "swapped_marker_map_median": float(allr.median_delta_swapped_map),
        "autofluorescence_median": float(allr.median_delta_autofluor),
        "per_lineage_cliffs_delta_partial":
            piv["partial"].round(4).to_dict(),
        "per_lineage_cliffs_delta_whole": piv["whole"].round(4).to_dict(),
        "whole_vs_partial_spearman_rho": float(rho),
        "whole_vs_partial_spearman_p": float(pv),
        "row_centred_diagonal_partial": diag["partial"].round(3).to_dict(),
        "autofluorescence_baseline_partial": afb["partial"].round(3).to_dict(),
        "lineages_failing_in_partials":
            [t for t in TYPES if piv.loc[t, "partial"] <= 0],
        "lineages_failing_in_whole_cells_too":
            [t for t in TYPES if piv.loc[t, "whole"] < 0.35],
    }, indent=2))
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
