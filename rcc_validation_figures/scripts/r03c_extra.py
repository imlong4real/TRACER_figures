#!/usr/bin/env python
"""r03c - the two remaining null effect sizes and the spillover retention ratio.

Cliff's delta is the primary effect size in the figure, so the swapped-marker-map
and autofluorescence controls need it too rather than a median contrast, and the
spillover comparison is easier to read as the fraction of matched-marker
positivity that SURVIVES outside the 0-2 um bleed zone.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcutil import mklog, cliffs_delta, trim
from rcc_defs import CANON, TYPES, AF

FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ, TAB = FINAL / "objects", FINAL / "tables"
MIN_PX, N_CAP = 8, 60_000
log = mklog("r03c")


def make_delta(P):
    tot = np.nansum(P, axis=1)
    k = np.maximum(np.isfinite(P).sum(axis=1) - 1, 1)
    idx = np.arange(len(P))
    return lambda code: P[idx, code] - (tot - np.nan_to_num(P[idx, code])) / k


def main() -> int:
    src = OBJ / "rcc_percell_protein.parquet"
    base = pd.read_parquet(src, columns=["entity_class", "cell_type", "n_px"])
    base["entity_class"] = base.entity_class.astype(str)
    base["cell_type"] = base.cell_type.astype(str)
    canon = [CANON[t] for t in TYPES]
    rng = np.random.default_rng(0)
    swap = np.arange(len(TYPES))
    while np.any(swap == np.arange(len(TYPES))):
        swap = rng.permutation(len(TYPES))

    out = []
    for view, sfx in (("all_pixels", "pos"), ("spillover_free", "pos_far")):
        P = np.column_stack([pd.read_parquet(src, columns=[f"{m}_{sfx}"])
                             .iloc[:, 0].to_numpy(np.float64) for m in canon])
        PAF = np.column_stack([pd.read_parquet(src, columns=[f"{m}_{sfx}"])
                               .iloc[:, 0].to_numpy(np.float64) for m in AF])
        ok = (base.n_px.to_numpy() >= MIN_PX) & base.cell_type.isin(TYPES).to_numpy()
        usable = ok & (np.isfinite(P).sum(1) >= len(TYPES) - 1)
        tcode = pd.Categorical(base.cell_type, categories=TYPES).codes.astype(int)
        for cls in ("partial", "whole"):
            m = usable & (base.entity_class.to_numpy() == cls)
            idx = np.flatnonzero(m)
            if len(idx) > N_CAP:
                idx = np.sort(rng.choice(idx, N_CAP, replace=False))
            code = tcode[idx]
            fd = make_delta(P[idx]); fdaf = make_delta(PAF[idx])
            off = rng.integers(1, len(TYPES), len(code))
            d_obs = fd(code)
            d_null = fd((code + off) % len(TYPES))
            d_swap = fd(swap[code])
            d_af = fdaf(rng.integers(0, PAF.shape[1], len(code)))
            d_afn = fdaf(rng.integers(0, PAF.shape[1], len(code)))
            out.append(dict(view=view, entity_class=cls, n=int(len(idx)),
                            cliffs_observed=cliffs_delta(d_obs, d_null),
                            cliffs_swapped_map=cliffs_delta(d_swap, d_null),
                            cliffs_autofluor=cliffs_delta(d_af, d_afn),
                            median_observed=float(np.median(d_obs)),
                            median_swapped=float(np.median(d_swap)),
                            median_autofluor=float(np.median(d_af))))
            del fd, fdaf, d_obs, d_null, d_swap, d_af, d_afn
        del P, PAF
        trim()
    N = pd.DataFrame(out)
    N.to_csv(TAB / "r03c_null_effect_sizes.csv", index=False)
    log(N.round(4).to_string(index=False))

    S = pd.read_csv(TAB / "r03b_spillover_tests.csv")
    S["retention"] = S.mean_beyond_2um / S.mean_within_2um
    piv = S.pivot_table(index="cell_type", columns="entity_class",
                        values="retention")
    piv["partial_minus_whole"] = piv["partial"] - piv["whole"]
    piv.reset_index().to_csv(TAB / "r03c_spillover_retention.csv", index=False)
    log(f"retention higher in partials for "
        f"{int((piv.partial_minus_whole > 0).sum())}/{len(piv)} lineages; "
        f"partial median {piv['partial'].median():.2f}, "
        f"whole median {piv['whole'].median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
