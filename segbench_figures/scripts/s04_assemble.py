"""Assemble one per-entity table per setting, and the summary tables the figure reads.

Every entity a method emitted is kept, scored or not. RCTD drops entities
below the frozen UMI floor before fitting; those are carried as
``call = "unscored"`` rather than silently omitted, because the fraction
differs by method (TRACER partials are small) and omitting them would make
a composition bar misstate entity counts.

TRACER whole vs partial comes from ``whole_partial_status`` in TRACER's own
RCTD input obs, keyed by ``tracer_id`` (== the RCTD ``cell_id``).
"""
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import OBJ, RUNS_DIR, SETTINGS
from labels import harmonize

TAB = OBJ / "tables"; TAB.mkdir(parents=True, exist_ok=True)
DISPLAY = {"baysor": "Baysor", "proseg": "ProSeg", "split": "SPLIT", "segger": "Segger",
           "celladmix": "CellAdmix", "bin2cell": "Bin2Cell", "tracer_noseg": "TRACER no-seg"}
CALLS = ["singlet", "doublet_certain", "doublet_uncertain", "reject", "unscored"]
# Candidates for the row-1 landscape: main-grid outputs that each tile the whole
# field on their own.  TRACER is split into whole/partial and SPLIT/CellAdmix
# inherit Original's geometry, so neither is a separate landscape.
LANDSCAPE = ["Original", "Baysor", "ProSeg", "Segger"]


def load(setting, tag, h5, tsv):
    obs = ad.read_h5ad(h5, backed="r").obs.copy()
    obs.index = obs.index.astype(str)
    a = pd.read_csv(tsv, sep="\t")
    a["cell_id"] = a["cell_id"].astype(str)
    a = a.set_index("cell_id")
    df = pd.DataFrame(index=obs.index)
    df["x"] = obs["x_centroid"].to_numpy(float) if "x_centroid" in obs else np.nan
    df["y"] = obs["y_centroid"].to_numpy(float) if "y_centroid" in obs else np.nan
    df["stratum"] = obs["whole_partial_status"].astype(str).to_numpy() \
        if "whole_partial_status" in obs else ""
    df = df.join(a[["dominant_celltype", "max_weight", "entropy", "doublet_status"]], how="left")
    extra = a.index.difference(df.index)
    if len(extra):
        raise SystemExit(f"{setting}/{tag}: {len(extra)} scored ids not in RCTD input")
    df["call"] = df["doublet_status"].fillna("unscored").astype(str)
    df["celltype"] = [harmonize(setting, c) if isinstance(c, str) else None
                      for c in df["dominant_celltype"]]
    return df.drop(columns="doublet_status").reset_index(names="cell_id")


def main():
    comp, summ, choice = [], [], []
    for setting in SETTINGS:
        parts = []
        o = OBJ / "original" / setting
        if (o / "rctd/rctd_cell_assignments_post.tsv").exists():
            d = load(setting, "Original", o / "original_rctd_input.h5ad",
                     o / "rctd/rctd_cell_assignments_post.tsv")
            d["method"] = "Original"; parts.append(d)
        else:
            print(f"[{setting}] Original RCTD not available yet", flush=True)
        for mdir in sorted((RUNS_DIR / setting / "methods").iterdir()):
            r = mdir / "rctd"
            if not (r / "rctd_cell_assignments_post.tsv").exists():
                continue
            d = load(setting, mdir.name, r / "rctd_input.h5ad",
                     r / "rctd_cell_assignments_post.tsv")
            if mdir.name == "tracer_seg":
                bad = ~d["stratum"].isin(["whole", "partial"])
                if bad.any():
                    raise SystemExit(f"{setting}: {bad.sum()} TRACER entities lack a stratum")
                d["method"] = d["stratum"].map({"whole": "TRACER whole",
                                                 "partial": "TRACER partial"})
            else:
                d["method"] = DISPLAY[mdir.name]
            parts.append(d)
        cells = pd.concat(parts, ignore_index=True)
        cells.insert(0, "setting", setting)
        cells.to_parquet(TAB / f"percell__{setting}.parquet", index=False)

        for m, g in cells.groupby("method", sort=False):
            n = g["call"].value_counts()
            for c in CALLS:
                comp.append({"setting": setting, "method": m, "call": c, "n": int(n.get(c, 0))})
            s = g[g["call"] != "unscored"]
            summ.append({"setting": setting, "method": m, "n_entities": len(g),
                         "n_scored": len(s), "frac_unscored": 1 - len(s) / len(g),
                         "n_singlet": int((g["call"] == "singlet").sum()),
                         "singlet_frac_of_entities": float((g["call"] == "singlet").mean()),
                         "max_weight_median": float(s["max_weight"].median()),
                         "entropy_median": float(s["entropy"].median())})
        # Landscape rule, fixed before plotting: the output with the most
        # confidently typed cells (RCTD singlets).  A fraction would reward a
        # sparse output - Baysor on whole lung has the highest singlet
        # fraction but a third of the cells.
        cand = [r for r in summ if r["setting"] == setting and r["method"] in LANDSCAPE]
        best = max(cand, key=lambda r: r["n_singlet"]) if cand else None
        for r in cand:
            choice.append({"setting": setting, "method": r["method"],
                           "n_singlet": r["n_singlet"],
                           "singlet_frac_of_entities": r["singlet_frac_of_entities"],
                           "chosen": best is not None and r["method"] == best["method"]})
        print(f"[{setting}] {len(cells):,} entities; landscape -> "
              f"{best['method'] if best else None}", flush=True)
    pd.DataFrame(comp).to_csv(TAB / "rctd_call_composition.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(TAB / "rctd_summary.tsv", sep="\t", index=False)
    pd.DataFrame(choice).to_csv(TAB / "landscape_choice.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
