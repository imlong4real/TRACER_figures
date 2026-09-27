"""Gate: a local refit of a PUBLISHED RCTD input must reproduce Cirro's output.

Original is only trustworthy if the local RCTD (same script, same spacexr
commit, same reference and parameters) returns what the Cirro evaluator
returned on an input it already scored. Compared per cell, on the cell ids
both runs scored.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import OBJ, RUNS_DIR

rows = []
for d in sorted((OBJ / "validation").glob("*__*")):
    setting, method = d.name.rsplit("__", 1)
    loc = d / "rctd/rctd_cell_assignments_post.tsv"
    pub = RUNS_DIR / setting / f"methods/{method}/rctd/rctd_cell_assignments_post.tsv"
    if not loc.exists():
        print(f"{d.name}: no local output yet"); continue
    a = pd.read_csv(pub, sep="\t").set_index("cell_id")
    b = pd.read_csv(loc, sep="\t").set_index("cell_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    r = {"setting": setting, "method": method,
         "n_published": int(len(pd.read_csv(pub, sep="\t", usecols=["cell_id"]))),
         "n_local": int(len(pd.read_csv(loc, sep="\t", usecols=["cell_id"]))),
         "n_common": int(len(common)),
         "spot_class_agreement": float((a.doublet_status == b.doublet_status).mean()),
         "dominant_type_agreement": float((a.dominant_celltype == b.dominant_celltype).mean()),
         "max_weight_max_abs_diff": float(np.nanmax(np.abs(a.max_weight - b.max_weight))),
         "entropy_max_abs_diff": float(np.nanmax(np.abs(a.entropy - b.entropy))),
         "max_weight_pearson": float(np.corrcoef(a.max_weight, b.max_weight)[0, 1]),
         "entropy_pearson": float(np.corrcoef(a.entropy, b.entropy)[0, 1])}
    # Reproduction, not approximation: same cells, same calls, weights equal
    # to numerical precision.
    r["pass"] = bool(r["n_published"] == r["n_local"] == r["n_common"]
                     and r["spot_class_agreement"] >= 0.999
                     and r["dominant_type_agreement"] >= 0.999
                     and r["max_weight_max_abs_diff"] < 1e-3
                     and r["entropy_max_abs_diff"] < 1e-3)
    rows.append(r)
    print(json.dumps(r, indent=1))
out = OBJ / "validation/validation_summary.tsv"
pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
print("wrote", out)
