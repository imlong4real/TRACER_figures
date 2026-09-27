"""Reference concordance and marker specificity for Original and TRACER whole/partial.

Cirro's evaluator never scored Original and scored TRACER only pooled.  These
are computed here with the evaluator's OWN functions
(``segbench.evaluate.reference_consistency`` / ``marker_specificity``), with
the same reference, the same cell-type universe (``common_celltypes``, >=50
held-out cells) and the same per-cell RCTD labels.

Gate: the pooled TRACER value and a published non-TRACER method are recomputed
through the identical call and must reproduce Cirro's published numbers to
1e-5 - float32 storage in the h5ad alone moves a correlation by ~1e-6.

    s06_bio_metrics.py <setting>
"""
import json, sys, tempfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/scratch4/adeshpa6/segmentation_benchmark_pipeline/src")
from settings import OBJ, RUNS_DIR
from s02_original_rctd import local_reference, recorded_cmd
import segbench.evaluate as ev

METRICS = ["kendall_tau_median", "pearson_r_median", "spearman_rho_median",
           "marker_logfc_median"]


def score(h5, per_cell, ref, col, kept):
    row = ev.EvalRow(dataset="segbench_v2", method="x")
    ev.reference_consistency(row, cell_h5ad=h5, rctd_per_cell=per_cell,
                             reference_h5ad=ref, celltype_col=col, kept_types=kept)
    ev.marker_specificity(row, cell_h5ad=h5, rctd_per_cell=per_cell,
                          reference_h5ad=ref, celltype_col=col, kept_types=kept)
    d = row.to_dict()
    return {m: d.get(m, np.nan) for m in METRICS}


def published(setting, method):
    """Cirro's value for a method, from the run's own tidy table."""
    pr = pd.read_csv(RUNS_DIR / setting / "evaluation/plot_ready_table.tsv", sep="\t")
    pr = pr[pr.method == method].set_index("metric")["value"]
    return {m: float(pr[m]) if m in pr and pd.notna(pr[m]) else np.nan for m in METRICS}


def main(setting):
    args = recorded_cmd(setting)
    ref = local_reference(Path(args[args.index("--reference-h5ad") + 1]).name)
    col = args[args.index("--reference-celltype-col") + 1]
    kept, dropped = ev.common_celltypes(ref, col, 50)
    rows, checks = [], []
    tr = RUNS_DIR / setting / "methods/tracer_seg/rctd"
    tmp = Path(tempfile.mkdtemp(dir=OBJ))

    # --- gate: reproduce published values through the identical call ---------
    for method_dir, pub_key in (("tracer_seg", "tracer"), ("baysor", "baysor"),
                                ("bin2cell", "bin2cell")):
        r = RUNS_DIR / setting / f"methods/{method_dir}/rctd"
        if not (r / "rctd_input.h5ad").exists():
            continue
        got = score(r / "rctd_input.h5ad", r / "rctd_cell_assignments_post.tsv", ref, col, kept)
        want = published(setting, pub_key)
        diff = max(abs(got[m] - want[m]) for m in METRICS
                   if np.isfinite(got[m]) and np.isfinite(want[m]))
        checks.append({"setting": setting, "method": pub_key, "max_abs_diff": diff,
                       "pass": bool(diff < 1e-5), **{f"local_{m}": got[m] for m in METRICS},
                       **{f"cirro_{m}": want[m] for m in METRICS}})
        print(f"[gate] {setting} {pub_key}: max |diff| = {diff:.2e}", flush=True)

    # --- Original -------------------------------------------------------------
    o = OBJ / "original" / setting
    rows.append({"method": "Original", **score(o / "original_rctd_input.h5ad",
                 o / "rctd/rctd_cell_assignments_post.tsv", ref, col, kept)})

    # --- TRACER whole / partial: same h5ad and labels, subset by stratum -------
    a = ad.read_h5ad(tr / "rctd_input.h5ad")
    status = a.obs["whole_partial_status"].astype(str)
    for stratum, label in (("whole", "TRACER whole"), ("partial", "TRACER partial")):
        sub = a[status.to_numpy() == stratum].copy()
        p = tmp / f"tracer_{stratum}.h5ad"; sub.write_h5ad(p)
        rows.append({"method": label, **score(p, tr / "rctd_cell_assignments_post.tsv",
                                               ref, col, kept)})
        p.unlink()
    del a
    tmp.rmdir()

    out = pd.DataFrame(rows); out.insert(0, "setting", setting)
    out["kept_celltypes"] = ",".join(kept); out["dropped_celltypes"] = ",".join(dropped)
    d = OBJ / "tables/bio"; d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"bio__{setting}.tsv", sep="\t", index=False)
    pd.DataFrame(checks).to_csv(d / f"gate__{setting}.tsv", sep="\t", index=False)
    print(out.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
