"""Build the Original segmentation's RCTD input and fit it with the evaluator's RCTD.

Original = the frozen input transcripts grouped by their input ``cell_id``,
dropping the pipeline's own unassigned sentinels (``make_exact_xenium_bundle.py``).
The inputs are already QV- and control-filtered upstream, so no further filter
is applied: Original sees exactly the transcript population every method saw.

RCTD is invoked with the exact arguments Cirro recorded in that run's
``rctd_cmd.txt`` - only the three paths are swapped for local ones - using the
repo's ``run_rctd.R`` (byte-identical to what every run staged) and spacexr at
the container's pinned commit.

    s02_original_rctd.py build   <setting>      # Original h5ad only
    s02_original_rctd.py rctd    <setting>      # build + fit
    s02_original_rctd.py validate <setting> <method>   # refit a PUBLISHED input
"""
import hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from settings import (OBJ, PIPE, ROI_UPLOAD, RUNS_DIR, SETTINGS, STAGING,
                      UNASSIGNED, original_source)

RSCRIPT = "/scratch4/adeshpa6/segbench_envs/envs/r2/bin/Rscript"
RCTD_R = PIPE / "workflow/scripts/run_rctd.R"
os.environ.setdefault("RETICULATE_PYTHON", "/scratch4/adeshpa6/TRACER/.venv_rcc/bin/python")

REF_DIRS = [ROI_UPLOAD / "references", STAGING / "tsu20/references",
            STAGING / "kidney/references"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def recorded_cmd(setting: str) -> list[str]:
    """The RCTD argv Cirro ran for this setting (identical across methods)."""
    cmd = (RUNS_DIR / setting / "methods/tracer_seg/rctd/rctd_cmd.txt").read_text().split()
    return cmd[cmd.index(next(a for a in cmd if a.endswith("run_rctd.R"))) + 1:]


def local_reference(name: str) -> Path:
    for d in REF_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def verify_reference(setting: str, ref: Path) -> dict:
    """Local reference must be the file Cirro scored against."""
    out = {"reference": str(ref), "local_sha256": sha256(ref)}
    man = RUNS_DIR / setting / "evaluation/reference_split_manifest.json"
    if man.exists():
        want = json.loads(man.read_text())["evaluation_holdout"]["sha256"]
        out["cirro_sha256"] = want
        if out["local_sha256"] != want:
            raise SystemExit(f"{setting}: local reference {ref.name} != Cirro holdout sha")
        out["verified"] = "sha256 matches reference_split_manifest.json"
    else:
        out["verified"] = "no split manifest published for this run (harmonized reference)"
    return out


def build_original(setting: str, outdir: Path) -> Path:
    import anndata as ad
    src = original_source(setting)
    t0 = time.time()
    tbl = pq.read_table(src, columns=["cell_id", "feature_name", "x", "y"])
    df = tbl.to_pandas()
    del tbl
    df["cell_id"] = df["cell_id"].astype(str)
    n_all = len(df)
    df = df[~df["cell_id"].isin(UNASSIGNED)]
    cells = pd.Categorical(df["cell_id"])
    genes = pd.Categorical(df["feature_name"].astype(str))
    X = sp.coo_matrix((np.ones(len(df), dtype=np.float64),
                       (cells.codes, genes.codes)),
                      shape=(len(cells.categories), len(genes.categories))).tocsr()
    X.sum_duplicates()
    cx = np.bincount(cells.codes, weights=df["x"].to_numpy(np.float64)) / np.bincount(cells.codes)
    cy = np.bincount(cells.codes, weights=df["y"].to_numpy(np.float64)) / np.bincount(cells.codes)
    obs = pd.DataFrame({"x_centroid": cx, "y_centroid": cy},
                       index=pd.Index(cells.categories.astype(str), name="cell_id"))
    var = pd.DataFrame(index=pd.Index(genes.categories.astype(str), name="gene"))
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = a.X.copy()
    outdir.mkdir(parents=True, exist_ok=True)
    h5 = outdir / "original_rctd_input.h5ad"
    a.write_h5ad(h5)
    receipt = {"setting": setting, "source": str(src), "source_sha256": sha256(src),
               "rule": "group frozen transcripts by input cell_id; drop "
                       + ",".join(sorted(repr(u) for u in UNASSIGNED)),
               "n_transcripts_source": int(n_all), "n_transcripts_assigned": int(len(df)),
               "n_cells": int(a.n_obs), "n_genes": int(a.n_vars),
               "median_transcripts_per_cell": float(np.median(np.asarray(X.sum(1)).ravel())),
               "seconds": round(time.time() - t0, 1)}
    (outdir / "original_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    return h5


def fit(setting: str, h5: Path, outdir: Path, cores: int) -> None:
    args = recorded_cmd(setting)
    i = args.index("--reference-h5ad")
    ref = local_reference(Path(args[i + 1]).name)
    ref_info = verify_reference(setting, ref)
    args[i + 1] = str(ref)
    args[args.index("--spatial-h5ad") + 1] = str(h5)
    args[args.index("--outdir") + 1] = str(outdir / "rctd")
    args[args.index("--max-cores") + 1] = str(cores)   # parallelism only
    cmd = [RSCRIPT, str(RCTD_R)] + args
    (outdir / "rctd").mkdir(parents=True, exist_ok=True)
    (outdir / "rctd/rctd_cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
    (outdir / "rctd/reference_verification.json").write_text(json.dumps(ref_info, indent=2))
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    (outdir / "rctd/rctd.log").write_text(p.stdout + "\n" + p.stderr)
    print(f"RCTD rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    if p.returncode != 0:
        print(p.stderr[-3000:]); raise SystemExit(p.returncode)


def main():
    mode, setting = sys.argv[1], sys.argv[2]
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    if mode in ("build", "rctd"):
        out = OBJ / "original" / setting
        h5 = build_original(setting, out)
        if mode == "rctd":
            fit(setting, h5, out, cores)
    elif mode == "validate":
        method = sys.argv[3]
        h5 = RUNS_DIR / setting / f"methods/{method}/rctd/rctd_input.h5ad"
        fit(setting, h5, OBJ / "validation" / f"{setting}__{method}", cores)


if __name__ == "__main__":
    main()
