"""Stream every file the v2 figure needs from the 12 selected Cirro runs.

Restartable: a file already on disk at the published size is skipped.
Files go through the SDK's boto3 download straight to disk (login node is
a 5 GB cgroup); read_bytes() would buffer a whole file in memory first.
"""
import re, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from settings import SETTINGS, PROJECT, RUNS_DIR
from cirro import DataPortal

KEEP = re.compile(
    r"(evaluation/[^/]+\.(tsv|json|csv|txt)$)"
    r"|(methods/[^/]+/rctd/(rctd_cell_assignments_post\.tsv|rctd_cmd\.txt|rctd_celltype_label_map\.tsv"
    r"|rctd_run_summary\.json|rctd_input_info\.json|rctd_input\.h5ad|rctd_entropy_metrics\.tsv)$)"
    r"|(methods/[^/]+/(resource_usage\.json|benchmark_stats\.json|config_receipt\.json)$)"
    r"|(methods/tracer_seg/outputs/transcripts_tracer_refined\.parquet$)"
    r"|(frozen_input_receipt\.json$)")

dp = DataPortal(); proj = dp.get_project_by_id(PROJECT)
manifest = {}
for key, (ds, plat, roi, did, _) in SETTINGS.items():
    d = proj.get_dataset_by_id(did)
    out = RUNS_DIR / key
    got = []
    for f in d.list_files():
        if not KEEP.search(f.name):
            continue
        rel = f.name.split("benchmark_results/", 1)[-1]
        dest = out / rel
        size = getattr(f, "size_bytes", None)
        if dest.exists() and (size is None or dest.stat().st_size == size):
            got.append(rel); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        landed = Path(f.download(str(out / "_dl")))       # boto3 -> disk, not memory
        landed.replace(dest)
        got.append(rel)
    manifest[key] = {"run_id": did, "status": str(d.status), "n_files": len(got)}
    print(f"{key:30s} {did[:8]} {d.status} files={len(got)}", flush=True)
(RUNS_DIR / "pull_manifest.json").write_text(json.dumps(manifest, indent=2))
