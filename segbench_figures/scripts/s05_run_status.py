"""Per-method attempt history for the 12 selected runs, from the Nextflow head log.

A method's attempts = 1 + the number of "failed -- Execution is retried" notes;
it is terminal-failed if Nextflow logged "Error is ignored" for it.  Causes for
the two recurrent terminal failures come from the evidence recorded in
``method_applicability.json`` (measured OOM ladder, hidden gene floor), not from
guessing at exit codes.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import OBJ, PIPE, PROJECT, SETTINGS
from cirro import DataPortal

PROC = {"BAYSOR": "Baysor", "PROSEG": "ProSeg", "SPLIT": "SPLIT", "SEGGER": "Segger",
        "CELLADMIX": "CellAdmix", "TRACER_SEG": "TRACER", "BIN2CELL": "Bin2Cell",
        "TRACER_NOSEG": "TRACER no-seg"}
IMAGING_ONLY = ["Baysor", "ProSeg", "SPLIT", "Segger", "CellAdmix"]
app = json.loads((PIPE / "results/segbench_multiplatform_v1/method_applicability.json").read_text())


def cause(method, setting):
    for e in app["not_applicable"]:
        if e["method"].lower() == method.lower() and setting in e.get("rois", []):
            if method == "Segger":
                g = e["evidence"]["genes_surviving_at_100_counts"][setting]
                return f"hidden gene floor: {g} genes < 64 PCA comps"
            if method == "CellAdmix":
                return "OOM >512 GiB"
    return "failed"


dp = DataPortal(); ex = dp._client.execution
rows = []
for setting, (ds, plat, roi, did, _) in SETTINGS.items():
    log = str(ex.get_execution_logs(PROJECT, did))
    seen = {}
    for m in re.finditer(r"Submitted process > ([A-Z_0-9]+) \(", log):
        seen.setdefault(m.group(1), {"retries": 0, "ignored": False})
    for m in re.finditer(r"Process `([A-Z_0-9]+) \([^)]*\)` failed -- (Execution is retried \((\d+)\)|Error is ignored)", log):
        s = seen.setdefault(m.group(1), {"retries": 0, "ignored": False})
        if m.group(3):
            s["retries"] = max(s["retries"], int(m.group(3)))
        else:
            s["ignored"] = True
    methods = [p for p in PROC if p in seen]
    for p in methods:
        m = PROC[p]; s = seen[p]
        status = "failed" if s["ignored"] else "success"
        rows.append({"setting": setting, "dataset": ds, "platform": plat, "roi": roi,
                     "run_id": did, "method": m, "status": status,
                     "attempts": s["retries"] + 1,
                     "note": cause(m, setting) if status == "failed" else ""})
    if ds == "visiumhd_kidney":
        for m in IMAGING_ONLY:
            rows.append({"setting": setting, "dataset": ds, "platform": plat, "roi": roi,
                         "run_id": did, "method": m, "status": "not applicable",
                         "attempts": 0, "note": "2 µm bins, no molecule coordinates"})
    print(setting, {PROC[p]: seen[p] for p in methods}, flush=True)
out = OBJ / "tables/run_status.tsv"
pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
print("wrote", out)
