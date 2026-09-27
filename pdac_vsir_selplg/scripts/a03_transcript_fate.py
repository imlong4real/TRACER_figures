#!/usr/bin/env python3
"""AUDIT 3 / data prep - Xenium transcript-fate accounting (streamed, row-group wise).

Produces the numbers behind the mechanism panels:
  * class flow across the TRACER phase ladder (for the alluvial)
  * how many transcripts are REDISTRIBUTED (start inside an original 10X cell,
    end inside a different entity)
  * how many are RECOVERED FROM UNASSIGNED (start outside any cell, end assigned)
  * how many end up constituting reconstructed PARTIAL cells

etype codes: 0 original cell, 1 partial, 3 unassigned, 5 neighboring cell.
Memory-safe: one row group (~1M rows) resident at a time; only counts kept.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pstyle as S

SRC = Path("/scratch4/adeshpa6/TRACER/datasets/pancreas_cancer_xenium_10x/"
           "pdac_io_partition_sequential.parquet")
CLASS = {0: "original cell", 1: "partial", 3: "unassigned", 5: "neighboring cell"}
PHASES = ["input", "phase1", "rescue", "group", "mid_qc", "post_group_rescue",
          "stitch", "demote", "final_rescue", "finalize"]
COLS = [f"etype_at_{p}" for p in PHASES] + ["cell_id", "label"]

pf = pq.ParquetFile(SRC)
flow = {p: Counter() for p in PHASES}
PAIRS = list(zip(PHASES[:-1], PHASES[1:])) + [
    ("input", "phase1"), ("phase1", "group"), ("group", "finalize"),
    ("input", "finalize")]          # coarse ladder used by the alluvial panel
PAIRS = list(dict.fromkeys(PAIRS))
trans = {f"{a}->{b}": Counter() for a, b in PAIRS}
fate = Counter()
total = 0

for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i, columns=COLS)
    d = t.to_pandas()
    total += len(d)
    for p in PHASES:
        u, c = np.unique(d[f"etype_at_{p}"].values, return_counts=True)
        flow[p].update(dict(zip(u.tolist(), c.tolist())))
    for a, b in PAIRS:
        key = f"{a}->{b}"
        ea = d[f"etype_at_{a}"].values.astype(np.int16)
        eb = d[f"etype_at_{b}"].values.astype(np.int16)
        code, cnt = np.unique(ea * 16 + eb, return_counts=True)
        trans[key].update(dict(zip(((c // 16, c % 16) for c in code.tolist()), cnt.tolist())))
    e_in, e_out = d.etype_at_input.values, d.etype_at_finalize.values
    same = (d.cell_id.values == d.label.values)
    assigned_out = np.isin(e_out, [0, 1, 5])
    fate["total"] += len(d)
    fate["in_orig_cell"] += int((e_in == 0).sum())
    fate["in_unassigned"] += int((e_in == 3).sum())
    fate["out_orig_cell"] += int((e_out == 0).sum())
    fate["out_partial"] += int((e_out == 1).sum())
    fate["out_neighboring"] += int((e_out == 5).sum())
    fate["out_unassigned"] += int((e_out == 3).sum())
    fate["redistributed"] += int(((e_in == 0) & assigned_out & ~same).sum())
    fate["kept_in_place"] += int(((e_in == 0) & assigned_out & same).sum())
    fate["recovered_from_unassigned"] += int(((e_in == 3) & assigned_out).sum())
    fate["unassigned_to_partial"] += int(((e_in == 3) & (e_out == 1)).sum())
    fate["unassigned_to_cell"] += int(((e_in == 3) & (e_out == 0)).sum())
    fate["origcell_to_partial"] += int(((e_in == 0) & (e_out == 1)).sum())
    fate["lost_to_unassigned"] += int(((e_in == 0) & (e_out == 3)).sum())
    del d, t
    print(f"  row group {i+1}/{pf.metadata.num_row_groups}  cum={total:,}", flush=True)

F = pd.DataFrame([{"phase": p, "etype": k, "cls": CLASS.get(k, str(k)), "n": v}
                  for p in PHASES for k, v in flow[p].items()])
F.to_csv(S.TAB / "xenium_phase_class_counts.tsv", sep="\t", index=False)
TR = pd.DataFrame([{"transition": k, "src": CLASS.get(a, a), "dst": CLASS.get(b, b), "n": v}
                   for k, c in trans.items() for (a, b), v in c.items()])
TR.to_csv(S.TAB / "xenium_phase_transitions.tsv", sep="\t", index=False)
FA = pd.DataFrame([{"quantity": k, "n": v, "pct_of_total": 100 * v / fate["total"]}
                   for k, v in fate.items()])
FA.to_csv(S.TAB / "xenium_transcript_fate.tsv", sep="\t", index=False)

print("\n=== transcript fate (n = {:,}) ===".format(fate["total"]))
print(FA.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("\n=== class counts per phase ===")
print(F.pivot_table(index="phase", columns="cls", values="n", fill_value=0)
      .reindex(PHASES).to_string())
