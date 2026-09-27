"""TRACER entity centroids for the row-1 maps.

TRACER's RCTD input carries no coordinates, so each entity's centroid is the
mean x/y of the transcripts TRACER assigned to it (``tracer_id`` in
``transcripts_tracer_refined.parquet``).  Unassigned transcripts are ignored.

    s09_tracer_centroids.py <setting> [<setting> ...]
"""
import sys
from pathlib import Path
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from settings import OBJ, RUNS_DIR

for s in sys.argv[1:]:
    t = pq.read_table(RUNS_DIR / s / "methods/tracer_seg/outputs/transcripts_tracer_refined.parquet",
                      columns=["tracer_id", "x", "y", "whole_partial_status"])
    t = t.filter(pc.is_in(t["whole_partial_status"], pa.array(["whole", "partial"])))
    g = t.group_by("tracer_id").aggregate([("x", "mean"), ("y", "mean"), ("x", "count")])
    g = g.rename_columns(["tracer_id", "x", "y", "n_tx"])
    out = OBJ / "tables" / f"tracer_centroids__{s}.parquet"
    pq.write_table(g, out)
    print(s, g.num_rows, "entities", flush=True)
