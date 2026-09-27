# SegBench figures

The active v2 pipeline was copied from
`tracer_campaign/final/scripts/segbench_figures/v2`. It writes to the existing
campaign `final/` tree through the paths in `scripts/settings.py`.

## Run order

1. `s01_pull_cirro.py`
2. `s02_original_rctd.py` (`slurm/s02_original_rctd.slurm` for heavy fits)
3. `s03_compare_validation.py`
4. `s04_assemble.py`
5. `s05_run_status.py`
6. `s06_bio_metrics.py` (`slurm/s06_bio_metrics.slurm`)
7. `s07_stats.py`
8. `s08_canonical_markers.py` (`slurm/s08_canonical_markers.slurm`)
9. `s09_tracer_centroids.py`
10. `s10_main_figure.py`
11. `s20_supp.py`

`figlib.py`, `labels.py`, `settings.py`, and the bundled `figstyle.py` are shared
helpers. Cirro and the SegBench evaluator remain external software/data
dependencies; their paths and run IDs are frozen in `settings.py`.

## Figure map

| Output | Generating script |
|---|---|
| `segbench_main/panels_v2/*` | `scripts/s10_main_figure.py` (`export_panels`) |
| `segbench_main/segbench_main_v2.png` | `scripts/s10_main_figure.py` |
| `segbench_supp/Supp_S1_availability.png` | `scripts/s20_supp.py` |
| `segbench_supp/Supp_S2_concordance_markers.png` | `scripts/s20_supp.py` |
| `segbench_supp/Supp_S3_runtime_memory.png` | `scripts/s20_supp.py` |
| `segbench_supp/Supp_S4_q25_q50_rois.png` | `scripts/s20_supp.py` (reuses `s10_main_figure.build`) |
