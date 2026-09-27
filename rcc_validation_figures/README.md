# RCC validation figures

This is the active RCC protein-validation chain from `tracer_campaign/final`.
The scripts intentionally retain the frozen source-data and campaign-output
paths used for the published working figures.

## Run order

1. `r01_extract_channels.py`
2. `r02_percell_protein.py`
3. `r03_stats.py`
4. `r03b_postprocess.py`
5. `r03c_extra.py`
6. `r04_roi_select.py`
7. `r05_fig_main.py`
8. `r06_fig_supp.py`
9. `r07_rna_states.py` via `slurm/r07_rna_states.slurm`
10. `r08_fig_rna_states.py`

Shared code is `rcc_defs.py`, `tcutil.py`, and `figstyle.py`. The exact label
transfer implementation imported by `r07` is bundled as
`helpers/label_transfer_spatial.py`.

## Figure map

| Output | Generating script |
|---|---|
| `Fig_RCC_protein_validation.png` (panels a-j) | `scripts/r05_fig_main.py`; ROI constituents are rendered by `roi_block`, quantitative constituents are assembled in `main` |
| `SuppRCC_S1_estimand_audit.png` | `scripts/r06_fig_supp.py` |
| `SuppRCC_S2_all_markers.png` | `scripts/r06_fig_supp.py` |
| `SuppRCC_S3_specificity_whole_and_spillfree.png` | `scripts/r06_fig_supp.py` |
| `SuppRCC_S4_roi_selection.png` | `scripts/r06_fig_supp.py` |
| `SuppRCC_S5_additional_rois.png` | `scripts/r06_fig_supp.py` using `r05_fig_main.roi_block` |
| `SuppRCC_S6_support.png` | `scripts/r06_fig_supp.py` |
| `SuppRCC_S7_rna_states.png` | compute: `scripts/r07_rna_states.py`; render: `scripts/r08_fig_rna_states.py` |

The working tree contains no separately exported a-j RCC panel files; their
exact provenance is the panel-building functions inside `r05_fig_main.py`.
