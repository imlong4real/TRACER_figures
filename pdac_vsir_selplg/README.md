# PDAC VISTA-VSIR / PSGL-1-SELPLG figures

`scripts/a01`-`a10` compute the frozen audit tables. `scripts/b01`-`b09` render
the standalone panels under `final/panels/pdac_main` and `pdac_supp`.
`FIGURE_ORDER.md` is the contemporaneous a-p panel plan. No assembled figure
was present in `final/panels` when this package was created, so
`scripts/assemble_main_figure.py` performs layout only and writes
`Fig_PDAC_VSIR_SELPLG.png`.

The restored Xenium 3D panel uses the bundled unmodified helpers
`helpers/build_3d_roi.py` and `helpers/fig2_style.py`. Six carried supplemental
panels came from the earlier active `audit_v3` pipeline; those exact scripts are
preserved under `scripts/upstream_audit_v3` rather than attributing the panels
to the copying wrapper. Marker-validation supplements S11-S15 have their own
active scripts under `scripts/marker_validation`.

## Run order

1. `a01_audit_xenium.py` through `a10_axis_population_identity.py` in numeric order.
2. `b01_xenium_quant.py` through `b09_identity_and_gal9.py` in numeric order.
3. For carried S3-S5 and S7-S9, run the upstream `audit_v3/30`-`45` chain in
   numeric order; `b06_supp.py` copies the frozen outputs into their current names.
4. Run marker validation `v01`, `v02`, `v03` for S11-S15.
5. Run `assemble_main_figure.py` after all main panels exist.

## Main figure/panel map

| Panel/output | Generating script |
|---|---|
| a `M1_xenium_transcript_fate_alluvial` | `b04_xenium_mechanism.py` |
| b `M2_xenium_transcript_fate_accounting` | `b01_xenium_quant.py` |
| c `M3_xenium_3d_roi` | `b08_xenium_3d_roi_restored.py` |
| d `M4_xenium_roi_immune_resolved` | `b04_xenium_mechanism.py` |
| e-g `M5`, `M6`, `M7` | `b01_xenium_quant.py` |
| h `M15`, `M15b`, `M15c` | `b05_hd_roi_footprints.py` |
| i `M8` and `M0` response legend | `b02_hd_centrepiece.py` |
| j `M9` | `b02_hd_centrepiece.py` |
| k `M11`, `M11b` | `b09_identity_and_gal9.py` |
| l `M10` | `b02_hd_centrepiece.py` |
| m `M12` patient panels and `M12b` | `b03_hd_vista_spatial.py` |
| n `M13` patient panels | `b03_hd_vista_spatial.py` |
| o `M14` | `b02_hd_centrepiece.py` |
| p `M16` patient panels and `M16b` | `b07_wholetissue_prepost.py` |
| `Fig_PDAC_VSIR_SELPLG.png` | `assemble_main_figure.py` (layout only) |

## Supplemental figure map

| Output(s) | Generating script |
|---|---|
| `S1_xenium_vsig4_proximity_density_audit` | `b01_xenium_quant.py` |
| `S2_hd_tigit_nectin2_interface`, `S2b` | `b06_supp.py` |
| `S3_hd_response_screen_chance` | `upstream_audit_v3/45_supp_panels.py` (`S2_response_screen_chance`; renamed by `b06_supp.py`) |
| `S4_hd_null_comparison` | `upstream_audit_v3/45_supp_panels.py` (`S3_null_comparison`; renamed by `b06_supp.py`) |
| `S5_hd_topology_radius` | `upstream_audit_v3/45_supp_panels.py` (`S4_topology_radius`; renamed by `b06_supp.py`) |
| `S6_hd_secondary_axes` | `b06_supp.py` |
| `S7_hd_label_confidence` | `upstream_audit_v3/45_supp_panels.py` (`S5_label_confidence`; renamed by `b06_supp.py`) |
| `S8_hd_boundary_gradient` | `upstream_audit_v3/45_supp_panels.py` (`S1_boundary_gradient`; renamed by `b06_supp.py`) |
| `S9_hd_roi_candidates` | `upstream_audit_v3/45_supp_panels.py` (`S6_roi_candidates`; renamed by `b06_supp.py`) |
| `S10_hd_gal9_tim3_framework` | `b09_identity_and_gal9.py` |
| `S11`-`S15` marker-validation panels | `marker_validation/v01_marker_tables.py`, then `v02_marker_panels.py`; `v03_report.py` writes provenance report |

Raw datasets and intermediate tables remain external. The absolute defaults in
the scientific scripts are intentionally retained to reproduce the current
campaign; the assembler accepts `--panels` and `--output` for other checkouts.
