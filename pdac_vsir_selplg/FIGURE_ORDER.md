# PDAC two-layer figure — proposed panel order

Every panel is standalone (PDF 600 dpi / PNG 450 dpi / SVG), no in-panel title.
Files: `panels/pdac_main/`, `panels/pdac_supp/`. Scripts: `scripts/pdac/`.

## Layer 1 — Xenium 10x: what TRACER does (mechanistic introducer)

| # | Panel | One-line message |
|---|---|---|
| a | `M1_xenium_transcript_fate_alluvial` | Transcripts are pruned, regrouped and rebuilt across the TRACER ladder |
| b | `M2_xenium_transcript_fate_accounting` | 19.5% move to a different cell, 12.4% are rescued from unassigned, 20.5% build partial cells |
| c | `M3_xenium_3d_roi` | Restored Open3D rendering: reconstructed partial cells and new immune–tumour proximities (<25 µm) |
| d | `M4_xenium_roi_immune_resolved` | An immune-poor ROI under 10X becomes immune-resolved: 6 → 144 immune cells |
| e | `M5_xenium_admixture_cleanup` | Cross-lineage admixture drops most for immune cells (Mac 0.33→0.13, B/T → 0.00) |
| f | `M6_xenium_vsig4_specificity` | VSIG4+ cells correctly attributed to macrophages 17% → 34% |
| g | `M7_xenium_immune_recovery` | T 2.13×, B 1.78×, Mac 1.35× more confidently typed cells |

## Layer 2 — Visium HD six-patient cohort: VISTA–PSGL-1 checkpoint domains

| # | Panel | One-line message |
|---|---|---|
| h | `M15_hd_roi_HC05_responder` (+ `M15b`, legend `M15c`) | 10X polygons vs TRACER footprints: T = 18 → 369 in one ROI |
| i | `M8_hd_axes_detectable` | Checkpoint axes become detectable in 5/6 patients (HC04 loses one — shown honestly) |
| j | `M9_hd_vista_enrichment_forest` | **2.80× (95% CI 1.93–4.05), 6/6 patients, sign p = 0.031, Stouffer p = 3.0×10⁻⁹** |
| k | `M11_hd_lineage_identity` (+ `M11b`) | All four checkpoint populations validate as their lineage, 6/6, raw / depth-matched / rarefied |
| l | `M10_hd_vista_partial_contribution` | Partial cells carry 0–18% of domain edges, and 100% in HC03 |
| m | `M12_hd_vista_roi_HC01_nonresponder` (+ `HC08`, legend `M12b`) | The domain appears: SELPLG+ 0 → 4 with a linked cluster |
| n | `M13_hd_vista_wholetissue_HC08_responder` (+ `HC01`) | Where the domains sit in the whole section, ROI boxed |
| o | `M14_hd_domain_structure` | Domain count and largest-domain size rise with partials |
| p | `M16_hd_wholetissue_prepost_HC05_responder` (+ `HC01`, legend `M16b`) | Whole-section immune map before vs after: T cells 1,729 → 18,586, T domains 1,041 → 4,848 |

Shared legend chips: `M0_legend_response`, `M12b_legend_vista`, `M15c_legend_celltype`.

## Supplementary

| Panel | Content |
|---|---|
| `S1_xenium_vsig4_proximity_density_audit` | **Honest negative**: the 117→72 µm proximity claim is a density artefact |
| `S2_hd_tigit_nectin2_interface` (+ `S2b`) | TIGIT–NECTIN2 contact interface — 0–2 NECTIN2+ per ROI, not main-figure material |
| `S3_hd_response_screen_chance` | The n = 6 response screen is at chance (164 vs 158.9 expected, Holm min p = 0.40) |
| `S4_hd_null_comparison` | Global vs 500 µm block null — scale matching of null to estimand |
| `S5_hd_topology_radius` | Radius sensitivity (R = 20, 30, 50, 100 µm) |
| `S6_hd_secondary_axes` | TIGIT–PVR and GAL9–TIM3 under the same within-type null |
| `S10_hd_gal9_tim3_framework` | GAL9–TIM3 under the full VISTA–PSGL1 framework: 5/6 but cohort CI includes 1 and 2 patients invert — stays supplementary |
| `S7_hd_label_confidence` | Label-transfer confidence sensitivity |
| `S8_hd_boundary_gradient` | Signed-distance tumour-boundary gradient (density-confounded; descriptive) |
| `S9_hd_roi_candidates` | Quantitative ROI ranking, showing ROIs were not hand-picked |
