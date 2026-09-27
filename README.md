# TRACER figure-generation code

Reproducibility code for the TRACER publication figures. This repository is a
code-only view of the working analysis under `tracer_campaign`; raw data,
intermediate matrices, job outputs, and generated figures are intentionally not
versioned.

- `segbench_figures/`: SegBench main Figure 3 and supplements S1-S4.
- `rcc_validation_figures/`: RCC protein-validation main figure and supplements
  S1-S7.
- `pdac_vsir_selplg/`: PDAC Xenium/Visium HD VISTA-VSIR / PSGL-1-SELPLG main
  panels, supplements, and the a-p assembly script.

Each folder contains the run order, input-path assumptions, and an exact
figure-to-script map. Scientific/statistical logic is copied from the active
working scripts; only import/job paths needed to make bundled helpers usable
from this checkout were adjusted.
