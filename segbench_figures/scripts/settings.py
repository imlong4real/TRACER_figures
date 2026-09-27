"""Frozen selection for the v2 SegBench figure: one Cirro run per setting."""
from pathlib import Path

PROJECT = "d57d7407-4c32-4256-bd2e-aa1e037569aa"
ROOT = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ = ROOT / "objects" / "segbench_fig"
RUNS_DIR = OBJ / "runs"
PANELS_MAIN = ROOT / "panels" / "segbench_main"
PANELS_SUPP = ROOT / "panels" / "segbench_supp"
PIPE = Path("/scratch4/adeshpa6/segmentation_benchmark_pipeline")
ROI_UPLOAD = PIPE / "results/segbench_multiplatform_v1/cirro_upload"
ROI_DESIGN = PIPE / "results/segbench_multiplatform_v1/roi_design"
STAGING = Path("/scratch4/adeshpa6/segbench_cirro_staging/datasets")

# setting key -> (dataset, platform, roi label, run id, main-figure column?)
SETTINGS = {
    "xenium_lung__whole":         ("xenium_lung",         "Xenium",   "whole", "e9688017-4f24-4989-97af-3004b9589694", True),
    "xenium5k_cervical__q25":     ("xenium5k_cervical",   "Xenium5K", "q25",   "b9390059-9904-4719-8e92-b054cb0555a9", False),
    "xenium5k_cervical__q50":     ("xenium5k_cervical",   "Xenium5K", "q50",   "6e3c866a-dbdc-4b6a-9358-7a59f33c2eba", False),
    "xenium5k_cervical__q75":     ("xenium5k_cervical",   "Xenium5K", "q75",   "f2456f15-b545-4681-9af4-33494a913495", True),
    "cosmx_nsclc__q25":           ("cosmx_nsclc",         "CosMx",    "q25",   "8890bf1d-5bff-49ef-a913-400d4ddaba5a", False),
    "cosmx_nsclc__q50":           ("cosmx_nsclc",         "CosMx",    "q50",   "d637223b-bcf9-4ddb-8420-005d2ee7cd61", False),
    "cosmx_nsclc__q75":           ("cosmx_nsclc",         "CosMx",    "q75",   "b9aab86e-5431-499a-82df-b28363d38b08", True),
    "merfish_mouse_ileum__whole": ("merfish_mouse_ileum", "MERFISH",  "whole", "7657e151-dbce-4132-bafb-acb7d524c558", True),
    "atera_cervical__q25":        ("atera_cervical",      "Atera",    "q25",   "3903d39c-f1a6-4e1a-8b3a-c9ecd70b81c3", False),
    "atera_cervical__q50":        ("atera_cervical",      "Atera",    "q50",   "8c94b588-5f99-4f3c-a882-8615ed688c4a", False),
    "atera_cervical__q75":        ("atera_cervical",      "Atera",    "q75",   "c8367e92-5356-4d91-851a-a28ab053c2d6", True),
    "visiumhd_kidney__whole":     ("visiumhd_kidney",     "VisiumHD", "whole", "b3066e11-ee81-4f70-9df1-0bffe6ed9d7c", True),
}
MAIN_ORDER = ["xenium_lung__whole", "xenium5k_cervical__q75", "cosmx_nsclc__q75",
              "merfish_mouse_ileum__whole", "atera_cervical__q75", "visiumhd_kidney__whole"]

# Original input: frozen transcripts grouped by input cell_id.
def original_source(setting: str) -> Path:
    if setting == "xenium_lung__whole":
        return STAGING / "tsu20/filtered_df_standardized.parquet"
    if setting == "visiumhd_kidney__whole":
        return STAGING / "kidney/kidney_seg_input.parquet"
    return ROI_UPLOAD / "rois" / f"{setting}.parquet"

UNASSIGNED = {"UNASSIGNED", "-1", "0", "None", "nan", ""}   # make_exact_xenium_bundle.py
