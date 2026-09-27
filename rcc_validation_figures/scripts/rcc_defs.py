"""Shared RCC definitions: markers, lineage assignments, palette."""
PANEL23 = ["cd4", "cd20", "cd8a", "cd3e", "cd138", "hla-dr", "cd11c", "cd68",
           "cd16", "granzymeb", "cd163", "cd45ra", "pcna", "cd45ro", "ki-67",
           "pd-1", "vista", "pd-l1", "lag-3", "beta-catenin", "cd31", "pten",
           "panck"]
CELLLOC = ["e-cadherin", "vimentin", "alphasma", "cd45"]
AF = ["af_blue", "af_green", "af_yellow", "af_red"]
MARKERS = PANEL23 + CELLLOC          # the 27 requested markers
ALLCH = MARKERS + AF

# one canonical, lineage-defining marker per transferred cell type.  These are
# the matched markers of the orthogonal-validation test; everything else is a
# mismatched control for that cell type.
CANON = {"T cell": "cd3e", "B cell": "cd20", "Plasma cell": "cd138",
         "Myeloid": "cd68", "Endothelial": "cd31", "Mural": "alphasma",
         "Tumor": "panck"}
TYPES = list(CANON)

# wider supporting sets, used only for the specificity matrices
SUPPORT = {
    "T cell": ["cd3e", "cd8a", "cd4", "cd45ro", "cd45ra", "cd45"],
    "B cell": ["cd20", "hla-dr", "cd45"],
    "Plasma cell": ["cd138", "cd45"],
    "Myeloid": ["cd68", "cd163", "cd11c", "hla-dr", "cd16", "cd45"],
    "Endothelial": ["cd31", "vimentin"],
    "Mural": ["alphasma", "vimentin"],
    "Tumor": ["panck", "e-cadherin", "beta-catenin", "pten"],
    "Mast cell": ["cd45"],
}

DISPLAY = {
    "cd4": "CD4", "cd20": "CD20", "cd8a": "CD8A", "cd3e": "CD3E",
    "cd138": "CD138", "hla-dr": "HLA-DR", "cd11c": "CD11c", "cd68": "CD68",
    "cd16": "CD16", "granzymeb": "GzmB", "cd163": "CD163", "cd45ra": "CD45RA",
    "pcna": "PCNA", "cd45ro": "CD45RO", "ki-67": "Ki-67", "pd-1": "PD-1",
    "vista": "VISTA", "pd-l1": "PD-L1", "lag-3": "LAG-3",
    "beta-catenin": "β-catenin", "cd31": "CD31", "pten": "PTEN",
    "panck": "PanCK", "e-cadherin": "E-cadherin", "vimentin": "Vimentin",
    "alphasma": "αSMA", "cd45": "CD45", "dapi": "DAPI",
    "af_blue": "AF blue", "af_green": "AF green", "af_yellow": "AF yellow",
    "af_red": "AF red",
}
