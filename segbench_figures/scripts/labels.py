"""Cell-type harmonization and the fixed colours used by every SegBench v2 panel.

Each setting keeps its own held-out reference vocabulary; only synonyms are
merged (``T``/``T_NK``, ``Fibroblasts``/``Fibroblast``, ``Cancer``/``Tumor
Epithelial``, ``EC``/``Endothelial``, ``B``/``B_cell``).  Nothing is coarsened
beyond that - the ileum and kidney epithelial types stay separate because the
landscape they draw is the point of the panel.
"""

SYNONYM = {
    "T": "T/NK", "T_NK": "T/NK",
    "B": "B", "B_cell": "B",
    "Plasma": "Plasma", "Lymphoid": "Lymphoid",
    "Myeloid": "Myeloid", "Neutrophil": "Neutrophil", "Mast": "Mast",
    "Fibroblasts": "Fibroblast", "Fibroblast": "Fibroblast",
    "FIB/VSMC/P": "Stroma", "Smooth_muscle": "Smooth muscle",
    "Endothelial": "Endothelial", "EC": "Endothelial",
    "Cancer": "Tumor", "Tumor Epithelial": "Tumor", "Ciliated": "Ciliated",
    # mouse ileum epithelium
    "Stem_TA": "Stem/TA", "Enterocyte": "Enterocyte", "Goblet": "Goblet",
    "Paneth": "Paneth", "Enteroendocrine": "Enteroendocrine", "Tuft": "Tuft",
    # kidney nephron
    "PT": "PT", "TAL": "TAL", "PC": "PC", "IC": "IC", "POD": "Podocyte",
}

# One colour per harmonized type, fixed across all panels.  Immune in blues /
# warm oranges, stroma in greens, vessels purple, tumour dark wine; the
# tissue-specific epithelia use hues chosen to stay separable within their own
# panel.
CT_COLOR = {
    "Tumor": "#7A1F2B", "Ciliated": "#E377C2",
    "T/NK": "#1F5FA8", "B": "#8CB9E3", "Plasma": "#17BECF", "Lymphoid": "#4C8FD0",
    "Myeloid": "#E8743B", "Neutrophil": "#F2B447", "Mast": "#9C7A1C",
    "Fibroblast": "#2E8B57", "Stroma": "#5E9E6E", "Smooth muscle": "#A6C46A",
    "Endothelial": "#9467BD",
    "Stem/TA": "#3B6FB6", "Enterocyte": "#F09A3E", "Goblet": "#4FA66A",
    "Paneth": "#D7263D", "Enteroendocrine": "#A56CC1", "Tuft": "#E6C229",
    "PT": "#C49C3F", "TAL": "#8C564B", "PC": "#F29C9C", "IC": "#C71585",
    "Podocyte": "#20B2AA",
}
UNSCORED = "#D9D9D9"

# Legend order per setting: tumour/epithelium first, then immune, then stroma.
ORDER = ["Tumor", "Ciliated", "Stem/TA", "Enterocyte", "Goblet", "Paneth",
         "Enteroendocrine", "Tuft", "PT", "TAL", "PC", "IC", "Podocyte",
         "T/NK", "Lymphoid", "B", "Plasma", "Myeloid", "Neutrophil", "Mast",
         "Fibroblast", "Stroma", "Smooth muscle", "Endothelial"]


def harmonize(setting: str, label: str) -> str:
    if label not in SYNONYM:
        raise KeyError(f"{setting}: unmapped RCTD label {label!r}")
    return SYNONYM[label]
