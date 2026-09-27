#!/usr/bin/env python
"""r05 - main RCC figure: orthogonal protein validation of TRACER partial cells.

c-e one ROI, chosen by the prespecified capture-ratio rule (r04), one row per
    displayed lineage (Myeloid, then the ROI's next two lineages by supported
    partial footprint): that lineage's marker, the 2 um territory of that
    lineage's whole 10x cells, the marker residual outside it, and the residual
    with that lineage's TRACER partial-cell footprints
a   whole-section DAPI overview with the ROI boxed and its coordinates
b   composite mIF of every lineage marker present in the ROI (+ DAPI)
f   per-lineage effect size - does an entity score higher on its OWN lineage
    marker than on another lineage's, measured on the same pixels
g   the fraction of matched-marker positivity that survives outside the 0-2 um
    optical-bleed zone
h   whole against partial: the lineages that fail in partials are the lineages
    whose marker also fails in whole cells
i   every null on one axis
j   cell type x channel specificity in partial cells, row-centred so the
    brightness term shared by all channels is removed
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F
from rcc_defs import CANON, TYPES, MARKERS, AF, DISPLAY

FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ, TAB, PAN = FINAL / "objects", FINAL / "tables", FINAL / "panels"
SPILL_UM = 2.0
FOCAL = "Myeloid"          # single displayed lineage in the ROI panels
RCCA = Path("/scratch4/adeshpa6/xenium_rcc/rcc_analysis")
BOUNDS = RCCA / "masks/cell_boundaries.parquet"
MASTER = RCCA / "label_transfer/master_cells.parquet"


def territory_edt(name, m, shape2, types=None, margin_um=12.0):
    """Distance (um) to the nearest whole 10x cell on the ROI's level-2 grid.

    Rasterised exactly as the global edt_L2 (whole-cell polygons, PIL fill,
    Euclidean distance transform), but only from cells whose transferred
    label is in `types` - the lineages displayed in that ROI.  types=None
    uses every whole cell and reproduces the global map.  A margin keeps cells
    just outside the crop in the transform.  Cached per ROI and lineage set.
    """
    key = "all" if types is None else "+".join(sorted(types)).replace(" ", "_")
    cache = OBJ / "roi_crops" / name / f"edt_L2_{key}.npy"
    if cache.exists():
        return np.load(cache)
    um = m["um_per_px_edt"]
    H, W = shape2
    r0, c0 = int(m["y0"] / um), int(m["x0"] / um)
    M = int(np.ceil(margin_um / um))
    x_lo, x_hi = (c0 - M) * um, (c0 + W + M) * um
    y_lo, y_hi = (r0 - M) * um, (r0 + H + M) * um
    pad = 40.0
    flt = [("entity_class", "==", "whole"),
           ("centroid_x", ">=", x_lo - pad), ("centroid_x", "<=", x_hi + pad),
           ("centroid_y", ">=", y_lo - pad), ("centroid_y", "<=", y_hi + pad)]
    ms = pq.read_table(MASTER, columns=["cell_id", "transferred_label"],
                       filters=flt).to_pandas()
    if types is not None:
        ms = ms[ms.transferred_label.isin(list(types))]
    keep = set(ms.cell_id.astype(str))
    pad2 = pad + 40.0
    cb = pq.read_table(BOUNDS, columns=["cell_id", "vertex_x", "vertex_y"],
                       filters=[("vertex_x", ">=", x_lo - pad2),
                                ("vertex_x", "<=", x_hi + pad2),
                                ("vertex_y", ">=", y_lo - pad2),
                                ("vertex_y", "<=", y_hi + pad2)]).to_pandas()
    cb = cb[cb.cell_id.astype(str).isin(keep)]
    img = Image.new("L", (W + 2 * M, H + 2 * M), 0)
    dr = ImageDraw.Draw(img)
    for _, g in cb.groupby("cell_id", sort=False):
        dr.polygon(list(zip(g.vertex_x.to_numpy() / um - (c0 - M),
                            g.vertex_y.to_numpy() / um - (r0 - M))), fill=1)
    mask = np.asarray(img, bool)
    edt = (ndi.distance_transform_edt(~mask) * um).astype(np.float32)
    edt = edt[M:M + H, M:M + W]
    np.save(cache, edt)
    return edt


def norm(img, ped, lo=1.0, hi=99.5, gamma=0.55):
    v = np.maximum(img.astype(np.float32) - ped, 0)
    a, b = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((v - a) / max(b - a, 1), 0, 1) ** gamma


N_ROWS = 3                 # displayed lineages per ROI: FOCAL, then the next two
BOX_C = "#F0E442"          # ROI box / callout on the tissue overview (CB-safe yellow)
PROT = RCCA / "protein"


def overview_dapi(level=5):
    """Whole-section DAPI at pyramid `level` (cached), and its um per pixel."""
    man = json.loads((PROT / "protein_manifest.json").read_text())
    um = man["pixel_um_level0"] * 2 ** level
    cache = OBJ / "roi_crops" / f"overview_dapi_L{level}.npy"
    if not cache.exists():
        import tifffile
        tf = tifffile.TiffFile(str(PROT / "tif" / man["markers"]["dapi"]
                                   ["channel_file"]), is_ome=False)
        np.save(cache, np.asarray(tf.series[0].levels[level].asarray()))
        tf.close()
    return np.load(cache), um


def roi_types(fp):
    """FOCAL first, then the next lineages by supported-partial footprint."""
    present = [t for t in fp.cell_type.value_counts().index if t in CANON]
    rest = [t for t in present if t != FOCAL]
    return ([FOCAL] if FOCAL in present else []) + rest[:N_ROWS - 1]


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def roi_block(fig, spec, thr, name="roi1", letters="abcde", scalebar=True):
    """One ROI: a row per displayed lineage (its marker | its 10x territory |
    residual | TRACER partials), plus a tissue overview with the ROI boxed and
    a composite mIF of every lineage marker present in the ROI.

    Territory logic is unchanged from the single-lineage version: in each row
    the 10x territory is the 2 um zone around whole 10x cells OF THAT ROW'S
    LINEAGE ONLY, and the residual is that row's marker outside it."""
    d = OBJ / "roi_crops" / name
    meta = json.loads((TAB / "r04_roi_selected.json").read_text())
    m = [x for x in meta["selected"] if x["name"] == name][0]
    fp = pd.read_parquet(d / "footprints.parquet")
    order = roi_types(fp)
    shape2 = np.load(d / "edt_L2.npy").shape
    layers = {}
    for t in [*order, *[u for u in m["types"].split(",") if u in CANON]]:
        if t in layers:
            continue
        mk = CANON[t]
        layers[t] = norm(np.load(d / f"{mk}.npy"), thr["markers"][mk]["pedestal"])
    h1, w1 = next(iter(layers.values())).shape

    def to_l1(mask2):
        u = np.kron(mask2, np.ones((2, 2), bool))[:h1, :w1]
        return np.pad(u, ((0, h1 - u.shape[0]), (0, w1 - u.shape[1])))

    def colour(t, im):
        col = np.array(matplotlib.colors.to_rgb(F.CT_COLORS[t]), np.float32)
        return im[..., None] * col[None, None, :]

    outer = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.16,
                                    width_ratios=[4.0, 1.45])
    rows = GridSpecFromSubplotSpec(len(order), 4, subplot_spec=outer[0, 0],
                                   wspace=0.05, hspace=0.16)
    for r, t in enumerate(order):
        mk = CANON[t]
        rgb = np.clip(colour(t, layers[t]), 0, 1)
        own = territory_edt(name, m, shape2, types=[t]) <= SPILL_UM
        claimed = to_l1(own)
        plural = {"T cell": "T cells", "B cell": "B cells",
                  "Plasma cell": "Plasma cells", "Mast cell": "Mast cells"}
        labs = [DISPLAY[mk], f"In-plane {plural.get(t, t)}", "Residual",
                "TRACER partials"]
        axes = []
        for i, lab in enumerate(labs):
            ax = fig.add_subplot(rows[r, i]); axes.append(ax)
            if i == 0:
                ax.imshow(rgb, interpolation="nearest")
            elif i == 1:
                sh = rgb.copy(); sh[claimed] = sh[claimed] * 0.22 + 0.55
                ax.imshow(sh, interpolation="nearest")
                ax.contour(claimed.astype(float), levels=[0.5],
                           colors=[F.CT_COLORS[t]], linewidths=0.5)
            else:
                r2 = rgb.copy(); r2[claimed] = 0
                ax.imshow(r2, interpolation="nearest")
                if i == 3:
                    s = fp[fp.cell_type == t]
                    mm = np.zeros(shape2, bool)
                    mm[np.clip(s.py, 0, shape2[0] - 1),
                       np.clip(s.px, 0, shape2[1] - 1)] = True
                    mm = to_l1(ndi.binary_dilation(mm, iterations=2))
                    ax.contour(mm.astype(float), levels=[0.5],
                               colors=["white"], linewidths=0.55)
            _clean(ax)
            ax.set_xlabel(lab, labelpad=1.2, fontsize=5.6)
        axes[0].set_ylabel(t, fontsize=6.2, color=F.CT_COLORS[t],
                           fontweight="bold", labelpad=2)
        if scalebar and r == 0:
            npx = 25.0 / m["um_per_px_crop"]
            axes[0].plot([w1 * 0.06, w1 * 0.06 + npx], [h1 * 0.93] * 2, "-",
                         color="white", lw=1.4)
            axes[0].text(w1 * 0.06 + npx / 2, h1 * 0.89, "25 µm",
                         color="white", ha="center", va="bottom", fontsize=5.0)
        F.panel(axes[0], letters[2 + r], dx=-0.30, dy=0.98)

    # ---- overview + composite (right column) ---------------------------
    ov, um_ov = overview_dapi()
    H, W = ov.shape
    side = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1],
                                   hspace=0.22, height_ratios=[0.62, 1.0])
    ax = fig.add_subplot(side[0, 0])
    v = ov.astype(np.float32)
    lo, hi = np.percentile(v[v > 0], [1, 99.5]) if (v > 0).any() else (0, 1)
    ax.imshow(np.clip((v - lo) / max(hi - lo, 1), 0, 1) ** 0.6, cmap="gray",
              extent=[0, W * um_ov / 1000, H * um_ov / 1000, 0],
              interpolation="nearest")
    x0, y0, w, h = m["x0"] / 1000, m["y0"] / 1000, m["w"] / 1000, m["h"] / 1000
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=BOX_C, lw=0.8))
    ax.add_patch(plt.Circle((x0 + w / 2, y0 + h / 2), 0.45, fill=False,
                            ec=BOX_C, lw=0.6))
    ax.set_xlim(0, W * um_ov / 1000); ax.set_ylim(H * um_ov / 1000, 0)
    ax.tick_params(labelsize=4.6, length=1.5, pad=1)
    ax.set_xlabel(f"x {m['x0']:,.0f}–{m['x0'] + m['w']:,.0f} µm, "
                  f"y {m['y0']:,.0f}–{m['y0'] + m['h']:,.0f} µm",
                  fontsize=5.2, labelpad=1.5)
    ax.set_ylabel("mm", fontsize=5.0, labelpad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    F.panel(ax, letters[0], dx=-0.13, dy=1.04)

    ax = fig.add_subplot(side[1, 0])
    comp = [u for u in layers]
    rgb = np.zeros((h1, w1, 3), np.float32)
    dapi = norm(np.load(d / "dapi.npy"), thr["markers"].get("dapi", {})
                .get("pedestal", 0.0))
    rgb = np.maximum(rgb, dapi[..., None] * 0.28)
    for t in comp:
        rgb = np.maximum(rgb, colour(t, layers[t]))
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    _clean(ax)
    ax.set_xlabel("Composite mIF", labelpad=1.2, fontsize=5.6)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=3.2,
                              color=F.CT_COLORS[t],
                              label=DISPLAY[CANON[t]]) for t in comp]
              + [Line2D([], [], marker="s", ls="", ms=3.2, color="#9A9A9A",
                        label="DAPI")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=5.0, frameon=False, handletextpad=0.2,
              columnspacing=0.7, borderaxespad=0.0)
    ax.text(0.04, 0.04, f"capture ×{m['capture_ratio']:.1f}",
            transform=ax.transAxes, color="white", fontsize=5.0, va="bottom")
    F.panel(ax, letters[1], dx=-0.12, dy=1.02)
    return order, comp


def main() -> int:
    F.use_style(7.0)
    PAN.mkdir(parents=True, exist_ok=True)
    thr = json.loads((OBJ / "rcc_marker_thresholds.json").read_text())
    C = pd.read_csv(TAB / "r03_matched_marker_contrast.csv")
    S = pd.read_csv(TAB / "r03b_specificity_centred.csv")
    ES = pd.read_csv(TAB / "r03b_effect_sizes.csv").set_index("cell_type")
    RET = pd.read_csv(TAB / "r03c_spillover_retention.csv").set_index("cell_type")
    NE = pd.read_csv(TAB / "r03c_null_effect_sizes.csv")
    hl = json.loads((TAB / "r03b_headline.json").read_text())

    fig = plt.figure(figsize=(7.2, 9.0))
    g_roi = GridSpec(1, 1, figure=fig, left=0.075, right=0.975,
                     top=0.985, bottom=0.572)
    g_mid = GridSpec(1, 4, figure=fig, left=0.075, right=0.975,
                     top=0.482, bottom=0.302, wspace=0.52)
    gs = GridSpec(1, 1, figure=fig, left=0.075, right=0.975,
                  top=0.242, bottom=0.084)
    order, _ = roi_block(fig, g_roi[0, 0], thr, name="roi1", letters="abcde")

    q = C[(C.view == "all_pixels") & (C.subset == "all")
          & (C.cell_type != "ALL")]
    ordb = sorted(TYPES, key=lambda t: -float(
        q[(q.cell_type == t) & (q.entity_class == "partial")]
        .cliffs_delta_vs_null.iloc[0]))
    ylab = [f"{t}  ({DISPLAY[CANON[t]]})" for t in ordb]
    y = np.arange(len(ordb))[::-1]

    # ---- b: per-lineage effect size --------------------------------------
    ax = fig.add_subplot(g_mid[0, 0])
    for j, t in enumerate(ordb):
        for cls, off, col in (("whole", 0.20, F.C_WHOLE),
                              ("partial", -0.20, F.C_PARTIAL)):
            r = q[(q.cell_type == t) & (q.entity_class == cls)].iloc[0]
            v = float(r.cliffs_delta_vs_null)
            ax.barh(y[j] + off, v, 0.36, color=col, lw=0)
            st = F.stars(r.q_bh)
            if st != "ns":
                ax.text(v + (0.025 if v >= 0 else -0.025), y[j] + off, st,
                        va="center", fontsize=4.6,
                        ha="left" if v >= 0 else "right")
    ax.axvline(0, color="#111111", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=5.8)
    ax.set_xlim(-0.25, 1.20)
    ax.set_xlabel("Own-lineage protein advantage (Cliff's δ)")
    ax.legend(handles=[Line2D([], [], color=F.C_WHOLE, lw=3, label="Whole"),
                       Line2D([], [], color=F.C_PARTIAL, lw=3, label="Partial")],
              loc="lower right", fontsize=5.8)
    F.panel(ax, "f", dx=-0.60)

    # ---- c: spillover retention ------------------------------------------
    ax = fig.add_subplot(g_mid[0, 1])
    for j, t in enumerate(ordb):
        if t not in RET.index:
            continue
        ax.barh(y[j] + 0.20, RET.loc[t, "whole"], 0.36, color=F.C_WHOLE, lw=0)
        ax.barh(y[j] - 0.20, RET.loc[t, "partial"], 0.36, color=F.C_PARTIAL,
                lw=0)
    ax.axvline(1.0, color="#111111", ls="--", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([t for t in ordb], fontsize=5.6)
    ax.set_xlabel("Positivity retained\noutside 0–2 µm")
    F.panel(ax, "g", dx=-0.16)

    # ---- d: whole against partial ----------------------------------------
    ax = fig.add_subplot(g_mid[0, 2])
    ax.axhline(0, color="#CCCCCC", lw=0.5); ax.axvline(0, color="#CCCCCC", lw=0.5)
    for t in TYPES:
        ax.plot(ES.loc[t, "whole"], ES.loc[t, "partial"], "o",
                color=F.CT_COLORS[t], ms=4.4, mec="#333333", mew=0.35)
        dx_, dy_ = (4, -1.5)
        if t == "Mural":
            dx_, dy_ = (5, -5.5)
        elif t == "Tumor":
            dx_, dy_ = (5, 1.5)
        ax.annotate(DISPLAY[CANON[t]], (ES.loc[t, "whole"], ES.loc[t, "partial"]),
                    textcoords="offset points", xytext=(dx_, dy_), fontsize=5.0,
                    color="#222222")
    ax.set_xlim(0.05, 1.30); ax.set_ylim(-0.25, 0.85)
    ax.set_xlabel("Whole cells (Cliff's δ)")
    ax.set_ylabel("Partial cells (Cliff's δ)")
    ax.text(0.04, 0.96, f"ρ = {hl['whole_vs_partial_spearman_rho']:.2f}, "
            f"P = {hl['whole_vs_partial_spearman_p']:.3f}",
            transform=ax.transAxes, fontsize=5.6, va="top")
    F.panel(ax, "h", dx=-0.40)

    # ---- e: nulls --------------------------------------------------------
    ax = fig.add_subplot(g_mid[0, 3])
    ap_ = NE[(NE.view == "all_pixels") & (NE.entity_class == "partial")].iloc[0]
    sf_ = NE[(NE.view == "spillover_free")
             & (NE.entity_class == "partial")].iloc[0]
    dis = C[(C.view == "all_pixels") & (C.subset == "discordant_neighbour")
            & (C.cell_type == "ALL") & (C.entity_class == "partial")]
    vals = [("Observed", float(ap_.cliffs_observed), F.C_PARTIAL),
            (">2 µm only", float(sf_.cliffs_observed), F.C_PARTIAL)]
    if len(dis):
        vals.append(("Discordant\nneighbour",
                     float(dis.cliffs_delta_vs_null.iloc[0]), F.C_PARTIAL))
    vals += [("Swapped\nmarker map", float(ap_.cliffs_swapped_map), F.C_NULL),
             ("Auto-\nfluorescence", float(ap_.cliffs_autofluor), F.C_NULL)]
    x = np.arange(len(vals))
    ax.bar(x, [v[1] for v in vals], 0.68, color=[v[2] for v in vals], lw=0)
    ax.axhline(0, color="#111111", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in vals], fontsize=5.2, rotation=40,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("Own-lineage advantage\n(Cliff's δ)")
    F.panel(ax, "i", dx=-0.40)

    # ---- f: row-centred specificity matrix -------------------------------
    ax = fig.add_subplot(gs[0, 0])
    mk = list(MARKERS) + list(AF)
    s = S[S.entity_class == "partial"]
    cts = [t for t in TYPES if len(s[s.cell_type == t])]
    if len(s[s.cell_type == "Mast cell"]):
        cts.append("Mast cell")
    M = np.full((len(cts), len(mk)), np.nan); Q = np.full_like(M, np.nan)
    for i, t in enumerate(cts):
        for j, m in enumerate(mk):
            r = s[(s.cell_type == t) & (s.marker == m)]
            if len(r):
                M[i, j] = r.rel_log2.iloc[0]; Q[i, j] = r.q_bh.iloc[0]
    vmax = float(np.nanpercentile(np.abs(M), 96))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    for i, t in enumerate(cts):
        if t in CANON:
            j = mk.index(CANON[t])
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   ec="#111111", lw=1.3))
    for i in range(len(cts)):
        for j in range(len(mk)):
            if F.stars(Q[i, j]) != "ns" and abs(M[i, j]) > 0.6:
                ax.text(j, i, "•", ha="center", va="center", fontsize=5.5,
                        color="#111111")
    ax.set_xticks(range(len(mk)))
    ax.set_xticklabels([DISPLAY[m] for m in mk], rotation=90, fontsize=5.2)
    ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts, fontsize=6)
    ax.axvline(len(MARKERS) - 0.5, color="#111111", lw=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.006)
    cb.set_label("Specificity (row-centred log₂)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4)
    F.panel(ax, "j", dx=-0.048, dy=1.02)

    F.save(fig, PAN / "Fig_RCC_protein_validation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
