"""Panel drawers shared by the SegBench v2 main and supplementary figures.

Each drawer takes an Axes and a setting key and draws exactly one panel, so the
assembled figure, the standalone per-panel exports and the q25/q50 supplement
all come from the same code.  No drawer writes a title: column headers and row
letters are placed by the layout, not inside panels.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

sys.path.insert(0, "/scratch4/adeshpa6/tracer_campaign/final/scripts")
sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs
from labels import CT_COLOR, ORDER, UNSCORED
from settings import OBJ, ROI_DESIGN, SETTINGS

TAB = OBJ / "tables"

# ---- methods: fixed slots and colours shared by every panel -----------------
METHODS = ["Original", "TRACER whole", "TRACER partial", "Baysor", "ProSeg",
           "SPLIT", "Segger", "CellAdmix"]
METHOD_COLOR = {
    "Original": "#8C8C8C", "TRACER whole": "#0072B2", "TRACER partial": "#D55E00",
    "Baysor": "#E69F00", "ProSeg": "#009E73", "SPLIT": "#56B4E9",
    "Segger": "#CC79A7", "CellAdmix": "#999933",
    "Bin2Cell": "#882255", "TRACER no-seg": "#332288",
}
# Visium HD is binned, so its comparison is the platform-native set
VISIUMHD = "visiumhd_kidney__whole"
VISIUMHD_METHODS = ["Original", "TRACER whole", "TRACER partial", "Bin2Cell", "TRACER no-seg"]


def methods_for(setting):
    return VISIUMHD_METHODS if setting == VISIUMHD else METHODS


CALLS = ["singlet", "doublet_certain", "doublet_uncertain", "reject", "unscored"]
CALL_LABEL = {"singlet": "Singlet", "doublet_certain": "Doublet (certain)",
              "doublet_uncertain": "Doublet (uncertain)", "reject": "Reject",
              "unscored": "Unscored (<10 UMI)"}
CALL_COLOR = {"singlet": "#3C5488", "doublet_certain": "#8491B4",
              "doublet_uncertain": "#C5CDE3", "reject": "#E64B35",
              "unscored": "#DCDCDC"}

HEADER = {
    "xenium_lung":         ("Xenium", "Human lung cancer"),
    "xenium5k_cervical":   ("Xenium 5K", "Human cervical cancer"),
    "cosmx_nsclc":         ("CosMx", "Human lung cancer"),
    "merfish_mouse_ileum": ("MERFISH", "Mouse ileum"),
    "atera_cervical":      ("Atera", "Human cervical cancer"),
    "visiumhd_kidney":     ("Visium HD", "Human kidney"),
}
ROI_LABEL = {"whole": "whole tissue", "q25": "q25 ROI", "q50": "q50 ROI", "q75": "q75 ROI"}

_status = None


def status():
    """method availability per setting: success / failed(note) / not applicable."""
    global _status
    if _status is None:
        _status = pd.read_csv(TAB / "run_status.tsv", sep="\t")
    return _status


def missing_note(setting, method):
    s = status()
    r = s[(s.setting == setting) & (s.method == method)]
    if r.empty:
        return None
    r = r.iloc[0]
    if r.status == "not applicable":
        return "n/a"
    if r.status == "failed":
        return ("OOM" if "OOM" in r.note else
                "gene floor" if "gene floor" in r.note else "failed")
    return None


_cells = {}


def cells(setting):
    if setting not in _cells:
        _cells[setting] = pd.read_parquet(TAB / f"percell__{setting}.parquet")
    return _cells[setting]


def roi_box(setting):
    ds, _, roi, _, _ = SETTINGS[setting]
    if roi == "whole":
        return None
    m = json.loads((ROI_DESIGN / "roi_manifest_frozen.json").read_text())
    r = m["datasets"][ds]["rois"][roi]
    return r["xmin_um"], r["xmax_um"], r["ymin_um"], r["ymax_um"]


# ---- row 1: spatial landscape ---------------------------------------------
def landscape_method(setting):
    ch = pd.read_csv(TAB / "landscape_choice.tsv", sep="\t")
    ch = ch[(ch.setting == setting) & ch.chosen]
    return ch.method.iloc[0] if len(ch) else "Original"


def draw_spatial(ax, setting, method=None, scalebar=True):
    method = method or landscape_method(setting)
    d = cells(setting)
    if method == "TRACER":
        # whole + partial together, placed at transcript-mean centroids (s09) and
        # coloured by TRACER's own RCTD dominant type
        d = d[d.method.isin(["TRACER whole", "TRACER partial"])].drop(columns=["x", "y"])
        cen = pd.read_parquet(TAB / f"tracer_centroids__{setting}.parquet")
        d = d.merge(cen[["tracer_id", "x", "y"]], left_on="cell_id", right_on="tracer_id",
                    how="inner")
    else:
        d = d[(d.method == method) & d.x.notna()]
    if d.empty:                                      # upstream fit not finished
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, f"{method}\nnot available", ha="center", va="center",
                fontsize=5, color="#8C8C8C", transform=ax.transAxes)
        return method, []
    box = roi_box(setting)
    if box:
        x0, x1, y0, y1 = box
    else:
        x0, x1 = np.nanpercentile(d.x, [0.2, 99.8]); y0, y1 = np.nanpercentile(d.y, [0.2, 99.8])
        pad = 0.03 * max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)          # image convention: y down
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#666666")
    fig = ax.figure; fig.canvas.draw_idle()
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    area_pt2 = bb.width * bb.height * 72 ** 2
    # marker area so that points roughly tile the panel without saturating it
    s = float(np.clip(0.55 * area_pt2 / max(len(d), 1), 0.02, 6.0))
    un = d[d.celltype.isna()]
    ax.scatter(un.x, un.y, s=s, c=UNSCORED, lw=0, rasterized=True, zorder=1)
    sc = d[d.celltype.notna()]
    ax.scatter(sc.x, sc.y, s=s, c=sc.celltype.map(CT_COLOR).to_numpy(), lw=0,
               rasterized=True, zorder=2)
    if scalebar:
        span = x1 - x0
        L = 100 if span <= 800 else 500 if span <= 4000 else 1000
        lab = f"{L} µm" if L < 1000 else f"{L // 1000} mm"
        bx, by = x0 + 0.06 * span, y1 - 0.06 * (y1 - y0)
        ax.plot([bx, bx + L], [by, by], color="black", lw=1.4, solid_capstyle="butt", zorder=5)
        ax.text(bx, by - 0.025 * (y1 - y0), lab, ha="left", va="bottom",   # stays inside narrow maps
                fontsize=5, zorder=5,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75))
    return method, sorted(sc.celltype.unique(), key=ORDER.index)


def draw_locator(ax, setting):
    """Whole-tissue transcript density (Stage-1 25 µm grid) with the ROI box."""
    ds = SETTINGS[setting][0]
    z = np.load(ROI_DESIGN / f"{ds}_hist.npz")
    h, (ox, oy), b = z["hist"], z["origin"], float(z["bin_um"][0])
    ext = [ox, ox + h.shape[0] * b, oy + h.shape[1] * b, oy]
    img = np.log1p(h.T) / np.log1p(np.percentile(h[h > 0], 99))
    ax.imshow(np.clip(img, 0, 1) * 0.55, cmap="Greys", extent=ext,   # tissue mid-grey
              interpolation="nearest", vmin=0, vmax=1)
    x0, x1, y0, y1 = roi_box(setting)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#E64B35", lw=0.9))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color("#999999")


# ---- rows 2-3: ridgeplots ---------------------------------------------------
def _kde(v, lo, hi, grid):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 5 or np.ptp(v) == 0:
        return None
    rng = np.random.default_rng(20260909)
    if len(v) > 20000:
        v = rng.choice(v, 20000, replace=False)
    k = gaussian_kde(v)
    y = k(grid)
    if lo is not None:
        y += k(2 * lo - grid)                          # reflect at the lower bound
    if hi is not None:
        y += k(2 * hi - grid)                          # and at the upper bound
    return y


def draw_ridges(ax, setting, metric, xlim, bounds, ref_line=True, overlap=1.55,
                methods=None, nslots=None):
    methods = methods or METHODS
    d = cells(setting)
    d = d[d.call != "unscored"]
    grid = np.linspace(*xlim, 400)
    n = nslots or len(methods)          # nslots > len(methods): same slot height, top-aligned
    ax.set_xlim(*xlim); ax.set_ylim(-0.35, n - 1 + overlap + 0.05)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    for i, m in enumerate(methods):
        y0 = n - 1 - i
        ax.axhline(y0, color="#BBBBBB", lw=0.3, zorder=0)
        v = d.loc[d.method == m, metric]
        dens = _kde(v, *bounds, grid) if len(v) else None
        if dens is None:
            note = missing_note(setting, m) or "–"
            ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), y0 + 0.18, note, fontsize=4.8,
                    color="#8C8C8C", style="italic", va="bottom", ha="left", zorder=2 * i + 1)
            continue
        dens = dens / dens.max() * overlap
        ax.fill_between(grid, y0, y0 + dens, color=METHOD_COLOR[m], alpha=0.92,
                        lw=0, zorder=2 * i + 1)
        ax.plot(grid, y0 + dens, color="black", lw=0.35, zorder=2 * i + 2)
        med = float(np.median(v))
        hm = np.interp(med, grid, dens)
        ax.plot([med, med], [y0, y0 + hm], color="black", lw=0.6, zorder=2 * i + 2)
    if ref_line:
        o = d.loc[d.method == "Original", metric]
        if len(o):
            # span only the occupied slots (a short comparison set leaves empty ones)
            lo = n - len(methods) - 0.2
            ax.plot([float(np.median(o))] * 2, [lo, ax.get_ylim()[1]], color="#5A5A5A",
                    lw=0.7, ls=(0, (2.2, 1.4)), zorder=40)


# ---- row 4: RCTD call composition ------------------------------------------
def draw_bars(ax, setting, scale=1e3, methods=None, offset=0.0, nslots=None):
    methods = methods or METHODS
    comp = pd.read_csv(TAB / "rctd_call_composition.tsv", sep="\t")
    comp = comp[comp.setting == setting]
    n = nslots or len(methods)
    tot_max = comp[comp.method.isin(methods)].groupby("method").n.sum().max() / scale
    for i, m in enumerate(methods):
        y = n - 1 - i + offset
        c = comp[comp.method == m].set_index("call").n
        if c.empty or c.sum() == 0:
            note = missing_note(setting, m) or "–"
            ax.text(0.01 * tot_max, y, note, fontsize=4.8, color="#8C8C8C",
                    style="italic", va="center", ha="left")
            continue
        left = 0.0
        for k in CALLS:
            w = c.get(k, 0) / scale
            ax.barh(y, w, left=left, height=0.72, color=CALL_COLOR[k], lw=0)
            left += w
    if not offset:
        ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, tot_max * 1.04)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, integer=False))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def method_key(ax, overlap=None, bars=False, methods=None, nslots=None, fontsize=5.8):
    """Left-hand label column: coloured swatch + method name at each slot."""
    methods = methods or METHODS
    n = nslots or len(methods)
    ax.set_xlim(0, 1)
    if bars:
        ax.set_ylim(-0.6, n - 0.4)
    else:
        ax.set_ylim(-0.35, n - 1 + (overlap or 1.55) + 0.05)
    ax.axis("off")
    for i, m in enumerate(methods):
        y = n - 1 - i + (0 if bars else 0.3)
        ax.add_patch(Rectangle((0.90, y - 0.22), 0.09, 0.44, color=METHOD_COLOR[m],
                               transform=ax.transData, clip_on=False))
        ax.text(0.86, y, m, ha="right", va="center", fontsize=fontsize)
