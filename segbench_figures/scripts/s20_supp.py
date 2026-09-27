"""SegBench v2 supplementary figures.

  S1  availability / failure matrix, 12 settings x methods
  S2  biological fidelity: scRNA concordance (Kendall, Pearson, Spearman) and
      marker specificity (log2FC)
  S3  compute cost: wall-clock, peak host RSS, Segger GPU VRAM
  S4  q25 / q50 ROIs in the main-figure layout
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import figlib as fl
from figlib import fs
from settings import MAIN_ORDER, OBJ, PANELS_SUPP, RUNS_DIR, SETTINGS
import s10_main_figure as mainfig

TAB = OBJ / "tables"
ROW_ORDER = ["xenium_lung__whole", "xenium5k_cervical__q25", "xenium5k_cervical__q50",
             "xenium5k_cervical__q75", "cosmx_nsclc__q25", "cosmx_nsclc__q50",
             "cosmx_nsclc__q75", "merfish_mouse_ileum__whole", "atera_cervical__q25",
             "atera_cervical__q50", "atera_cervical__q75", "visiumhd_kidney__whole"]
PLAT_MARK = {"Xenium": "o", "Xenium5K": "s", "CosMx": "^", "MERFISH": "D",
             "Atera": "v", "VisiumHD": "P"}
# evaluator method keys -> display (kidney also carries the mislabelled
# 'tracer_seq', which is TRACER no-seg: its RCTD input is methods/tracer_noseg)
KEY = {"baysor": "Baysor", "proseg": "ProSeg", "split": "SPLIT", "segger": "Segger",
       "celladmix": "CellAdmix", "bin2cell": "Bin2Cell", "tracer_noseg": "TRACER no-seg",
       "tracer_seq": "TRACER no-seg", "tracer": "TRACER", "tracer_seg": "TRACER"}


def row_label(s):
    ds, plat, roi, _, _ = SETTINGS[s]
    tissue = {"xenium_lung": "lung", "xenium5k_cervical": "cervix", "cosmx_nsclc": "lung",
              "merfish_mouse_ileum": "ileum", "atera_cervical": "cervix",
              "visiumhd_kidney": "kidney"}[ds]
    return f"{fl.HEADER[ds][0]} {tissue} · {fl.ROI_LABEL[roi].replace(' ROI', '')}"


# ---------------------------------------------------------------- S1 --------
def s1_availability():
    """Minimal availability grid: fill = outcome; text only where a failure has a
    cause worth reading (Segger gene floor, CellAdmix OOM)."""
    fs.use_style(6.5)
    st = pd.read_csv(TAB / "run_status.tsv", sep="\t")
    cols = ["TRACER", "Baysor", "ProSeg", "SPLIT", "Segger", "CellAdmix",
            "Bin2Cell", "TRACER no-seg"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.subplots_adjust(left=0.19, right=0.995, top=0.90, bottom=0.10)
    FILL = {"success": "#D4EAD9", "failed": "#F3C4BF", "n/a": "#EFEFEF"}
    for r, s in enumerate(ROW_ORDER):
        y = len(ROW_ORDER) - 1 - r
        for c, m in enumerate(cols):
            q = st[(st.setting == s) & (st.method == m)]
            kind, txt = "n/a", ""
            if not q.empty and q.iloc[0].status in ("success", "failed"):
                q = q.iloc[0]; kind = q.status
                if kind == "failed" and "gene floor" in q.note:
                    txt = "gene floor\n" + q.note.split(": ")[1].split(" < ")[0]
                elif kind == "failed" and "OOM" in q.note:
                    txt = "OOM >512 GiB"
            ax.add_patch(Rectangle((c + 0.04, y + 0.06), 0.92, 0.88, fc=FILL[kind], ec="none"))
            if txt:
                ax.text(c + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=5.2)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(ROW_ORDER))
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, fontsize=6); ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(ROW_ORDER)) + 0.5)
    ax.set_yticklabels([row_label(s) for s in ROW_ORDER[::-1]], fontsize=6)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    hs = [Rectangle((0, 0), 1, 1, fc=FILL[k], ec="none") for k in ("success", "failed", "n/a")]
    fig.legend(hs, ["Completed", "Failed", "Not applicable"], loc="lower center", ncol=3,
               fontsize=5.8, bbox_to_anchor=(0.59, 0.0))
    fs.save(fig, PANELS_SUPP / "Supp_S1_availability.png")


# ---------------------------------------------------------------- data ------
def bio_table():
    rows = []
    for s in ROW_ORDER:
        pr = pd.read_csv(RUNS_DIR / s / "evaluation/plot_ready_table.tsv", sep="\t")
        for k, g in pr.groupby("method"):
            if KEY.get(k) in (None, "TRACER"):
                continue                               # TRACER is taken stratified below
            v = g.set_index("metric")["value"]
            for m in ("kendall_tau_median", "pearson_r_median", "spearman_rho_median",
                      "marker_logfc_median"):
                if m in v and pd.notna(v[m]):
                    rows.append({"setting": s, "method": KEY[k], "metric": m, "value": float(v[m])})
        b = TAB / "bio" / f"bio__{s}.tsv"
        if b.exists():
            bb = pd.read_csv(b, sep="\t")
            for _, r in bb.iterrows():
                for m in ("kendall_tau_median", "pearson_r_median", "spearman_rho_median",
                          "marker_logfc_median"):
                    if pd.notna(r[m]):
                        rows.append({"setting": s, "method": r.method, "metric": m,
                                     "value": float(r[m])})
    d = pd.DataFrame(rows).drop_duplicates(["setting", "method", "metric"])
    d.to_csv(TAB / "bio_metrics_all.tsv", sep="\t", index=False)
    return d


def resource_table():
    rows = []
    for s in ROW_ORDER:
        ru = pd.read_csv(RUNS_DIR / s / "evaluation/resource_usage.tsv", sep="\t")
        for _, r in ru.iterrows():
            rows.append({"setting": s, "method": KEY.get(r.method, r.method),
                         "wall_h": r.wall_clock_seconds / 3600,
                         "host_rss_gib": r.peak_host_rss_gb,
                         "host_req_gib": r.host_memory_requested_gb,
                         "gpu_vram_gib": (r.peak_gpu_vram_mb / 1024
                                          if pd.notna(r.get("peak_gpu_vram_mb")) else np.nan)})
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "resource_all.tsv", sep="\t", index=False)
    return d


def dotplot(ax, d, methods, value, log=False, xlabel="", quiet_empty=False,
            flag=None):
    n = len(methods)
    for i, m in enumerate(methods):
        y = n - 1 - i
        ax.axhline(y, color="#EEEEEE", lw=0.5, zorder=0)
        g = d[d.method == m]
        if g.empty:
            if quiet_empty:
                continue
            ax.text(0.02, y, "not evaluated" if m == "Original" else "–",
                    transform=ax.get_yaxis_transform(), fontsize=5, color="#8C8C8C",
                    style="italic", va="center")
            continue
        for _, r in g.iterrows():
            plat = SETTINGS[r.setting][1]; main = r.setting in MAIN_ORDER
            col = fl.METHOD_COLOR.get(m, "#444444")
            ax.scatter(r[value], y + (0.0 if main else -0.18), marker=PLAT_MARK[plat],
                       s=16 if main else 9, fc=col if main else "white", ec=col,
                       lw=0.6, zorder=3)
            if flag is not None and flag(r):
                ax.scatter(r[value], y + (0.0 if main else -0.18), marker="x", s=10,
                           c="black", lw=0.6, zorder=5)
        med = g.loc[g.setting.isin(MAIN_ORDER), value].median()
        if np.isfinite(med):
            ax.plot([med, med], [y - 0.32, y + 0.32], color="black", lw=0.9, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels(methods[::-1])
    ax.set_ylim(-0.6, n - 0.4)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)


def platform_legend(fig, y, extra=()):
    hs = [Line2D([], [], marker=mk, ls="", ms=3.6, mfc="#555555", mec="#555555", label=p)
          for p, mk in PLAT_MARK.items()]
    hs += [Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec="#555555",
                  label="q25/q50 ROI (open)"),
           Line2D([], [], color="black", lw=0.9, label="median, main settings")]
    hs += list(extra)
    fig.legend(handles=hs, loc="lower center", ncol=min(len(hs), 9), fontsize=5.4,
               bbox_to_anchor=(0.55, y), handletextpad=0.3, columnspacing=1.0)


# ---------------------------------------------------------------- S2 --------
def canonical_table():
    """Canonical-marker log2FC (s08) in the long format dotplot() reads."""
    fr = [pd.read_csv(f, sep="\t") for f in sorted((TAB / "canonical").glob("canonical__*.tsv"))]
    c = pd.concat(fr, ignore_index=True)
    c.to_csv(TAB / "canonical_markers_all.tsv", sep="\t", index=False)
    return c.rename(columns={"canonical_marker_log2fc": "v"})[["setting", "method", "v"]].dropna()


def s2_bio(d):
    """Canonical-marker log2FC only (s08: frozen lineage panel, true log2).

    Reference concordance is deliberately not drawn: the split is cell-level with
    shared donors and TRACER's prior saw evaluation cells on cervical and kidney
    (BIAS_AUDIT_concordance_markers.md).  Its values stay in
    tables/segbench_v2/bio_metrics_all.tsv.  ``d`` is still written out by
    bio_table() so those tables remain current.
    """
    fs.use_style(6.5)
    methods = ["Original", "TRACER whole", "TRACER partial", "TRACER pooled", "Baysor",
               "ProSeg", "SPLIT", "Segger", "CellAdmix", "Bin2Cell", "TRACER no-seg"]
    fl.METHOD_COLOR.setdefault("TRACER pooled", "#5D3A9B")
    c = pd.read_csv(TAB / "canonical_markers_all.tsv", sep="\t")
    ok = c.dropna(subset=["canonical_marker_log2fc"]).rename(
        columns={"canonical_marker_log2fc": "v"})
    na = c[c.canonical_marker_log2fc.isna()]
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0.14, 0.20, 0.60, 0.76])
    dotplot(ax, ok, methods, "v", xlabel="Canonical marker log2FC (cell type vs other types)")
    ax.axvline(0, color="#BBBBBB", lw=0.5, zorder=0)
    # NA strip: settings where the method ran but too few canonical markers survive
    axn = fig.add_axes([0.77, 0.20, 0.21, 0.76], sharey=ax)
    axn.set_xlim(0, 1); axn.axis("off")
    n = len(methods)
    axn.text(0.0, n - 0.1, "NA: too few canonical\nmarkers in panel/output", fontsize=5.6,
             va="bottom", ha="left", color="#555555")
    for i, m in enumerate(methods):
        y = n - 1 - i
        g = na[na.method == m]
        for j, s_ in enumerate(sorted(g.setting, key=ROW_ORDER.index)):
            plat = SETTINGS[s_][1]; main = s_ in MAIN_ORDER
            axn.scatter(0.05 + 0.09 * j, y, marker=PLAT_MARK[plat], s=16 if main else 9,
                        fc="#9A9A9A" if main else "white", ec="#9A9A9A", lw=0.6)
        if len(g):
            axn.text(0.05 + 0.09 * len(g) + 0.02, y, "NA", fontsize=5.4, color="#777777",
                     va="center")
    fs.panel(ax, "", dx=0)
    platform_legend(fig, 0.0)
    fs.save(fig, PANELS_SUPP / "Supp_S2_concordance_markers.png")


# ---------------------------------------------------------------- S3 --------
def s3_cost(d):
    fs.use_style(6.5)
    methods = ["TRACER", "Baysor", "ProSeg", "SPLIT", "Segger", "CellAdmix",
               "Bin2Cell", "TRACER no-seg"]
    fl.METHOD_COLOR.setdefault("TRACER", fl.METHOD_COLOR["TRACER whole"])
    fig = plt.figure(figsize=(7.2, 2.7))
    a1 = fig.add_axes([0.12, 0.26, 0.36, 0.70])
    a2 = fig.add_axes([0.52, 0.26, 0.28, 0.70], sharey=a1)
    a3 = fig.add_axes([0.84, 0.26, 0.14, 0.70])        # own axis: sharey would strip A's labels
    dotplot(a1, d, methods, "wall_h", log=True, xlabel="Wall-clock (h)")
    # RSS is summed over the process tree, so shared pages are counted once per
    # forked worker; a value above the job's own allocation cannot be real.
    dotplot(a2, d, methods, "host_rss_gib", log=True, xlabel="Peak host RSS (GiB)",
            flag=lambda r: r.host_rss_gib > r.host_req_gib)
    plt.setp(a2.get_yticklabels(), visible=False)
    seg = d[d.method == "Segger"].dropna(subset=["gpu_vram_gib"])
    dotplot(a3, seg, methods, "gpu_vram_gib", xlabel="GPU VRAM (GiB)", quiet_empty=True)
    a3.set_ylim(a1.get_ylim()); a3.tick_params(axis="y", left=False, labelleft=False)
    a3.set_xlim(0, 24); a3.set_xticks([0, 12, 24])
    for ax, l, dx in ((a1, "A", -0.30), (a2, "B", -0.06), (a3, "C", -0.55)):
        fs.panel(ax, l, dx=dx)
    platform_legend(fig, 0.0, extra=[Line2D([], [], marker="x", ls="", ms=3.6, mec="black",
                                            label="RSS > allocation (over-count)")])
    fs.save(fig, PANELS_SUPP / "Supp_S3_runtime_memory.png")


if __name__ == "__main__":
    PANELS_SUPP.mkdir(parents=True, exist_ok=True)
    s1_availability()
    s2_bio(bio_table())
    s3_cost(resource_table())
    mainfig.build(["xenium5k_cervical__q25", "xenium5k_cervical__q50",
                   "cosmx_nsclc__q25", "cosmx_nsclc__q50",
                   "atera_cervical__q25", "atera_cervical__q50"],
                  PANELS_SUPP / "Supp_S4_q25_q50_rois")
