#!/usr/bin/env python
"""r04 - choose the validation ROIs by prespecified quantitative support.

WHY
---
The published ROI was chosen for "strong-marker balanced diversity" and reads
poorly: the field is so densely packed that a 3 um halo around the 10x
segmentation covers ~94 % of it, so after blackout there is almost nothing left
to see and the partial-cell outlines sit on black.  An ROI that cannot contain
residual protein cannot demonstrate residual protein capture.

Selection rule, fixed BEFORE any image was inspected.  Candidate windows are a
150 um grid at 75 um stride over on-section tissue.  A window is ELIGIBLE if

    R1  residual area fraction    >= 0.15   (pixels > 2 um from any 10x cell)
    R2  supported partial cells   >= 8      (n_px >= 8, n_tx >= 15 and
                                             matched-marker contrast above the
                                             75th percentile of all partials)
    R3  distinct lineages present >= 3

and eligible windows are RANKED by capture ratio: the share of residual
matched-marker-positive pixels that falls inside supported partial footprints,
divided by the share of residual AREA those footprints occupy.  A ratio of 1
means the partials are no better than a random patch of the same size.  The full
ranking is exported so the choice is auditable, and the figure reports where the
chosen windows sit in that distribution.

Outputs: final/tables/r04_roi_candidates.csv, r04_roi_selected.json
         final/objects/roi_crops/<roi>/<marker>.npy  (level 1 display crops)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, tifffile, zarr
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcutil import trim, mklog
from rcc_defs import CANON, TYPES, MARKERS, DISPLAY

RCC   = Path("/scratch4/adeshpa6/xenium_rcc/rcc_analysis")
FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ, TAB = FINAL / "objects", FINAL / "tables"
PROT = RCC / "protein"
REFINED = Path("/scratch4/adeshpa6/xenium_rcc/tracer/run_wt/outputs/"
               "transcripts_tracer_refined.parquet")
WIN_UM, STRIDE_UM = 150.0, 75.0
SPILL_UM = 2.0
R1_RESIDUAL, R2_PARTIALS, R3_TYPES = 0.15, 8, 3
N_STAGE_B = 40
N_SELECT = 3
CROP_LEVEL = 1
log = mklog("r04")


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = json.loads((PROT / "protein_manifest.json").read_text())
    thr = json.loads((OBJ / "rcc_marker_thresholds.json").read_text())
    umpp2 = man["pixel_um_level0"] * 4.0                      # level 2
    umpp3 = man["pixel_um_level0"] * 8.0                      # level 3

    edt_path = RCC / "validation/edt_L2.npy"
    with open(edt_path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        (H2, W2), _, dt = np.lib.format._read_array_header(f, ver)
        off = f.tell()
    # banded read; np.fromfile on the whole .npy would also swallow its header
    H3, W3 = (H2 + 1) // 2, (W2 + 1) // 2
    edt3 = np.empty((H3, W3), dt)
    for r0 in range(0, H2, 2048):
        r1 = min(r0 + 2048, H2)
        blk = np.fromfile(edt_path, dtype=dt, count=(r1 - r0) * W2,
                          offset=off + r0 * W2 * dt.itemsize).reshape(r1 - r0, W2)
        sl = blk[::2, ::2]
        a = r0 // 2
        edt3[a:a + sl.shape[0], :sl.shape[1]] = sl
        del blk, sl
    acc = np.zeros((H3, W3), np.uint8)
    for mk, cut in (("dapi", 30.0), ("panck", 70.0), ("vimentin", 70.0),
                    ("af_green", 70.0)):
        a = np.load(PROT / f"downsampled/{mk}_L3.npy")[:H3, :W3]
        acc += (a > cut).astype(np.uint8); del a
    tissue3 = acc >= 2; del acc
    resid3 = tissue3 & (edt3 > SPILL_UM)
    log(f"L3 grid {H3}x{W3}; tissue {100*tissue3.mean():.1f}%, "
        f"residual (>{SPILL_UM} um from a 10x cell) "
        f"{100*resid3[tissue3].mean():.1f}% of tissue")

    E = pd.read_parquet(OBJ / "rcc_entity_summary.parquet",
                        columns=["entity", "entity_class", "cell_type", "n_px",
                                 "n_tx", "centroid_x", "centroid_y",
                                 "matched_delta"])
    E["cell_type"] = E.cell_type.astype(str)
    E["entity_class"] = E.entity_class.astype(str)
    P = E[(E.entity_class == "partial") & E.cell_type.isin(TYPES)
          & (E.n_px >= 8) & (E.n_tx >= 15)].copy()
    cut = float(np.nanpercentile(P.matched_delta, 75))
    P["supported"] = P.matched_delta >= cut
    SUP = P[P.supported]
    log(f"{len(P):,} candidate partials; matched-contrast 75th pct = {cut:.4f}; "
        f"{len(SUP):,} supported")

    # ---------------- stage A: window scan ---------------------------------
    ys, xs = np.nonzero(tissue3)
    x0g, x1g = xs.min() * umpp3, xs.max() * umpp3
    y0g, y1g = ys.min() * umpp3, ys.max() * umpp3
    del ys, xs
    gx = np.arange(x0g, x1g - WIN_UM, STRIDE_UM)
    gy = np.arange(y0g, y1g - WIN_UM, STRIDE_UM)
    sx = SUP.centroid_x.to_numpy(); sy = SUP.centroid_y.to_numpy()
    st_ = SUP.cell_type.to_numpy()
    rows = []
    for X0 in gx:
        for Y0 in gy:
            c0, c1 = int(X0 / umpp3), int((X0 + WIN_UM) / umpp3)
            r0, r1 = int(Y0 / umpp3), int((Y0 + WIN_UM) / umpp3)
            tw = tissue3[r0:r1, c0:c1]
            if tw.sum() < 0.5 * tw.size:
                continue
            rf = float(resid3[r0:r1, c0:c1].sum() / max(tw.sum(), 1))
            if rf < R1_RESIDUAL:
                continue
            m = (sx >= X0) & (sx < X0 + WIN_UM) & (sy >= Y0) & (sy < Y0 + WIN_UM)
            n = int(m.sum())
            if n < R2_PARTIALS:
                continue
            nt = int(pd.Series(st_[m]).nunique())
            if nt < R3_TYPES:
                continue
            rows.append(dict(x0=float(X0), y0=float(Y0), w=WIN_UM, h=WIN_UM,
                             residual_frac=rf, n_supported=n, n_types=nt,
                             types=",".join(sorted(set(st_[m]))),
                             tissue_frac=float(tw.mean())))
    A = pd.DataFrame(rows)
    log(f"stage A: {len(A):,} eligible windows of "
        f"{len(gx)*len(gy):,} scanned ({time.time()-t0:.0f}s)")
    if not len(A):
        raise SystemExit("no eligible window - relax R1-R3 and re-run")

    # ---------------- footprints of supported partials ---------------------
    keep = set(SUP.entity.astype(str))
    idx = pd.Index(SUP.entity.astype(str))
    pf = pq.ParquetFile(REFINED)
    C, PX, PY = [], [], []
    nb_ = 0
    for bt in pf.iter_batches(batch_size=150_000, columns=["stitched", "x", "y"]):
        dic = pa.compute.dictionary_encode(bt.column("stitched"))
        lut = idx.get_indexer(pd.Index(dic.dictionary.to_pylist()))
        c = lut[np.asarray(dic.indices)]
        m = c >= 0
        if m.any():
            C.append(c[m].astype(np.int32))
            PX.append(np.clip((np.asarray(bt.column("x"), np.float64)[m]
                               / umpp2).astype(np.int32), 0, W2 - 1))
            PY.append(np.clip((np.asarray(bt.column("y"), np.float64)[m]
                               / umpp2).astype(np.int32), 0, H2 - 1))
        del bt, dic, lut, c, m
        nb_ += 1
        if nb_ % 20 == 0:
            trim()
    trim()
    fc = np.concatenate(C); fx = np.concatenate(PX); fy = np.concatenate(PY)
    del C, PX, PY
    ftype = SUP.cell_type.to_numpy()[fc]
    log(f"{len(fc):,} footprint transcripts for supported partials "
        f"({time.time()-t0:.0f}s)")

    # ---------------- stage B: capture ratio for the top windows -----------
    A = A.sort_values(["n_supported", "n_types", "residual_frac"],
                      ascending=False).reset_index(drop=True)
    B = A.head(N_STAGE_B).copy()
    # one channel open at a time: seven simultaneous JPEG2000 readers plus their
    # tile caches do not fit the shared login cgroup
    readers = {}

    def reader(mk):
        if mk not in readers:
            for k in list(readers):
                readers[k][0].close(); del readers[k]
            tf = tifffile.TiffFile(str(PROT / "tif"
                                       / man["markers"][mk]["channel_file"]),
                                   is_ome=False)
            readers[mk] = (tf, zarr.open(tf.series[0].levels[2].aszarr(),
                                         mode="r"))
        return readers[mk][1]
    cap, area, rat = [], [], []
    for _, w in B.iterrows():
        c0, c1 = int(w.x0 / umpp2), int((w.x0 + w.w) / umpp2)
        r0, r1 = int(w.y0 / umpp2), int((w.y0 + w.h) / umpp2)
        res = (np.asarray(np.fromfile(edt_path, dtype=dt,
                                      count=(r1 - r0) * W2,
                                      offset=off + r0 * W2 * dt.itemsize)
                          .reshape(r1 - r0, W2)[:, c0:c1]) > SPILL_UM)
        inw = (fx >= c0) & (fx < c1) & (fy >= r0) & (fy < r1)
        n_res = float(res.sum())
        from scipy import ndimage as ndi
        num = den = wsum = 0.0
        cs = as_ = 0.0
        for t in sorted(set(ftype[inw])):
            mk = CANON[t]
            ped = thr["markers"][mk]["pedestal"]
            th = thr["markers"][mk]["thr_used"]
            img = np.asarray(reader(mk)[r0:r1, c0:c1]).astype(np.float32)
            pos = (img - ped) > th
            fm = np.zeros_like(res)
            sel = inw & (ftype == t)
            fm[fy[sel] - r0, fx[sel] - c0] = True
            fm = ndi.binary_dilation(fm, iterations=2)      # ~1.7 um footprint
            pr = float((pos & res).sum())
            if pr < 20 or n_res <= 0:
                del img, pos, fm
                continue
            cap_t = float((pos & res & fm).sum()) / pr
            area_t = float((res & fm).sum()) / n_res
            if area_t <= 0:
                del img, pos, fm
                continue
            # weight each lineage by how much residual positive signal its own
            # marker actually has in this window
            num += pr * (cap_t / area_t); wsum += pr
            cs += pr * cap_t; as_ += pr * area_t
            del img, pos, fm
        if wsum > 0:
            cap.append(cs / wsum); area.append(as_ / wsum); rat.append(num / wsum)
        else:
            cap.append(np.nan); area.append(np.nan); rat.append(np.nan)
        del res
    B["capture_frac"] = cap; B["area_frac"] = area; B["capture_ratio"] = rat
    for k in list(readers):
        readers[k][0].close(); del readers[k]
    trim()
    B = (B.dropna(subset=["capture_ratio"])
         .sort_values("capture_ratio", ascending=False).reset_index(drop=True))
    B.to_csv(TAB / "r04_roi_candidates.csv", index=False)
    log(f"stage B: capture ratio {B.capture_ratio.min():.2f}-"
        f"{B.capture_ratio.max():.2f} over {len(B)} windows "
        f"({time.time()-t0:.0f}s)")

    # ---------------- export display crops for the selected ROIs -----------
    sel = B.head(N_SELECT).reset_index(drop=True)
    um1 = man["pixel_um_level0"] * 2.0
    outdir = OBJ / "roi_crops"; outdir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, w in sel.iterrows():
        name = f"roi{i+1}"
        d = outdir / name; d.mkdir(exist_ok=True)
        r0, r1 = int(w.y0 / um1), int((w.y0 + w.h) / um1)
        c0, c1 = int(w.x0 / um1), int((w.x0 + w.w) / um1)
        for mk in sorted(set([CANON[t] for t in TYPES]) | {"dapi", "cd45"}):
            tf = tifffile.TiffFile(str(PROT / "tif" / man["markers"][mk]
                                       ["channel_file"]), is_ome=False)
            z = zarr.open(tf.series[0].levels[CROP_LEVEL].aszarr(), mode="r")
            np.save(d / f"{mk}.npy", np.asarray(z[r0:r1, c0:c1]).astype(np.uint16))
            tf.close(); trim()
        # residual mask and footprints at level 2 for overlay
        rr0, rr1 = int(w.y0 / umpp2), int((w.y0 + w.h) / umpp2)
        cc0, cc1 = int(w.x0 / umpp2), int((w.x0 + w.w) / umpp2)
        np.save(d / "edt_L2.npy",
                np.fromfile(edt_path, dtype=dt, count=(rr1 - rr0) * W2,
                            offset=off + rr0 * W2 * dt.itemsize
                            ).reshape(rr1 - rr0, W2)[:, cc0:cc1])
        inw = (fx >= cc0) & (fx < cc1) & (fy >= rr0) & (fy < rr1)
        pd.DataFrame({"entity_row": fc[inw], "px": fx[inw] - cc0,
                      "py": fy[inw] - rr0,
                      "cell_type": ftype[inw]}).to_parquet(d / "footprints.parquet",
                                                           index=False)
        meta.append(dict(name=name, x0=float(w.x0), y0=float(w.y0),
                         w=float(w.w), h=float(w.h),
                         residual_frac=float(w.residual_frac),
                         n_supported=int(w.n_supported),
                         n_types=int(w.n_types), types=w.types,
                         capture_frac=float(w.capture_frac),
                         area_frac=float(w.area_frac),
                         capture_ratio=float(w.capture_ratio),
                         crop_level=CROP_LEVEL, um_per_px_crop=um1,
                         um_per_px_edt=umpp2))
    (TAB / "r04_roi_selected.json").write_text(json.dumps({
        "rule": {"window_um": WIN_UM, "stride_um": STRIDE_UM,
                 "R1_residual_frac_min": R1_RESIDUAL,
                 "R2_supported_partials_min": R2_PARTIALS,
                 "R3_distinct_lineages_min": R3_TYPES,
                 "rank_by": "capture_ratio",
                 "matched_contrast_cut_pct75": cut,
                 "spill_um": SPILL_UM},
        "n_windows_scanned": int(len(gx) * len(gy)),
        "n_eligible": int(len(A)), "n_stage_b": int(len(B)),
        "selected": meta}, indent=2))
    log(f"DONE {time.time()-t0:.0f}s - selected {len(meta)} ROIs, "
        f"capture ratios " + ", ".join(f"{m['capture_ratio']:.2f}" for m in meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
