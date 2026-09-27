#!/usr/bin/env python
"""r02 - per-entity protein measurements for every TRACER entity, rebuilt.

WHY THIS REPLACES pr8/scripts/20_prep_figure_data.py
----------------------------------------------------
Two estimands in the previous per-entity table cannot support an orthogonal
validation claim.

 1. `fold = mean_intensity / background_median`.  Every Xenium protein channel
    carries an additive pedestal near 99 counts (cell-free background median
    98-101 for all 23 markers; whole-image p99 only 100-250).  Dividing by the
    pedestal turns a genuine 3-fold signal into a 1.03-fold ratio - which is why
    the published intensity panel spans 0.95-1.20 and whole cells, partial cells
    and empty tissue are indistinguishable.  Fixed by SUBTRACTING the pedestal:
    S = max(I - o_m, 0).

 2. `positive = I > background + 2 * MAD`.  On integer-quantised images the MAD
    of cell-free background is exactly 0 or 1 count, so the threshold is 99 or
    102 and differs between markers for purely numerical reasons; where MAD = 0
    the threshold equals the background median itself.  Fixed by taking a fixed
    upper quantile of S over cell-free tissue, which pins the per-pixel false
    positive rate at 1 % for EVERY marker (95 / 99 / 99.9 kept as sensitivity).

Three further changes:
 * measurements are taken over each entity's UNIQUE footprint pixels rather than
   once per transcript, so transcript-dense entities are not up-weighted;
 * every quantity is computed twice - over all footprint pixels and over pixels
   further than SPILL_UM from any 10x whole-cell mask - which is the
   optical-spillover control;
 * the four separately imaged Cellular Localization channels (E-cadherin,
   Vimentin, alphaSMA, CD45) and the four autofluorescence background channels
   are included.  The AF channels are the negative control that separates real
   residual staining from a locally bright patch of tissue; no earlier analysis
   used them.

A toroidal-shift null is accumulated alongside: the same footprints sample the
same image after a rigid displacement of >= 25 um, which preserves staining
texture and cell packing but destroys registration.

IMPLEMENTATION NOTE.  Both SLURM allocations are exhausted (adeshpa6
6.01M/6.00M CPU-min, aszalay1 59.8k/60.0k, quarterly reset), so this runs inside
a 5 GB login cgroup shared with other processes.  Nothing large is ever held
resident: transcripts are bucketed to disk by image row, footprint dedupe runs
one bucket at a time, index slices are re-read with np.fromfile rather than kept
mapped, images are read in 1024-row bands through the TIFF's own zarr store, and
per-marker results go straight into a disk-backed float32 matrix.  Peak RSS is
about 350 MB at full level-2 resolution.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np, pandas as pd, tifffile, zarr
import pyarrow as pa, pyarrow.parquet as pq
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcutil import trim, mklog, wait_for_headroom

RCC   = Path("/scratch4/adeshpa6/xenium_rcc/rcc_analysis")
FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OUT, TMP = FINAL / "objects", FINAL / "objects/_tmp_r02"
PROT = RCC / "protein"
REFINED = Path("/scratch4/adeshpa6/xenium_rcc/tracer/run_wt/outputs/"
               "transcripts_tracer_refined.parquet")
LEVEL, SPILL_UM = 2, 2.0
SUB_FRAC = 0.06        # pixel subsample used for the toroidal-shift null
N_SHIFT, MIN_SHIFT_UM = 24, 25.0
BAND, BUCKET_ROWS = 1024, 256

PANEL23 = ["cd4", "cd20", "cd8a", "cd3e", "cd138", "hla-dr", "cd11c", "cd68",
           "cd16", "granzymeb", "cd163", "cd45ra", "pcna", "cd45ro", "ki-67",
           "pd-1", "vista", "pd-l1", "lag-3", "beta-catenin", "cd31", "pten",
           "panck"]
CELLLOC = ["e-cadherin", "vimentin", "alphasma", "cd45"]
AF      = ["af_blue", "af_green", "af_yellow", "af_red"]
ALLCH   = PANEL23 + CELLLOC + AF
STATS   = ["int", "pos", "pos_far", "pos_near"]
log = mklog("r02")


def rd(path, dtype, s, e):
    """Read rows [s, e) of a flat binary file without leaving a mapping behind."""
    it = np.dtype(dtype).itemsize
    return np.fromfile(path, dtype=dtype, count=int(e - s), offset=int(s) * it)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    man = json.loads((PROT / "protein_manifest.json").read_text())
    umpp = man["pixel_um_level0"] * (2 ** LEVEL)
    t0 = time.time()
    wait_for_headroom(400, log=log)

    edt_path = RCC / "validation/edt_L2.npy"
    with open(edt_path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        (H, W), _, edt_dtype = np.lib.format._read_array_header(f, ver)
        edt_off = f.tell()
    log(f"grid {H}x{W} @ {umpp:.4f} um/px")

    def edt_band(r0, r1):
        return np.fromfile(edt_path, dtype=edt_dtype, count=(r1 - r0) * W,
                           offset=edt_off + r0 * W * edt_dtype.itemsize
                           ).reshape(r1 - r0, W)

    master = pd.read_parquet(RCC / "label_transfer/master_cells.parquet")
    master["cell_id"] = master.cell_id.astype(str)
    ent_ids = master.cell_id.to_numpy()
    ent_index = pd.Index(ent_ids)
    n_ent = len(ent_ids)

    # ---------------- pass 1: transcripts -> row buckets on disk ------------
    nb = (H + BUCKET_ROWS - 1) // BUCKET_ROWS
    fh = [open(TMP / f"b{i:03d}.bin", "wb") for i in range(nb)]
    ntx = np.zeros(n_ent, np.int64); sx = np.zeros(n_ent); sy = np.zeros(n_ent)
    pf = pq.ParquetFile(REFINED)
    nbatch = 0
    for bt in pf.iter_batches(batch_size=250_000, columns=["stitched", "x", "y"]):
        dic = pa.compute.dictionary_encode(bt.column("stitched"))
        lut = ent_index.get_indexer(pd.Index(dic.dictionary.to_pylist()))
        c = lut[np.asarray(dic.indices)]
        del dic, lut
        keep = c >= 0
        c = c[keep].astype(np.int32)
        x = np.asarray(bt.column("x"), np.float64)[keep]
        y = np.asarray(bt.column("y"), np.float64)[keep]
        del bt, keep
        ntx += np.bincount(c, minlength=n_ent)
        sx += np.bincount(c, weights=x, minlength=n_ent)
        sy += np.bincount(c, weights=y, minlength=n_ent)
        py = np.clip((y / umpp).astype(np.int32), 0, H - 1)
        pxx = np.clip((x / umpp).astype(np.int32), 0, W - 1)
        del x, y
        b = py // BUCKET_ROWS
        o = np.argsort(b, kind="stable")
        b, c, py, pxx = b[o], c[o], py[o], pxx[o]
        edges = np.searchsorted(b, np.arange(nb + 1))
        rec = np.empty(len(c), dtype=[("c", "<i4"), ("y", "<i2"), ("x", "<i2")])
        rec["c"] = c; rec["y"] = py.astype(np.int16); rec["x"] = pxx.astype(np.int16)
        for i in range(nb):
            if edges[i + 1] > edges[i]:
                fh[i].write(rec[edges[i]:edges[i + 1]].tobytes())
        del rec, b, c, py, pxx, o, edges
        nbatch += 1
        if nbatch % 8 == 0:
            trim()
    for f in fh:
        f.close()
    del pf, fh, ent_index
    trim()
    log(f"pass 1: {int(ntx.sum()):,} assigned transcripts, "
        f"{int((ntx>0).sum()):,} entities ({time.time()-t0:.0f}s)")

    # ---------------- pass 2: dedupe per row bucket, row-sorted -------------
    dt = np.dtype([("c", "<i4"), ("y", "<i2"), ("x", "<i2")])
    fc = open(TMP / "uc.i32", "wb"); fy = open(TMP / "uy.i16", "wb")
    fx = open(TMP / "ux.i16", "wb"); ff = open(TMP / "far.u1", "wb")
    M = 0
    row_count = np.zeros(H, np.int64)
    n_upx = np.zeros(n_ent, np.int64); nfar = np.zeros(n_ent, np.int64)
    s_d = np.zeros(n_ent); mx_d = np.zeros(n_ent)
    for i in range(nb):
        p = TMP / f"b{i:03d}.bin"
        if not p.exists():
            continue
        if p.stat().st_size == 0:
            os.remove(p); continue
        r = np.fromfile(p, dtype=dt)
        key = (r["c"].astype(np.int64) << 32) | \
              (r["y"].astype(np.int64) << 16) | (r["x"].astype(np.int64) & 0xFFFF)
        del r
        u = np.unique(key); del key
        yy = ((u >> 16) & 0xFFFF).astype(np.int32)
        xx = (u & 0xFFFF).astype(np.int32)
        cc = (u >> 32).astype(np.int32); del u
        o = np.lexsort((xx, yy)); cc, yy, xx = cc[o], yy[o], xx[o]; del o
        r0, r1 = int(yy[0]), int(yy[-1]) + 1
        d = edt_band(r0, r1)[yy - r0, xx]
        fr = d > SPILL_UM
        row_count += np.bincount(yy, minlength=H)
        n_upx += np.bincount(cc, minlength=n_ent)
        nfar += np.bincount(cc, weights=fr, minlength=n_ent).astype(np.int64)
        s_d += np.bincount(cc, weights=d.astype(np.float64), minlength=n_ent)
        np.maximum.at(mx_d, cc, d.astype(np.float64))
        fc.write(cc.astype(np.int32).tobytes())
        fy.write(yy.astype(np.int16).tobytes())
        fx.write(xx.astype(np.int16).tobytes())
        ff.write(fr.astype(np.uint8).tobytes())
        M += len(cc)
        del cc, yy, xx, d, fr
        os.remove(p); trim()
    fc.close(); fy.close(); fx.close(); ff.close()
    row_start = np.concatenate([[0], np.cumsum(row_count)]).astype(np.int64)
    log(f"{M:,} unique entity-pixel pairs "
        f"(median {np.median(n_upx[n_upx>0]):.0f}/entity, {time.time()-t0:.0f}s)")

    # ---------------- per-entity geometry -----------------------------------
    df = pd.DataFrame({"entity": ent_ids, "n_tx": ntx.astype(np.int32),
                       "n_px": n_upx.astype(np.int32),
                       "n_px_far": nfar.astype(np.int32)})
    with np.errstate(invalid="ignore", divide="ignore"):
        df["centroid_x"] = (sx / np.maximum(ntx, 1)).astype(np.float32)
        df["centroid_y"] = (sy / np.maximum(ntx, 1)).astype(np.float32)
    df.loc[df.n_tx == 0, ["centroid_x", "centroid_y"]] = np.nan
    df["frac_px_far"] = (df.n_px_far / np.maximum(df.n_px, 1)).astype(np.float32)
    df["dist_um_mean"] = (s_d / np.maximum(n_upx, 1)).astype(np.float32)
    df["dist_um_max"] = mx_d.astype(np.float32)
    df["entity_class"] = pd.Categorical(master.entity_class.to_numpy())
    df["cell_type"] = pd.Categorical(master.transferred_label.to_numpy())
    df["transfer_conf"] = master.transfer_confidence.to_numpy().astype(np.float32)

    from scipy.spatial import cKDTree
    wm = master[master.entity_class == "whole"]
    tree = cKDTree(wm[["centroid_x", "centroid_y"]].to_numpy())
    xy = df[["centroid_x", "centroid_y"]].to_numpy()
    ok = np.isfinite(xy).all(1)
    dd = np.full(len(df), np.nan); tt = np.array([None] * len(df), object)
    dist_w, idx_w = tree.query(xy[ok], k=2)
    pick = np.where((df.entity_class.to_numpy() == "whole")[ok], 1, 0)
    ar = np.arange(int(ok.sum()))
    dd[ok] = dist_w[ar, pick]
    tt[ok] = wm.transferred_label.to_numpy()[idx_w[ar, pick]]
    df["nn_whole_dist_um"] = dd.astype(np.float32)
    df["nn_whole_type"] = pd.Categorical(tt)
    del tree, dist_w, idx_w, xy, wm, master, s_d, mx_d, dd, tt, sx, sy
    trim()
    log(f"geometry + nearest-whole-cell done ({time.time()-t0:.0f}s)")

    # ---------------- marker loop, banded, disk-backed output ---------------
    rng = np.random.default_rng(0)
    msh = int(round(MIN_SHIFT_UM / umpp))
    shifts = [(int(rng.integers(msh, H - msh)), int(rng.integers(msh, W - msh)))
              for _ in range(N_SHIFT)]
    tcodes, tuniq = pd.factorize(df.cell_type.astype(str).to_numpy())
    tcodes = tcodes.astype(np.int8)
    ncol = len(ALLCH) * len(STATS)
    npx = np.maximum(n_upx, 1).astype(np.float64)
    nfr = np.maximum(nfar, 1).astype(np.float64)
    nnr = np.maximum(n_upx - nfar, 1).astype(np.float64)
    # cell-free tissue mask at L3, built once
    h3, w3 = np.load(PROT / f"downsampled/{ALLCH[0]}_L3.npy", mmap_mode="r").shape
    cfmask = np.zeros((h3, w3), bool)
    for r0 in range(0, H, BAND):
        r1 = min(r0 + BAND, H)
        blk = edt_band(r0, r1)[::2, ::2]
        a = r0 // 2
        b = min(a + blk.shape[0], h3)
        if a >= h3:
            break
        cfmask[a:b] = blk[:b - a, :w3] > 3.0
        del blk
    trim()

    # Toroidal-shift null.  Evaluating 24 shifts over all 37.5 M footprint
    # pixels for 32 channels is I/O bound, and the null only needs a per-type
    # rate, so it is estimated on a type-stratified pixel subsample; with
    # >= 50,000 pixels per type the binomial SE of the null rate is < 0.25 %.
    tden = np.zeros(len(tuniq)); tden_far = np.zeros(len(tuniq))
    sub_c, sub_y, sub_x, sub_f = [], [], [], []
    rng_s = np.random.default_rng(1)
    for r0 in range(0, H, BAND):
        r1 = min(r0 + BAND, H)
        s, e = int(row_start[r0]), int(row_start[r1])
        if e <= s:
            continue
        cc = rd(TMP / "uc.i32", np.int32, s, e)
        fr = rd(TMP / "far.u1", np.uint8, s, e).astype(bool)
        tt = tcodes[cc]
        tden += np.bincount(tt, minlength=len(tuniq))
        tden_far += np.bincount(tt, weights=fr.astype(float),
                                minlength=len(tuniq))
        take = rng_s.random(e - s) < SUB_FRAC
        if take.any():
            sub_c.append(cc[take])
            sub_y.append(rd(TMP / "uy.i16", np.int16, s, e)[take])
            sub_x.append(rd(TMP / "ux.i16", np.int16, s, e)[take])
            sub_f.append(fr[take])
        del cc, fr, tt, take
    sub_c = np.concatenate(sub_c); sub_y = np.concatenate(sub_y).astype(np.int32)
    sub_x = np.concatenate(sub_x).astype(np.int32)
    sub_f = np.concatenate(sub_f)
    sub_t = tcodes[sub_c]
    del sub_c
    sden = np.bincount(sub_t, minlength=len(tuniq)).astype(float)
    sden_far = np.bincount(sub_t, weights=sub_f.astype(float),
                           minlength=len(tuniq))
    log(f"toroidal-null subsample: {len(sub_t):,} pixels; per type " +
        ", ".join(f"{t}={int(n):,}" for t, n in zip(tuniq, sden)))
    null_rows, thr_info = [], {}

    for mi, mk in enumerate(ALLCH):
        wait_for_headroom(150, log=log)
        small = np.load(PROT / f"downsampled/{mk}_L3.npy")
        cf = small[cfmask & (small > 0)].astype(np.float32)
        vals, cnts = np.unique(cf.astype(np.int32), return_counts=True)
        ped = float(vals[int(np.argmax(cnts))])
        S = np.maximum(cf - ped, 0)
        q = {f"q{p}": float(np.percentile(S, p)) for p in (95, 99, 99.9)}
        thr = max(q["q99"], 1.0)
        thr_info[mk] = dict(pedestal=ped, thresholds=q, thr_used=thr,
                            cellfree_frac_pos=float((S > thr).mean()),
                            cellfree_px=int(len(S)))
        del small, cf, vals, cnts, S
        trim()

        ch = man["markers"][mk]["channel_file"]
        tf = tifffile.TiffFile(str(PROT / "tif" / ch), is_ome=False)
        store = tf.series[0].levels[LEVEL].aszarr(); z = zarr.open(store, mode="r")
        s_int = np.zeros(n_ent); s_pos = np.zeros(n_ent); s_posf = np.zeros(n_ent)
        nul = np.zeros((N_SHIFT, len(tuniq))); nulf = np.zeros((N_SHIFT, len(tuniq)))
        for r0 in range(0, H, BAND):
            r1 = min(r0 + BAND, H)
            blk = np.asarray(z[r0:r1, :]).astype(np.float32)
            blk -= ped; np.maximum(blk, 0, out=blk)
            s, e = int(row_start[r0]), int(row_start[r1])
            if e > s:
                cc = rd(TMP / "uc.i32", np.int32, s, e)
                yy = rd(TMP / "uy.i16", np.int16, s, e).astype(np.int32) - r0
                xx = rd(TMP / "ux.i16", np.int16, s, e).astype(np.int32)
                fr = rd(TMP / "far.u1", np.uint8, s, e).astype(bool)
                v = blk[yy, xx]; p_ = v > thr
                np.add.at(s_int, cc, v.astype(np.float64))
                np.add.at(s_pos, cc, p_.astype(np.float64))
                np.add.at(s_posf, cc, (p_ & fr).astype(np.float64))
                del cc, yy, xx, fr, v, p_
            for k, (dy, dx) in enumerate(shifts):
                yy = (sub_y + dy) % H
                m = (yy >= r0) & (yy < r1)
                if not m.any():
                    del yy
                    continue
                xx = (sub_x[m] + dx) % W
                p_ = blk[yy[m] - r0, xx] > thr
                tt = sub_t[m]
                nul[k] += np.bincount(tt, weights=p_.astype(float),
                                      minlength=len(tuniq))
                nulf[k] += np.bincount(tt, weights=(p_ & sub_f[m]).astype(float),
                                       minlength=len(tuniq))
                del yy, m, xx, p_, tt
            del blk
        store.close(); tf.close(); trim()

        np.save(TMP / f"col_{mk}.npy", np.column_stack([
            (s_int / npx),
            (s_pos / npx),
            np.where(nfar > 0, s_posf / nfr, np.nan),
            np.where((n_upx - nfar) > 0, (s_pos - s_posf) / nnr, np.nan),
        ]).astype(np.float32))
        for k in range(N_SHIFT):
            for ti, tname in enumerate(tuniq):
                null_rows.append(dict(marker=mk, shift=k, cell_type=tname,
                                      pos_frac=nul[k, ti] / max(sden[ti], 1),
                                      pos_frac_far=nulf[k, ti]
                                      / max(sden_far[ti], 1)))
        del s_int, s_pos, s_posf, nul, nulf
        log(f"  {mk:12s} ped={ped:6.1f} thr=+{thr:5.1f} "
            f"cfFPR={thr_info[mk]['cellfree_frac_pos']:.4f} "
            f"({time.time()-t0:.0f}s)")

    del tcodes, npx, nfr, nnr, sub_y, sub_x, sub_f, sub_t
    trim()

    # ---------------- write output in row blocks ---------------------------
    writer = None
    for a in range(0, n_ent, 50_000):
        b = min(a + 50_000, n_ent)
        blk = df.iloc[a:b].reset_index(drop=True)
        for mk in ALLCH:
            c = np.load(TMP / f"col_{mk}.npy", mmap_mode="r")[a:b]
            for si, st in enumerate(STATS):
                blk[f"{mk}_{st}"] = np.asarray(c[:, si])
            del c
        tbl = pa.Table.from_pandas(blk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT / "rcc_percell_protein.parquet",
                                      tbl.schema, compression="zstd")
        writer.write_table(tbl)
        del blk, tbl
        trim()
    writer.close()
    pd.DataFrame(null_rows).to_parquet(OUT / "rcc_torus_null.parquet", index=False)
    (OUT / "rcc_marker_thresholds.json").write_text(json.dumps({
        "level": LEVEL, "um_per_px": umpp, "spill_um": SPILL_UM,
        "n_shift": N_SHIFT, "min_shift_um": MIN_SHIFT_UM,
        "shifts_px": shifts, "n_transcripts": int(ntx.sum()),
        "n_unique_px": int(M), "n_entities": int(n_ent),
        "stats": STATS, "markers_panel23": PANEL23,
        "markers_cellloc": CELLLOC, "markers_af": AF,
        "markers": thr_info}, indent=2))
    for f in (["uc.i32", "uy.i16", "ux.i16", "far.u1"]
              + [f"col_{mk}.npy" for mk in ALLCH]):
        try:
            os.remove(TMP / f)
        except OSError:
            pass
    log(f"DONE {time.time()-t0:.0f}s -> {n_ent} entities x {ncol} marker columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
