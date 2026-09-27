#!/usr/bin/env python
"""r03 - does a reconstructed partial cell carry ITS OWN lineage's protein?

THE ESTIMAND, AND WHY IT IS NOT THE PREVIOUS ONE
------------------------------------------------
Earlier panels compared a partial cell's matched-marker intensity with a global
tissue background.  That comparison is confounded by local brightness: a partial
sitting in a dense, autofluorescent or thick region is brighter in EVERY
channel, so "above background" carries no identity information - and with the
pedestal-ratio estimand the whole comparison lived inside a 5 % dynamic range.

The statistic here is a within-entity paired contrast

    D_i = pos_i(canonical marker of its own transferred type)
          - mean_{other six canonical markers} pos_i(m)

measured on the same pixels of the same entity, where pos is the fraction of the
entity's footprint pixels above a threshold that fixes the cell-free false
positive rate at 1 %.  Any per-entity brightness factor cancels: D > 0 means the
entity is preferentially positive for its OWN lineage marker, not generically
bright.

Nulls (all reported):
  N1 identity permutation - a random OTHER transferred type supplies the
     canonical marker; preserves geometry, images and the marker profile of each
     entity, destroys only the identity match.
  N2 toroidal image shift - footprints resample the same image displaced by
     >= 25 um (computed in r02); preserves staining texture and cell packing,
     destroys registration.  Used for the specificity matrices.
  N3 swapped marker map - a derangement of cell type -> canonical marker, so
     every entity is scored on another lineage's marker.
  N4 autofluorescence - the four AF background channels; a type-specific
     enrichment there would mean the effect is optical, not immunological.

Spillover controls:
  S1 pixels further than 2 um from any 10x whole-cell mask;
  S2 partial cells whose NEAREST whole cell has a DIFFERENT transferred type -
     optical bleed would import the neighbour's identity, not the partial's own.

Effect sizes are medians with bootstrap CIs plus Cliff's delta against the null;
p values are permutation-based and BH-corrected across the reported grid.

Outputs (final/tables/): r03_*.csv, r03_summary.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcutil import trim, mklog, bh, boot_ci, cliffs_delta
from rcc_defs import PANEL23, CELLLOC, AF, MARKERS, ALLCH, CANON, TYPES

FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
OBJ, TAB = FINAL / "objects", FINAL / "tables"
MIN_PX = 8
N_PERM = 1000
N_CAP = 60_000     # entities per test; the median is already
                   # precise at this size and the permutation
                   # null then costs 6x less on the large groups
log = mklog("r03")


def delta(P, code):
    """Matched-minus-mismatched positivity contrast, per entity."""
    return make_delta(P)(code)


def make_delta(P):
    """Contrast evaluator with the code-independent terms precomputed.

    The row sum and the count of finite markers do not depend on which marker is
    treated as "own", so hoisting them out turns each of the 2,000 permutation
    draws into a single fancy index instead of two full passes over a 20 MB
    matrix.
    """
    tot = np.nansum(P, axis=1)
    k = np.maximum(np.isfinite(P).sum(axis=1) - 1, 1)
    idx = np.arange(len(P))

    def f(code):
        own = P[idx, code]
        return own - (tot - np.nan_to_num(own)) / k
    return f


def subsample(mask, cap, rng):
    """Cap a group at `cap` entities, reported alongside the full n."""
    idx = np.flatnonzero(mask)
    if len(idx) <= cap:
        return idx
    return np.sort(rng.choice(idx, cap, replace=False))


def other_codes(code, n_types, rng):
    """A random type index different from the observed one, per entity."""
    off = rng.integers(1, n_types, len(code), dtype=np.int64)
    return (code + off) % n_types


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = json.loads((OBJ / "rcc_marker_thresholds.json").read_text())
    # The per-entity table is 507k x 138 columns; loading it whole exceeds the
    # shared login cgroup, so columns are pulled on demand and cached only while
    # they are in use.
    SRC = OBJ / "rcc_percell_protein.parquet"
    base = ["entity", "entity_class", "cell_type", "transfer_conf", "n_tx",
            "n_px", "n_px_far", "frac_px_far", "dist_um_mean", "dist_um_max",
            "centroid_x", "centroid_y", "nn_whole_dist_um", "nn_whole_type"]
    D = pd.read_parquet(SRC, columns=base)
    for c in ("cell_type", "entity_class", "nn_whole_type"):
        D[c] = D[c].astype(str)
    _cache = {}

    def col(name):
        if name not in _cache:
            if len(_cache) > 40:
                _cache.clear(); trim()
            _cache[name] = pd.read_parquet(SRC, columns=[name])[name] \
                .to_numpy(np.float64)
        return _cache[name]

    log(f"loaded {D.shape[0]:,} entities, base columns only")

    canon = [CANON[t] for t in TYPES]
    nT = len(TYPES)
    sel = (D.n_px >= MIN_PX) & D.cell_type.isin(TYPES)
    S = D[sel].reset_index(drop=True)
    tcode = pd.Categorical(S.cell_type, categories=TYPES).codes.astype(int)
    is_part = (S.entity_class == "partial").to_numpy()
    discord = S.nn_whole_type.to_numpy() != S.cell_type.to_numpy()
    log(f"{len(S):,} scored entities ({int(is_part.sum()):,} partial, "
        f"{int((~is_part).sum()):,} whole); "
        f"{int((is_part & discord).sum()):,} partials with a discordant "
        f"nearest whole cell")

    rng = np.random.default_rng(0)
    # a derangement of the canonical marker map (N3)
    swap = np.arange(nT)
    while np.any(swap == np.arange(nT)):
        swap = rng.permutation(nT)

    rows = []
    keep_idx = np.flatnonzero(sel.to_numpy())
    for view, sfx in (("all_pixels", "pos"), ("spillover_free", "pos_far")):
        P = np.column_stack([col(f"{m}_{sfx}")[keep_idx] for m in canon])
        PAF = np.column_stack([col(f"{m}_{sfx}")[keep_idx] for m in AF])
        usable = np.isfinite(P).sum(1) >= nT - 1
        for cls, m0 in (("partial", is_part), ("whole", ~is_part)):
            for sub, m1 in (("all", np.ones(len(S), bool)),
                            ("discordant_neighbour", discord)):
                for t in ["ALL"] + TYPES:
                    m = usable & m0 & m1
                    if t != "ALL":
                        m = m & (tcode == np.array(TYPES).tolist().index(t))
                    n = int(m.sum())
                    if n < 30:
                        continue
                    take = subsample(m, N_CAP, rng)
                    n_used = len(take)
                    code = tcode[take]
                    Pm = P[take]; PAFm = PAF[take]
                    fd = make_delta(Pm)
                    d = fd(code)
                    # per-entity null draw, so Cliff's delta compares like with
                    # like: the same entities scored on another lineage's marker
                    d_null = fd(other_codes(code, nT, rng))
                    nul = np.array([np.median(fd(other_codes(code, nT, rng)))
                                    for _ in range(N_PERM)])
                    obs = float(np.nanmedian(d))
                    lo, hi = boot_ci(d, np.median, n=1000)
                    d_swap = fd(swap[code])                 # N3 swapped map
                    afc = rng.integers(0, PAFm.shape[1], n_used)
                    d_af = delta(PAFm, afc)
                    rows.append(dict(
                        view=view, entity_class=cls, subset=sub, cell_type=t,
                        marker=CANON.get(t, "canonical set"), n=n,
                        n_used=n_used,
                        median_delta=obs, ci_lo=lo, ci_hi=hi,
                        mean_delta=float(np.nanmean(d)),
                        frac_positive_delta=float(np.nanmean(d > 0)),
                        null_median=float(np.median(nul)),
                        null_sd=float(np.std(nul, ddof=1)),
                        z_vs_null=float((obs - np.median(nul))
                                        / max(np.std(nul, ddof=1), 1e-12)),
                        cliffs_delta_vs_null=cliffs_delta(d, d_null),
                        median_delta_swapped_map=float(np.nanmedian(d_swap)),
                        median_delta_autofluor=float(np.nanmedian(d_af)),
                        p_perm=float((np.sum(nul >= obs) + 1) / (N_PERM + 1))))
                    del Pm, PAFm, fd, d, d_null, nul, d_swap, d_af
                trim()
        del P, PAF
        _cache.clear(); trim()
        log(f"view={view} done ({time.time()-t0:.0f}s)")
    R = pd.DataFrame(rows)
    R["q_bh"] = bh(R.p_perm.to_numpy())
    R.to_csv(TAB / "r03_matched_marker_contrast.csv", index=False)
    log(f"matched-marker contrast: {len(R)} rows")

    # ---- specificity matrices vs the toroidal-shift null -------------------
    NUL = pd.read_parquet(OBJ / "rcc_torus_null.parquet")
    nsum = (NUL.groupby(["marker", "cell_type"])
            .agg(null_mean=("pos_frac", "mean"), null_sd=("pos_frac", "std"),
                 null_mean_far=("pos_frac_far", "mean"),
                 null_sd_far=("pos_frac_far", "std")).reset_index())
    from scipy import stats as st
    spec = []
    okpx = (D.n_px >= MIN_PX).to_numpy()
    grp = {}
    for cls in ("partial", "whole"):
        for t in sorted(set(D.cell_type)):
            if t == "nan":
                continue
            idx = np.flatnonzero(okpx & (D.entity_class.to_numpy() == cls)
                                 & (D.cell_type.to_numpy() == t))
            if len(idx) >= 30:
                grp[(cls, t)] = idx
    for m in ALLCH:
        v = col(f"{m}_pos"); vf = col(f"{m}_pos_far")
        for (cls, t), idx in grp.items():
            nr = nsum[(nsum.marker == m) & (nsum.cell_type == t)]
            if not len(nr):
                continue
            a = v[idx]; b = vf[idx]
            nfin = int(np.isfinite(b).sum())
            o = float(np.nanmean(a)); of = float(np.nanmean(b))
            nm, ns = float(nr.null_mean.iloc[0]), float(nr.null_sd.iloc[0])
            nmf = float(nr.null_mean_far.iloc[0])
            nsf = float(nr.null_sd_far.iloc[0])
            # entity-level SE, so the test is not driven by the tiny
            # between-shift SD of a group-level null
            se = float(np.nanstd(a, ddof=1) / np.sqrt(len(idx)))
            sef = float(np.nanstd(b, ddof=1) / np.sqrt(max(nfin, 1)))
            spec.append(dict(
                entity_class=cls, cell_type=t, marker=m, n=int(len(idx)),
                obs=o, null=nm, null_sd=ns, obs_se=se,
                log2_enrich=float(np.log2((o + 1e-4) / (nm + 1e-4))),
                z=float((o - nm) / max(np.hypot(se, ns), 1e-12)),
                obs_far=of, null_far=nmf,
                log2_enrich_far=float(np.log2((of + 1e-4) / (nmf + 1e-4))),
                z_far=float((of - nmf) / max(np.hypot(sef, nsf), 1e-12)),
                is_canonical=bool(CANON.get(t) == m),
                is_af=bool(m in AF)))
        _cache.clear(); trim()
    SP = pd.DataFrame(spec)
    SP["p_z"] = 2 * st.norm.sf(np.abs(SP.z))
    SP["q_bh"] = bh(SP.p_z.to_numpy())
    SP.to_csv(TAB / "r03_specificity_matrix.csv", index=False)
    log(f"specificity matrix: {len(SP)} rows; "
        f"AF max |log2| = {SP[SP.is_af].log2_enrich.abs().max():.3f}")

    # ---- per-marker distributions, whole vs partial vs registration null --
    dist = []
    cls_idx = {c: np.flatnonzero(okpx & (D.entity_class.to_numpy() == c))
               for c in ("whole", "partial")}
    for m in ALLCH:
        nm = float(nsum[nsum.marker == m].null_mean.mean())
        pv = col(f"{m}_pos"); iv = col(f"{m}_int")
        for cls in ("whole", "partial"):
            g = cls_idx[cls]
            v = pv[g]; i = iv[g]
            lo, hi = boot_ci(v, np.mean, n=400)
            dist.append(dict(
                marker=m, entity_class=cls, n=int(len(g)),
                mean_pos=float(np.nanmean(v)), ci_lo=lo, ci_hi=hi,
                median_pos=float(np.nanmedian(v)),
                mean_int=float(np.nanmean(i)), median_int=float(np.nanmedian(i)),
                null_pos=nm,
                log2_enrich=float(np.log2((np.nanmean(v) + 1e-4) / (nm + 1e-4))),
                pedestal=meta["markers"][m]["pedestal"],
                thr=meta["markers"][m]["thr_used"],
                cellfree_fpr=meta["markers"][m]["cellfree_frac_pos"]))
        _cache.clear(); trim()
    pd.DataFrame(dist).to_csv(TAB / "r03_marker_distributions.csv", index=False)

    # ---- spillover: matched-marker positivity inside vs outside 0-2 um ----
    sp2 = []
    for t in TYPES:
        m = CANON[t]
        an = col(f"{m}_pos_near"); af_ = col(f"{m}_pos_far"); ap = col(f"{m}_pos")
        for cls in ("partial", "whole"):
            idx = np.flatnonzero(okpx & (D.cell_type.to_numpy() == t)
                                 & (D.entity_class.to_numpy() == cls))
            if len(idx) < 30:
                continue
            g = D.iloc[idx]
            near = an[idx]; far = af_[idx]
            both = np.isfinite(near) & np.isfinite(far)
            lo, hi = boot_ci(far[both] - near[both], np.median, n=1000)
            sp2.append(dict(
                cell_type=t, marker=m, entity_class=cls, n=int(len(g)),
                n_paired=int(both.sum()),
                pos_all=float(np.nanmean(ap[idx])),
                pos_within_2um=float(np.nanmean(near)),
                pos_beyond_2um=float(np.nanmean(far)),
                paired_median_diff=float(np.nanmedian(far[both] - near[both])),
                ci_lo=lo, ci_hi=hi,
                frac_px_beyond_2um=float(np.nanmean(g.frac_px_far))))
        _cache.clear(); trim()
    pd.DataFrame(sp2).to_csv(TAB / "r03_spillover.csv", index=False)

    # ---- distance to nearest original whole cell --------------------------
    dd = (D[D.n_px >= MIN_PX].groupby("entity_class")
          .agg(n=("entity", "size"), median_dist=("dist_um_mean", "median"),
               q25=("dist_um_mean", lambda v: float(np.nanpercentile(v, 25))),
               q75=("dist_um_mean", lambda v: float(np.nanpercentile(v, 75))),
               median_nn=("nn_whole_dist_um", "median"),
               mean_frac_beyond_2um=("frac_px_far", "mean")).reset_index())
    dd.to_csv(TAB / "r03_distance_summary.csv", index=False)
    keepc = ["entity", "entity_class", "cell_type", "dist_um_mean",
             "dist_um_max", "nn_whole_dist_um", "nn_whole_type", "frac_px_far",
             "n_px", "n_tx", "centroid_x", "centroid_y", "transfer_conf"]
    keep2 = np.flatnonzero(okpx)
    out = D.iloc[keep2][keepc].copy().reset_index(drop=True)
    for t in TYPES:
        out[f"pos_{CANON[t]}"] = col(f"{CANON[t]}_pos")[keep2]
        out[f"posfar_{CANON[t]}"] = col(f"{CANON[t]}_pos_far")[keep2]
    Pall = np.column_stack([out[f"pos_{CANON[t]}"].to_numpy(np.float64)
                            for t in TYPES])
    oc = pd.Categorical(out.cell_type, categories=TYPES).codes
    dvals = np.full(len(out), np.nan)
    good = oc >= 0
    dvals[good] = delta(Pall[good], oc[good].astype(int))
    out["matched_delta"] = dvals
    out.to_parquet(OBJ / "rcc_entity_summary.parquet", index=False)

    (TAB / "r03_summary.json").write_text(json.dumps({
        "min_px": MIN_PX, "n_perm": N_PERM,
        "n_entities_scored": int(len(S)),
        "n_partial_scored": int(is_part.sum()),
        "n_whole_scored": int((~is_part).sum()),
        "canonical_markers": CANON,
        "swapped_map": {TYPES[i]: CANON[TYPES[swap[i]]] for i in range(nT)},
        "runtime_s": round(time.time() - t0, 1)}, indent=2))
    log(f"DONE {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
