#!/usr/bin/env python
"""r07 - RCC: do TRACER whole and partial entities occupy coherent transcriptional
states relative to the 10x segmentation?  (compute stage; figure is r08)

Four arms, all built from the SAME refined transcript table
(tracer/run_wt/outputs/transcripts_tracer_refined.parquet):
    10x              transcripts grouped by the vendor `cell_id`
    TRACER whole     grouped by `stitched`, entity_class == whole
    TRACER partial   grouped by `stitched`, entity_class == partial
    TRACER whole+partial   union of the two
Every arm uses the same [5, 6000] transcript filter as the published transfer.

Labels.  The published transfer (rcc_analysis/scripts/01_label_transfer.py) was
run on TRACER entities only; no 10x-arm labels exist.  The identical frozen
procedure - same reference.h5ad, same `select_anchors` call (clean_marker,
max 5000 / min 50 per class, seed 0), same cosine-centroid transfer at
temperature 0.05 in the shared-gene space - is therefore re-run here on BOTH
queries.  The TRACER re-run must reproduce the published labels (checked and
written to r07_label_reproduction.json); the 10x run is new.

Embedding.  One embedding for all arms: CP10K + log1p on the 405 panel genes,
z-scored with pooled statistics, PCA(30) and UMAP fitted ONCE on a balanced pool
(20k entities from each of 10x, whole, partial), then every displayed entity is
projected with the frozen transform.  No arm gets its own optimised UMAP.

Coherence.  k = 15 nearest-neighbour label agreement in the shared PCA space,
per arm and per type, against (i) the type-frequency baseline and (ii) TRACER
whole cells binomially thinned to the partial-cell depth distribution and
re-labelled with the same transfer - the depth-matched reference for partials.

Outputs: final/tables/r07_*.csv|json, final/objects/r07/*.parquet
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
import pyarrow as pa, pyarrow.parquet as pq

_LOCAL_HELPERS = Path(__file__).resolve().parents[1] / "helpers"
sys.path.insert(0, str(_LOCAL_HELPERS))
sys.path.insert(1, "/scratch4/adeshpa6/TRACER/scripts")
from label_transfer_spatial import select_anchors, transfer_labels_python  # noqa

BASE = Path("/scratch4/adeshpa6/xenium_rcc")
REFINED = BASE / "tracer/run_wt/outputs/transcripts_tracer_refined.parquet"
REF_H5 = BASE / "rcc_analysis/reference/reference.h5ad"
PUB_ANN = BASE / "rcc_analysis/label_transfer/rcc_transferred_cell_annotations.csv"
FINAL = Path("/scratch4/adeshpa6/tracer_campaign/final")
TAB, OBJ = FINAL / "tables", FINAL / "objects/r07"

MIN_TX, MAX_TX = 5, 6000                    # as 01_label_transfer.py
DROP_IDS = {"UNASSIGNED", "DROP", "-1", "nan", "NA", "", "0"}
TEMP, LOW_CONF = 0.05, 0.4                  # as 01_label_transfer.py
N_POOL, N_SHOW, N_KNN, K = 20000, 50000, 20000, 15
N_PC, SEED = 30, 0

# canonical markers, fixed order; every gene is on the Xenium panel.  Lists
# follow rcc_analysis/scripts/03_marker_qc.py where the gene is on the panel,
# topped up with established markers (CD14, HPGDS) to four per type.
CANONICAL = {
    "Tumor":       ["CD70", "KRT8", "KRT18", "EPCAM"],
    "Myeloid":     ["CD68", "CD163", "AIF1", "CD14"],
    "T cell":      ["CD3E", "CD3D", "CD2", "TRAC"],
    "B cell":      ["MS4A1", "CD79A", "CD19", "BANK1"],
    "Plasma cell": ["MZB1", "DERL3", "TNFRSF17", "SDC1"],
    "Endothelial": ["PECAM1", "VWF", "EGFL7", "CD34"],
    "Mural":       ["ACTA2", "PDGFRB", "MYH11", "DES"],
    "Mast cell":   ["CPA3", "MS4A2", "KIT", "HPGDS"],
}
TYPES = list(CANONICAL)
# 03_marker_qc.py lineage lists (panel-restricted at run time) - used only to
# reproduce the published `marker_supported` flag for every arm
LINEAGE_03 = {
    "Tumor": ["CA9", "NDUFA4L2", "ANGPTL4", "VEGFA", "EPCAM", "KRT8", "KRT18",
              "KRT19", "PAX8", "CD70", "SLC17A3"],
    "Myeloid": ["CD68", "CD163", "ITGAX", "AIF1", "LYZ", "C1QA", "C1QB", "CSF1R",
                "FCGR3A", "MRC1", "ITGAM"],
    "T cell": ["CD3E", "CD3D", "CD2", "TRAC", "CD8A", "CD4", "IL7R", "CCL5", "GZMK"],
    "B cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Plasma cell": ["MZB1", "DERL3", "SDC1", "TNFRSF17", "XBP1", "IGHG1", "PRDM1"],
    "Endothelial": ["PECAM1", "VWF", "CD34", "EGFL7", "ACKR1", "PLVAP", "CLDN5",
                    "CDH5", "FLT1"],
    "Mural": ["ACTA2", "PDGFRB", "RGS5", "TAGLN", "MYH11", "NOTCH3", "DES"],
    "Mast cell": ["CPA3", "MS4A2", "KIT", "TPSAB1", "TPSB2"],
}


def log(m, t0=[time.time()]):
    print(f"[r07 {time.time()-t0[0]:7.0f}s] {m}", flush=True)


# --------------------------------------------------------------------------
def encode(col: pa.ChunkedArray):
    arr = col.combine_chunks()
    if pa.types.is_dictionary(arr.type):
        return arr.indices.to_numpy(zero_copy_only=False), \
            np.asarray(arr.dictionary.to_pylist(), dtype=object)
    enc = arr.dictionary_encode()
    return enc.indices.to_numpy(zero_copy_only=False), \
        np.asarray(enc.dictionary.to_pylist(), dtype=object)


def count_matrix(row_codes, row_cats, gcodes, n_gene):
    ok = ~pd.Series(row_cats).astype(str).isin(DROP_IDS).to_numpy()
    keep_tx = ok[row_codes]
    r, g = row_codes[keep_tx], gcodes[keep_tx]
    ur, rr = np.unique(r, return_inverse=True)
    X = sp.csr_matrix((np.ones(len(rr), np.int32), (rr, g)),
                      shape=(len(ur), n_gene))
    X.sum_duplicates()
    ids = row_cats[ur].astype(str)
    tot = np.asarray(X.sum(1)).ravel()
    keep = (tot >= MIN_TX) & (tot <= MAX_TX)
    return X[keep], ids[keep]


def lognorm(X):
    X = X.astype(np.float32).tocsr(copy=True)
    s = np.asarray(X.sum(1)).ravel(); s[s == 0] = 1
    X = sp.diags(1e4 / s) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def to_adata(X, ids, genes):
    return ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(ids, name="cell_id")),
                      var=pd.DataFrame(index=pd.Index(genes, name="gene")))


def marker_support(X, genes, labels):
    """Exact re-implementation of 03_marker_qc.py (lenient) plus a strict
    variant that does not let an all-zero marker profile count as support."""
    gi = {g: i for i, g in enumerate(genes)}
    tot = np.asarray(X.sum(1)).ravel().astype(np.float64); tot[tot == 0] = 1
    order = list(LINEAGE_03)
    S = np.zeros((X.shape[0], len(order)))
    for j, ct in enumerate(order):
        cols = [gi[g] for g in LINEAGE_03[ct] if g in gi]
        sub = X[:, cols].astype(np.float64).multiply(1.0 / tot[:, None]).tocsr()
        sub.data = np.log1p(sub.data * 1e3)
        S[:, j] = np.asarray(sub.mean(1)).ravel()
    rank = np.argsort(np.argsort(-S, axis=1, kind="quicksort"), axis=1)
    own = np.array([order.index(l) if l in order else -1 for l in labels])
    own_rank = np.where(own >= 0, rank[np.arange(len(own)), np.maximum(own, 0)], 99)
    own_score = np.where(own >= 0, S[np.arange(len(own)), np.maximum(own, 0)], 0)
    zero = S.max(1) == 0
    return own_rank <= 1, (own_rank <= 1) & (own_score > 0), zero


def thin(X, target, rng):
    """Binomial thinning of each row to (approximately) its target depth."""
    X = X.tocsr().astype(np.int64)
    tot = np.asarray(X.sum(1)).ravel()
    p = np.minimum(1.0, target / np.maximum(tot, 1))
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    data = rng.binomial(X.data, p[rows])
    Y = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    Y.eliminate_zeros()
    return Y.astype(np.int32)


# --------------------------------------------------------------------------
def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True); OBJ.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- 1. count matrices for both groupings from one table --------------
    t = pq.read_table(REFINED, columns=["feature_name", "cell_id", "stitched"])
    t = t.unify_dictionaries()
    gcodes, genes = encode(t.column("feature_name"))
    genes = genes.astype(str)
    ccodes, ccats = encode(t.column("cell_id"))
    scodes, scats = encode(t.column("stitched"))
    n_tx = t.num_rows
    del t
    log(f"{n_tx:,} transcripts, {len(genes)} panel features")
    X10, id10 = count_matrix(ccodes, ccats, gcodes, len(genes)); del ccodes, ccats
    XTr, idTr = count_matrix(scodes, scats, gcodes, len(genes)); del scodes, scats
    del gcodes
    log(f"10x entities {X10.shape[0]:,}; TRACER entities {XTr.shape[0]:,}")

    pub = pd.read_csv(PUB_ANN)
    pub["cell_id"] = pub.cell_id.astype(str)
    pub = pub.set_index("cell_id")
    same = set(idTr) == set(pub.index)
    log(f"TRACER entity set identical to published transfer: {same} "
        f"({len(idTr):,} vs {len(pub):,})")
    cls = pub.entity_class.reindex(idTr).to_numpy()

    # ---- 2. identical frozen transfer on both queries ----------------------
    ref = ad.read_h5ad(REF_H5)
    ref.obs["transferred_label_input"] = ref.obs["cell_type_coarse"].astype(str).values
    ref_anchor, anc = select_anchors(ref, label_col="transferred_label_input",
                                     strategy="clean_marker", max_per_type=5000,
                                     min_per_type=50, random_seed=0)
    anc.to_csv(TAB / "r07_reference_anchors.csv", index=False)

    # reference support for the canonical markers (independent of Xenium)
    rn = lognorm(ref.X.tocsr()); rg = {g: i for i, g in enumerate(ref.var_names)}
    rl = ref.obs.cell_type_coarse.astype(str).to_numpy()
    rows = []
    for ct, ms in CANONICAL.items():
        for g in ms:
            if g not in rg:
                rows.append(dict(cell_type=ct, gene=g, in_reference=False)); continue
            v = np.asarray(rn[:, rg[g]].todense()).ravel()
            means = {c: float(v[rl == c].mean()) for c in TYPES}
            top = max(means, key=means.get)
            rows.append(dict(cell_type=ct, gene=g, in_reference=True,
                             ref_mean_own=means[ct], ref_top_class=top,
                             ref_own_is_top=top == ct,
                             ref_own_over_next=means[ct] - sorted(
                                 [m for c, m in means.items() if c != ct])[-1]))
    pd.DataFrame(rows).to_csv(TAB / "r07_reference_marker_support.csv", index=False)
    del rn

    def transfer(X, ids):
        a, _ = transfer_labels_python(to_adata(X, ids, genes), ref_anchor,
                                      label_col="transferred_label_input",
                                      temperature=TEMP,
                                      softmax_low_confidence=LOW_CONF)
        return a.set_index("cell_id").reindex(ids)

    aTr = transfer(XTr, idTr)
    agree = float((aTr.transferred_label.to_numpy()
                   == pub.transferred_label.reindex(idTr).to_numpy()).mean())
    dconf = float(np.abs(aTr.transfer_confidence.to_numpy()
                         - pub.transfer_confidence.reindex(idTr).to_numpy()).max())
    log(f"TRACER re-run vs published: label agreement {agree:.6f}, max |dconf| {dconf:.2e}")
    a10 = transfer(X10, id10)
    shared = int(a10.shared_genes.iloc[0])
    (TAB / "r07_label_reproduction.json").write_text(json.dumps(dict(
        tracer_entity_set_identical=bool(same),
        tracer_label_agreement_with_published=agree,
        tracer_max_abs_confidence_difference=dconf,
        n_shared_genes=shared, n_panel_features=int(len(genes)),
        panel_features_not_in_reference=sorted(set(genes) - set(ref.var_names)),
        n_reference_cells=int(ref.n_obs), n_anchor_cells=int(ref_anchor.n_obs),
        temperature=TEMP, low_confidence_threshold=LOW_CONF,
        min_tx=MIN_TX, max_tx=MAX_TX), indent=2))
    a10.reset_index().to_parquet(OBJ / "r07_10x_labels.parquet", index=False)

    # ---- 3. arm tables -------------------------------------------------------
    arms = {
        "10x": (X10, id10, a10),
        "TRACER whole": (XTr[cls == "whole"], idTr[cls == "whole"],
                         aTr[cls == "whole"]),
        "TRACER partial": (XTr[cls == "partial"], idTr[cls == "partial"],
                           aTr[cls == "partial"]),
        "TRACER whole+partial": (XTr, idTr, aTr),
    }
    obs = {}
    for arm, (X, ids, a) in arms.items():
        lab = a.transferred_label.to_numpy().astype(str)
        sup, sup_strict, zero = marker_support(X, genes, lab)
        obs[arm] = pd.DataFrame(dict(
            entity=ids, arm=arm, label=lab,
            conf=a.transfer_confidence.to_numpy(),
            n_tx=np.asarray(X.sum(1)).ravel(),
            marker_supported=sup, marker_supported_strict=sup_strict,
            no_marker_signal=zero))
    # published marker_supported must be reproduced for the TRACER arms
    mls = pd.read_parquet(BASE / "rcc_analysis/qc/marker_lineage_scores.parquet",
                          columns=["cell_id", "marker_supported"])
    mls["cell_id"] = mls.cell_id.astype(str)
    o = obs["TRACER whole+partial"]
    ms_agree = float((o.marker_supported.to_numpy() ==
                      mls.set_index("cell_id").marker_supported
                      .reindex(o.entity).to_numpy()).mean())
    log(f"marker_supported reproduction: {ms_agree:.6f}")

    rows = []
    for arm, o in obs.items():
        for ct in TYPES + ["ALL"]:
            d = o if ct == "ALL" else o[o.label == ct]
            rows.append(dict(
                arm=arm, cell_type=ct, n=len(d),
                frac_of_arm=len(d) / len(o),
                median_tx=float(d.n_tx.median()) if len(d) else np.nan,
                mean_conf=float(d.conf.mean()) if len(d) else np.nan,
                low_conf_frac=float((d.conf < LOW_CONF).mean()) if len(d) else np.nan,
                marker_supported=float(d.marker_supported.mean()) if len(d) else np.nan,
                marker_supported_strict=float(d.marker_supported_strict.mean())
                if len(d) else np.nan,
                no_marker_signal=float(d.no_marker_signal.mean()) if len(d) else np.nan))
    C = pd.DataFrame(rows)
    C["reference_n"] = C.cell_type.map(
        ref.obs.cell_type_coarse.value_counts().to_dict())
    C.to_csv(TAB / "r07_arm_counts.csv", index=False)
    log("arm counts:\n" + C[C.cell_type == "ALL"].to_string(index=False))

    # ---- 4. dot-plot statistics (same normalisation for every arm) ---------
    gi = {g: i for i, g in enumerate(genes)}
    mk = [g for ct in TYPES for g in CANONICAL[ct]]
    rows = []
    for arm, (X, ids, a) in arms.items():
        N = lognorm(X)[:, [gi[g] for g in mk]].toarray()
        lab = obs[arm].label.to_numpy()
        for ct in TYPES:
            m = lab == ct
            for j, g in enumerate(mk):
                v = N[m, j]
                rows.append(dict(arm=arm, cell_type=ct, gene=g, n=int(m.sum()),
                                 mean_expr=float(v.mean()) if m.any() else np.nan,
                                 frac_expr=float((v > 0).mean()) if m.any() else np.nan))
    pd.DataFrame(rows).to_csv(TAB / "r07_dotplot.csv", index=False)

    # ---- 5. shared embedding -----------------------------------------------
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import roc_auc_score
    import umap

    def pick(n, k):
        return np.sort(rng.choice(n, min(n, k), replace=False))

    pool_idx = {a: pick(arms[a][0].shape[0], N_POOL)
                for a in ("10x", "TRACER whole", "TRACER partial")}
    P = sp.vstack([lognorm(arms[a][0][i]) for a, i in pool_idx.items()]).toarray()
    mu, sd = P.mean(0), P.std(0); sd[sd < 1e-6] = 1

    def z(X):
        return np.clip((lognorm(X).toarray() - mu) / sd, -10, 10).astype(np.float32)

    pca = PCA(n_components=N_PC, random_state=SEED).fit(
        np.clip((P - mu) / sd, -10, 10))
    Zpool = pca.transform(np.clip((P - mu) / sd, -10, 10))
    del P
    um = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean",
                   random_state=SEED).fit(Zpool)
    log(f"embedding fitted on {Zpool.shape[0]:,} pooled entities "
        f"(PCA var explained {pca.explained_variance_ratio_.sum():.3f})")

    # display sets: each base arm N_SHOW (pool included); whole+partial is drawn
    # from the embedded whole and partial entities in its true proportion
    show, pcs = {}, {}
    for a in ("10x", "TRACER whole", "TRACER partial"):
        X = arms[a][0]
        idx = np.union1d(pool_idx[a], pick(X.shape[0], N_SHOW))[:N_SHOW] \
            if X.shape[0] > N_SHOW else np.arange(X.shape[0])
        Zs = pca.transform(z(X[idx]))
        pcs[a] = (idx, Zs)
        U = um.transform(Zs)
        show[a] = obs[a].iloc[idx].assign(umap1=U[:, 0], umap2=U[:, 1])
    fw = len(obs["TRACER whole"]) / len(obs["TRACER whole+partial"])
    nw = int(round(N_SHOW * fw))
    sw = show["TRACER whole"].sample(min(nw, len(show["TRACER whole"])), random_state=SEED)
    spp = show["TRACER partial"].sample(min(N_SHOW - nw, len(show["TRACER partial"])),
                                        random_state=SEED)
    show["TRACER whole+partial"] = pd.concat([sw, spp]).assign(arm="TRACER whole+partial")
    U = pd.concat(show.values(), ignore_index=True)
    U.to_parquet(OBJ / "r07_umap.parquet", index=False)

    # ---- 6. depth-matched control: whole cells thinned to partial depth ----
    Xw, idw = arms["TRACER whole"][0], arms["TRACER whole"][1]
    iw = pick(Xw.shape[0], N_KNN)
    ptot = obs["TRACER partial"].n_tx.to_numpy()
    Xt = thin(Xw[iw], rng.choice(ptot, len(iw)), rng)
    tt = np.asarray(Xt.sum(1)).ravel(); okt = tt >= MIN_TX
    Xt, idt = Xt[okt], idw[iw][okt]
    at = transfer(Xt, np.array([f"{i}__thin" for i in idt]))
    full_lab = obs["TRACER whole"].set_index("entity").label.reindex(idt).to_numpy()
    thin_lab = at.transferred_label.to_numpy().astype(str)
    stab = pd.DataFrame(dict(full=full_lab, thin=thin_lab))
    lab_stab = (stab.assign(same=stab.full == stab.thin)
                .groupby("full").same.agg(["mean", "size"]).reset_index()
                .rename(columns={"full": "cell_type", "mean": "label_kept_at_partial_depth",
                                 "size": "n"}))
    lab_stab.to_csv(TAB / "r07_label_stability_thinned.csv", index=False)
    log(f"whole cells thinned to partial depth: median {np.median(tt[okt]):.0f} tx; "
        f"label kept {float((stab.full == stab.thin).mean()):.3f}")

    # ---- 7. kNN label coherence + marker AUROC in the shared space --------
    def coherence(Zs, lab):
        nn = NearestNeighbors(n_neighbors=K + 1).fit(Zs)
        _, ind = nn.kneighbors(Zs)
        return (lab[ind[:, 1:]] == lab[:, None]).mean(1)

    def lineage_score(X):
        N = lognorm(X)
        return {ct: np.asarray(N[:, [gi[g] for g in CANONICAL[ct]]].mean(1)).ravel()
                for ct in TYPES}

    sets = {}
    for a in ("10x", "TRACER whole", "TRACER partial", "TRACER whole+partial"):
        X, ids, _ = arms[a]
        i = pick(X.shape[0], N_KNN)
        sets[a] = (X[i], obs[a].label.to_numpy()[i])
    sets["TRACER whole, thinned to partial depth"] = (Xt, thin_lab)
    rows, rows_auc = [], []
    for a, (X, lab) in sets.items():
        Zs = pca.transform(z(X))
        coh = coherence(Zs, lab)
        L = lineage_score(X)
        for ct in TYPES:
            m = lab == ct
            freq = float(m.mean())
            rows.append(dict(arm=a, cell_type=ct, n=int(m.sum()),
                             knn_same_label=float(coh[m].mean()) if m.any() else np.nan,
                             baseline_type_freq=freq,
                             enrichment=(float(coh[m].mean()) / freq)
                             if m.any() and freq > 0 else np.nan))
            if 0 < m.sum() < len(m):
                rows_auc.append(dict(arm=a, cell_type=ct, n=int(m.sum()),
                                     marker_auroc=float(roc_auc_score(m, L[ct]))))
    pd.DataFrame(rows).to_csv(TAB / "r07_knn_coherence.csv", index=False)
    pd.DataFrame(rows_auc).to_csv(TAB / "r07_marker_auroc.csv", index=False)

    # ---- 8. prespecified flags ----------------------------------------------
    F = C[C.cell_type != "ALL"].copy()
    F["flag_reference_small"] = F.reference_n < 150
    F["flag_marker_support_lt_0.6"] = F.marker_supported < 0.6
    F["flag_low_conf_gt_0.25"] = F.low_conf_frac > 0.25
    F["flag_n_lt_200"] = F.n < 200
    F["flagged"] = F.filter(like="flag_").any(axis=1)
    F.to_csv(TAB / "r07_flags.csv", index=False)
    (TAB / "r07_summary.json").write_text(json.dumps(dict(
        marker_supported_reproduction=ms_agree,
        pca_variance_explained=float(pca.explained_variance_ratio_.sum()),
        pool_per_arm=N_POOL, shown_per_arm=N_SHOW, knn_per_arm=N_KNN, k=K,
        umap=dict(n_neighbors=30, min_dist=0.3, random_state=SEED),
        thinned_whole_median_tx=float(np.median(tt[okt])),
        thinned_label_kept=float((stab.full == stab.thin).mean())), indent=2))
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
