"""Numbers behind the SegBench v2 figure map and claims table.

Per setting x method: median RCTD entropy / max weight with a 2,000-draw
bootstrap 95% CI (cells resampled), singlet / reject / unscored fractions.

Across the six main settings, each method is compared with Original by the
per-setting difference in medians.  With n = 6 settings the exact sign-flip
test floors at P = 2/64 = 0.031 (two-sided), so this is descriptive support,
not validation.

TRACER is also reported POOLED (whole + partial).  TRACER routes transcripts it
cannot place into partial entities, so TRACER whole is a selected subset and
its sharper calls are partly selection by construction; pooled-vs-Original is
the comparison that does not benefit from that selection.
"""
import itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import MAIN_ORDER, OBJ, SETTINGS

TAB = OBJ / "tables"
OUT = Path("/scratch4/adeshpa6/tracer_campaign/final/tables/segbench_v2")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260909)


def boot_median(v, n=2000):
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n, len(v))) if len(v) <= 20000 else None
    if idx is None:                                    # large n: subsample draws
        meds = [np.median(rng.choice(v, len(v))) for _ in range(400)]
    else:
        meds = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def signflip_p(d):
    d = np.asarray([x for x in d if np.isfinite(x)])
    obs = abs(d.mean())
    null = [abs((d * s).mean()) for s in itertools.product([-1, 1], repeat=len(d))]
    return float(np.mean(np.asarray(null) >= obs - 1e-12)), len(d)


rows = []
for s in SETTINGS:
    c = pd.read_parquet(TAB / f"percell__{s}.parquet")
    c = pd.concat([c, c[c.method.isin(["TRACER whole", "TRACER partial"])]
                   .assign(method="TRACER pooled")])
    for m, g in c.groupby("method"):
        sc = g[g.call != "unscored"]
        r = {"setting": s, "main": s in MAIN_ORDER, "method": m, "n_entities": len(g),
             "n_scored": len(sc)}
        for k in ("singlet", "reject", "unscored", "doublet_certain", "doublet_uncertain"):
            r[f"frac_{k}"] = float((g.call == k).mean())
        for metric in ("entropy", "max_weight"):
            med, lo, hi = boot_median(sc[metric].to_numpy())
            r[f"{metric}_median"], r[f"{metric}_lo"], r[f"{metric}_hi"] = med, lo, hi
        rows.append(r)
per = pd.DataFrame(rows)
per.to_csv(OUT / "per_setting_method.tsv", sep="\t", index=False)

cmp = []
main = per[per.main]
for m in sorted(set(main.method) - {"Original"}):
    for metric in ("max_weight_median", "entropy_median", "frac_singlet", "frac_unscored"):
        d = []
        for s in MAIN_ORDER:
            a = main[(main.setting == s) & (main.method == m)][metric]
            o = main[(main.setting == s) & (main.method == "Original")][metric]
            if len(a) and len(o):
                d.append(float(a.iloc[0] - o.iloc[0]))
        if not d:
            continue
        p, n = signflip_p(d)
        cmp.append({"method": m, "metric": metric, "n_settings": n,
                    "median_delta_vs_original": float(np.median(d)),
                    "min_delta": float(np.min(d)), "max_delta": float(np.max(d)),
                    "n_positive": int(np.sum(np.asarray(d) > 0)),
                    "signflip_p_two_sided": p,
                    "deltas": ";".join(f"{x:.3f}" for x in d)})
cmp = pd.DataFrame(cmp)
cmp.to_csv(OUT / "vs_original_main_settings.tsv", sep="\t", index=False)
pd.set_option("display.width", 220)
print(cmp[cmp.metric.isin(["max_weight_median", "entropy_median", "frac_unscored"])]
      .round(3).to_string(index=False))
