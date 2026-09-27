#!/usr/bin/env python3
"""Add the missing checkpoint markers to the PRE arm so pre/post is like-for-like."""
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"
NEED=["TIGIT","NECTIN2","PVR","VSIR","SELPLG","VSIG4","CD5",
      "LGALS9","HAVCR2","CD274","PDCD1","IGSF11"]
for P in ["HC01","HC03","HC04","HC05","HC07","HC08"]:
    f=S/f"{P}_pre_spatial.parquet"; d=pd.read_parquet(f)
    miss=[g for g in NEED if g not in d.columns]
    if not miss: print(f"[{P}] complete"); continue
    X=sp.load_npz(D/f"intermediate/{P}/arm_pre_X.npz").tocsr()
    genes=np.load(D/f"intermediate/{P}/genes.npy",allow_pickle=True)
    gi=pd.Series(np.arange(len(genes)),index=genes)
    meta=pd.read_parquet(D/f"intermediate/{P}/arm_pre_meta.parquet")
    pos=pd.Series(np.arange(len(meta)),index=meta.entity_id.astype(str))
    idx=pos.reindex(d.entity_id.astype(str)).to_numpy()
    ok=~pd.isna(idx); idx2=np.where(ok,idx,0).astype(int)
    added=[]
    for g in miss:
        if g in gi.index:
            v=np.asarray(X[idx2,int(gi[g])].todense()).ravel(); v[~ok]=0
            d[g]=v; added.append(g)
    d.to_parquet(f,index=False)
    print(f"[{P}] added {added}",flush=True)
print("[done]",flush=True)
