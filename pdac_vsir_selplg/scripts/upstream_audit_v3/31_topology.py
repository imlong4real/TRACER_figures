#!/usr/bin/env python3
"""Checkpoint-niche topology: is there organisation beyond having more cells?

Graph: nodes are the checkpoint-positive cells of one axis (ligand-positive cells
of the anchor context and receptor-positive T cells); an edge joins any two nodes
within R um. A "mixed component" contains BOTH a ligand-positive and a
receptor-positive node and is the operational definition of a checkpoint domain.

Null: checkpoint status is permuted WITHIN each cell type, so the number of
positive cells, the cell-type map and all positions are preserved and only which
cells carry the marker is randomised. This separates organisation from the fact
that TRACER simply recovers more cells. The unadjusted counts are reported too,
because recovery is itself part of the TRACER effect.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"; O=D/"audit_v3"
PTS=["HC01","HC03","HC04","HC05","HC07","HC08"]
RESP={"HC05":"responder","HC07":"responder","HC08":"responder",
      "HC01":"non_responder","HC03":"non_responder","HC04":"non_responder"}
TUMOR=["Ductal cell type 1","Ductal cell type 2"]
RADII=[20,30,50,100]; R_PRIMARY=50; NPERM=200
AXES=[("TIGIT-NECTIN2 (tumour)","NECTIN2","TUMOR","TIGIT","T"),
      ("TIGIT-PVR (tumour)","PVR","TUMOR","TIGIT","T"),
      ("VISTA-PSGL1","VSIR","TAM","SELPLG","T"),
      ("GAL9-TIM3","LGALS9","TAM","HAVCR2","T")]

def load(P,arm):
    if arm=="pre": return pd.read_parquet(S/f"{P}_pre_spatial.parquet")
    if arm=="post_whole": return pd.read_parquet(S/f"{P}_post_whole_spatial.parquet")
    return pd.concat([pd.read_parquet(S/f"{P}_post_whole_spatial.parquet"),
                      pd.read_parquet(S/f"{P}_post_partial_spatial.parquet")],ignore_index=True)

def ctx(d,w):
    if w=="TUMOR": return d.cell_type.isin(TUMOR).to_numpy()
    if w=="TAM":   return (d.cell_type=="Macrophage cell").to_numpy()
    return (d.cell_type=="T cell").to_numpy()

def topo(xy,is_lig,R):
    """Components of the radius graph; mixed = contains both node classes."""
    n=len(xy)
    if n<2: return dict(n_nodes=n,n_edges=0,n_mixed_edges=0,n_comp=n,
                        n_mixed_comp=0,frac_in_mixed=0.0,largest_mixed=0,
                        largest_comp=1 if n else 0)
    t=cKDTree(xy); pairs=t.query_pairs(R,output_type="ndarray")
    if len(pairs)==0: return dict(n_nodes=n,n_edges=0,n_mixed_edges=0,n_comp=n,
                        n_mixed_comp=0,frac_in_mixed=0.0,largest_mixed=0,largest_comp=1)
    g=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n))
    nc,lab=connected_components(g,directed=False)
    mixed_edges=int((is_lig[pairs[:,0]]!=is_lig[pairs[:,1]]).sum())
    df=pd.DataFrame({"lab":lab,"lig":is_lig})
    gsz=df.groupby("lab").size()
    has_both=df.groupby("lab").lig.nunique()==2
    mixed_labs=set(has_both[has_both].index)
    in_mixed=np.isin(lab,list(mixed_labs))
    return dict(n_nodes=n,n_edges=int(len(pairs)),n_mixed_edges=mixed_edges,
                n_comp=int(nc),n_mixed_comp=int(len(mixed_labs)),
                frac_in_mixed=float(in_mixed.mean()),
                largest_mixed=int(gsz[list(mixed_labs)].max()) if mixed_labs else 0,
                largest_comp=int(gsz.max()))

rng=np.random.default_rng(1); rows=[]
for P in PTS:
    for arm in ["pre","post_whole","post_all"]:
        d=load(P,arm)
        for lab,lg,lc,rg,rc in AXES:
            if lg not in d.columns or rg not in d.columns: continue
            am=ctx(d,lc); tm=ctx(d,rc)
            lig=d[(d[lg].to_numpy()>0)&am]; rec=d[(d[rg].to_numpy()>0)&tm]
            if len(lig)<5 or len(rec)<5:
                rows.append({"patient":P,"response":RESP[P],"arm":arm,"axis":lab,"R":R_PRIMARY,
                             "n_lig":len(lig),"n_rec":len(rec),"status":"too_sparse"}); continue
            xy=np.vstack([lig[["x","y"]].to_numpy(),rec[["x","y"]].to_numpy()])
            is_lig=np.r_[np.ones(len(lig),bool),np.zeros(len(rec),bool)]
            for R in RADII:
                o=topo(xy,is_lig,R)
                r={"patient":P,"response":RESP[P],"arm":arm,"axis":lab,"R":R,
                   "n_lig":len(lig),"n_rec":len(rec),"status":"ok",**o}
                if R==R_PRIMARY:
                    # within-type permutation null
                    ai=np.flatnonzero(am); ti=np.flatnonzero(tm)
                    XY=d[["x","y"]].to_numpy()
                    nm=[]; nf=[]; nlm=[]
                    for _ in range(NPERM):
                        ls=rng.choice(ai,len(lig),replace=False)
                        rs=rng.choice(ti,len(rec),replace=False)
                        xy2=np.vstack([XY[ls],XY[rs]])
                        oo=topo(xy2,is_lig,R)
                        nm.append(oo["n_mixed_edges"]); nf.append(oo["frac_in_mixed"])
                        nlm.append(oo["largest_mixed"])
                    nm=np.array(nm); nf=np.array(nf); nlm=np.array(nlm)
                    r.update(null_mixed_edges=float(nm.mean()),null_mixed_edges_sd=float(nm.std()),
                             p_mixed_edges=float((nm>=o["n_mixed_edges"]).mean()),
                             enrich_mixed_edges=float(o["n_mixed_edges"]/nm.mean()) if nm.mean()>0 else np.nan,
                             null_frac_in_mixed=float(nf.mean()),
                             p_frac_in_mixed=float((nf>=o["frac_in_mixed"]).mean()),
                             null_largest_mixed=float(nlm.mean()),
                             p_largest_mixed=float((nlm>=o["largest_mixed"]).mean()))
                rows.append(r)
        print(f"[{P} {arm}] done",flush=True)
T=pd.DataFrame(rows); T.to_csv(O/"topology.tsv",sep="\t",index=False)
q=T[(T.axis=="TIGIT-NECTIN2 (tumour)")&(T.R==R_PRIMARY)&(T.status=="ok")]
print("\n=== TIGIT-NECTIN2 (tumour), R=50 um ===")
print(q[["patient","response","arm","n_lig","n_rec","n_mixed_edges","null_mixed_edges",
         "enrich_mixed_edges","p_mixed_edges","n_mixed_comp","frac_in_mixed","largest_mixed",
         "p_largest_mixed"]].to_string(index=False,float_format=lambda v:f"{v:,.3f}"))
