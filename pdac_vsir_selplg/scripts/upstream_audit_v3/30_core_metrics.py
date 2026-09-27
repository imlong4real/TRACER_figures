#!/usr/bin/env python3
"""Core metrics for the redesigned panels.

  tissue area          occupied 100 um tiles x tile area (per patient, arm-invariant)
  densities            T cells and TAMs per mm2, per arm
  evaluable axes       axes with >=10 anchor-positive AND >=10 target-positive cells
  TIGIT-NECTIN2 niche  % NECTIN2+ tumour with a TIGIT+ T within 20 um, and the
                       median NECTIN2+ tumour -> nearest TIGIT+ T distance
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"; O=D/"audit_v3"
PTS=["HC01","HC03","HC04","HC05","HC07","HC08"]
RESP={"HC05":"responder","HC07":"responder","HC08":"responder",
      "HC01":"non_responder","HC03":"non_responder","HC04":"non_responder"}
TUMOR=["Ductal cell type 1","Ductal cell type 2"]
TILE=100.0; R_CONTACT=20.0
AXES=[("TIGIT-NECTIN2 (tumour)","NECTIN2","TUMOR","TIGIT","T"),
      ("TIGIT-PVR (tumour)","PVR","TUMOR","TIGIT","T"),
      ("TIGIT-NECTIN2 (TAM)","NECTIN2","TAM","TIGIT","T"),
      ("TIGIT-PVR (TAM)","PVR","TAM","TIGIT","T"),
      ("GAL9-TIM3","LGALS9","TAM","HAVCR2","T"),
      ("VISTA-PSGL1","VSIR","TAM","SELPLG","T"),
      ("VISTA-VSIG3","VSIR","TAM","IGSF11","TUMOR"),
      ("VSIG4-CD5","VSIG4","TAM","CD5","T"),
      ("PD-L1-PD-1 (tumour)","CD274","TUMOR","PDCD1","T"),
      ("PD-L1-PD-1 (TAM)","CD274","TAM","PDCD1","T")]

def load(P,arm):
    if arm=="pre": return pd.read_parquet(S/f"{P}_pre_spatial.parquet")
    if arm=="post_whole": return pd.read_parquet(S/f"{P}_post_whole_spatial.parquet")
    return pd.concat([pd.read_parquet(S/f"{P}_post_whole_spatial.parquet"),
                      pd.read_parquet(S/f"{P}_post_partial_spatial.parquet")],ignore_index=True)

def ctx(d,which):
    if which=="TUMOR": return d.cell_type.isin(TUMOR).to_numpy()
    if which=="TAM":   return (d.cell_type=="Macrophage cell").to_numpy()
    return (d.cell_type=="T cell").to_numpy()

rows_d=[]; rows_a=[]; rows_n=[]
for P in PTS:
    ref=load(P,"post_all")
    tiles=set(zip(np.floor(ref.x/TILE).astype(int),np.floor(ref.y/TILE).astype(int)))
    area_mm2=len(tiles)*(TILE/1000.0)**2
    for arm in ["pre","post_whole","post_all"]:
        d=load(P,arm)
        nT=int((d.cell_type=="T cell").sum()); nM=int((d.cell_type=="Macrophage cell").sum())
        rows_d.append({"patient":P,"response":RESP[P],"arm":arm,"tissue_area_mm2":area_mm2,
                       "n_T":nT,"n_TAM":nM,"T_per_mm2":nT/area_mm2,"TAM_per_mm2":nM/area_mm2})
        n_eval=0
        for lab,lg,lc,rg,rc in AXES:
            if lg not in d.columns or rg not in d.columns: continue
            a=int(((d[lg].to_numpy()>0)&ctx(d,lc)).sum())
            b=int(((d[rg].to_numpy()>0)&ctx(d,rc)).sum())
            ev=(a>=10)and(b>=10)
            n_eval+=int(ev)
            rows_a.append({"patient":P,"response":RESP[P],"arm":arm,"axis":lab,
                           "n_anchor_pos":a,"n_target_pos":b,"evaluable":ev})
        rows_d[-1]["n_evaluable_axes"]=n_eval
        # TIGIT-NECTIN2 niche metric
        tum=d[d.cell_type.isin(TUMOR)]; tc=d[d.cell_type=="T cell"]
        n2=tum[tum.NECTIN2>0]; tg=tc[tc.TIGIT>0]
        rec={"patient":P,"response":RESP[P],"arm":arm,
             "n_NECTIN2_tumour":len(n2),"n_TIGIT_T":len(tg)}
        if len(n2)>0 and len(tg)>0:
            tr=cKDTree(tg[["x","y"]].to_numpy())
            dist,_=tr.query(n2[["x","y"]].to_numpy(),k=1)
            rec["pct_NECTIN2_with_TIGIT_within20"]=100.0*float((dist<=R_CONTACT).mean())
            rec["median_dist_NECTIN2_to_TIGIT_um"]=float(np.median(dist))
        else:
            rec["pct_NECTIN2_with_TIGIT_within20"]=np.nan
            rec["median_dist_NECTIN2_to_TIGIT_um"]=np.nan
        rows_n.append(rec)
    print(f"[{P}] area {area_mm2:.1f} mm2",flush=True)

pd.DataFrame(rows_d).to_csv(O/"density_and_evaluable.tsv",sep="\t",index=False)
pd.DataFrame(rows_a).to_csv(O/"axis_evaluability.tsv",sep="\t",index=False)
pd.DataFrame(rows_n).to_csv(O/"tigit_nectin2_niche.tsv",sep="\t",index=False)
print("\n=== densities and evaluable axes ===")
print(pd.DataFrame(rows_d)[["patient","arm","tissue_area_mm2","T_per_mm2","TAM_per_mm2","n_evaluable_axes"]]
      .to_string(index=False,float_format=lambda v:f"{v:,.1f}"))
print("\n=== TIGIT-NECTIN2 niche ===")
print(pd.DataFrame(rows_n).to_string(index=False,float_format=lambda v:f"{v:,.2f}"))
