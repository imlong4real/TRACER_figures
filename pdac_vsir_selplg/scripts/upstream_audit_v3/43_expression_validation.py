#!/usr/bin/env python3
"""P5: does the checkpoint-positive state carry the expected lineage identity?

Per patient, cells are grouped into NECTIN2+/- tumour and TIGIT+/- T cells and a
PSEUDOBULK profile is formed per patient x group (sum of counts, CP10k, log1p).
The dot plot shows the across-patient mean; the replicate unit is the patient, so
no cell-level pseudo-replication enters. Differences are described, not tested:
with n = 6 and groups defined by the marker itself, a formal DE test would be
circular for the defining gene and underpowered for the rest.
"""
import sys; sys.path.insert(0,"/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1/audit_v3")
from style import *
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
import matplotlib.pyplot as plt
from pathlib import Path

D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
SRC=Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
GENES=["EPCAM","KRT19","KRT8","CD3D","CD3E","CD2","PTPRC","CD68","C1QA",
       "NECTIN2","PVR","TIGIT","VSIR","SELPLG","CD274","PDCD1","LGALS9","HAVCR2"]
LAB={"EPCAM":"EPCAM","KRT19":"KRT19","KRT8":"KRT8","CD3D":"CD3D","CD3E":"CD3E","CD2":"CD2",
     "PTPRC":"PTPRC","CD68":"CD68","C1QA":"C1QA","NECTIN2":"NECTIN2","PVR":"PVR",
     "TIGIT":"TIGIT","VSIR":"VISTA","SELPLG":"PSGL-1","CD274":"PD-L1","PDCD1":"PD-1",
     "LGALS9":"GAL-9","HAVCR2":"TIM-3"}
GROUPS=["NECTIN2$^+$ tumour","NECTIN2$^-$ tumour","TIGIT$^+$ T","TIGIT$^-$ T"]
rows=[]
for P in PTS:
    lab=pd.read_parquet(D/f"intermediate/{P}/labels_post_whole.parquet",
                        columns=["entity_id","cell_type"])
    with h5py.File(SRC/P/"data/tracer_results/outputs/cell_by_gene_tracer.h5ad","r") as f:
        shape=tuple(f["X"].attrs["shape"])
        g=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in f["var/feature_name"][:]],dtype=object)
        ids=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in f["obs/_index"][:]],dtype=object)
        gi=pd.Series(np.arange(len(g)),index=g)
        cols=[int(gi[x]) for x in GENES if x in gi.index]
        keep=[x for x in GENES if x in gi.index]
        X=sp.csr_matrix((f["X/data"][:],f["X/indices"][:],f["X/indptr"][:]),shape=shape)
    tot=np.asarray(X.sum(1)).ravel()
    sub=np.asarray(X[:,cols].todense())
    pos=pd.Series(np.arange(len(ids)),index=ids)
    idx=pos.reindex(lab.entity_id.astype(str)).to_numpy()
    ct=lab.cell_type.to_numpy()
    n2=sub[idx][:,keep.index("NECTIN2")]; tg=sub[idx][:,keep.index("TIGIT")]
    tum=np.isin(ct,TUMOR); tc=ct=="T cell"
    masks={"NECTIN2$^+$ tumour":tum&(n2>0),"NECTIN2$^-$ tumour":tum&(n2==0),
           "TIGIT$^+$ T":tc&(tg>0),"TIGIT$^-$ T":tc&(tg==0)}
    for gname,m in masks.items():
        if m.sum()<5: continue
        ii=idx[m]
        pb=sub[ii].sum(0); depth=tot[ii].sum()
        expr=np.log1p(pb/max(depth,1)*1e4)
        frac=(sub[ii]>0).mean(0)
        for k,gn in enumerate(keep):
            rows.append({"patient":P,"group":gname,"gene":gn,"expr":float(expr[k]),
                         "frac":float(frac[k]),"n_cells":int(m.sum())})
    print(f"[{P}] "+", ".join(f"{k.split('$')[0]}{'+' if '^+' in k else '-'}={int(v.sum())}"
                              for k,v in masks.items()),flush=True)
E=pd.DataFrame(rows); E.to_csv(D/"audit_v3/expression_validation.tsv",sep="\t",index=False)

m=E.groupby(["group","gene"]).agg(expr=("expr","mean"),frac=("frac","mean"),
                                  n_pat=("patient","nunique")).reset_index()
order=[g for g in GENES if g in set(m.gene)]
fig,ax=plt.subplots(figsize=(4.5,1.95))
zz=m.pivot(index="group",columns="gene",values="expr").reindex(GROUPS)[order]
z=(zz-zz.mean())/zz.std().replace(0,1)
ff=m.pivot(index="group",columns="gene",values="frac").reindex(GROUPS)[order]
for i,gr in enumerate(GROUPS):
    for j,gn in enumerate(order):
        ax.scatter(j,len(GROUPS)-1-i,s=6+150*ff.loc[gr,gn],c=[z.loc[gr,gn]],
                   cmap="RdBu_r",vmin=-1.6,vmax=1.6,lw=0.3,edgecolors="#444444")
ax.set_xticks(range(len(order))); ax.set_xticklabels([LAB[g] for g in order],rotation=45,ha="right",fontsize=6)
ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(GROUPS[::-1],fontsize=6.5)
ax.set_xlim(-0.7,len(order)-0.3); ax.set_ylim(-0.7,len(GROUPS)-0.3)
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
save(fig,"P5_expression_validation"); plt.close(fig)

fig,ax=plt.subplots(figsize=(3.6,0.58)); ax.axis("off")
h=[ax.scatter([],[],s=6+150*f,c="#BBBBBB",lw=0.3,edgecolors="#444444",label=f"{int(f*100)}%")
   for f in (0.1,0.4,0.8)]
l1=ax.legend(handles=h,frameon=False,loc="center left",bbox_to_anchor=(-0.02,0.45),
             ncol=3,title="cells expressing",title_fontsize=6,handlelength=1.0,
             handletextpad=0.6,columnspacing=1.9,borderpad=0.0)
import matplotlib as mpl
cax=fig.add_axes([0.68,0.34,0.30,0.26])
mpl.colorbar.ColorbarBase(cax,cmap=plt.get_cmap("RdBu_r"),
    norm=mpl.colors.Normalize(-1.6,1.6),orientation="horizontal")
cax.set_xticks([-1.5,0,1.5]); cax.tick_params(labelsize=5.5,length=2)
cax.set_xlabel("scaled mean",fontsize=6,labelpad=1)
save(fig,"P5b_legend"); plt.close(fig)
print("[expression validation done]",flush=True)
