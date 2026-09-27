#!/usr/bin/env python3
"""P1/P2: whole-section H&E with semi-transparent cell-type domains + ROI boxes.

Domains are rasterised: the section is binned at 25 um and each bin is coloured by
its dominant transferred cell type with alpha proportional to local density, so
the domain structure reads at whole-tissue scale instead of drawing 10^5 markers.
"""
import sys; sys.path.insert(0,"/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1/audit_v3")
from style import *
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"; AFF=json.load(open(D/"audit_v2/image_affines.json"))
GRID=25.0
SHOW=["T cell","Macrophage cell","Ductal cell type 2","Fibroblast cell"]

def to_img(P,x,y):
    a=AFF[P];cx,cy=a["px"],a["py"];s=a["hires_scalef"]
    return (cx[0]*x+cx[1]*y+cx[2])*s,(cy[0]*x+cy[1]*y+cy[2])*s
def he(P):
    return mpimg.imread(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/binned_outputs/square_002um/spatial/tissue_hires_image.png")
def load(P,arm):
    if arm=="pre": return pd.read_parquet(S/f"{P}_pre_spatial.parquet")
    if arm=="post_whole": return pd.read_parquet(S/f"{P}_post_whole_spatial.parquet")
    return pd.concat([pd.read_parquet(S/f"{P}_post_whole_spatial.parquet"),
                      pd.read_parquet(S/f"{P}_post_partial_spatial.parquet")],ignore_index=True)

def domains(ax,P,d):
    """One semi-transparent layer per cell type, rasterised on a 25 um grid."""
    gx=np.floor(d.x/GRID).astype(int); gy=np.floor(d.y/GRID).astype(int)
    key=gx.astype(np.int64)*100000+gy
    tot=pd.Series(1,index=key).groupby(level=0).size()
    for ct in SHOW:
        m=(d.cell_type==ct).to_numpy()
        if m.sum()==0: continue
        cnt=pd.Series(1,index=key[m]).groupby(level=0).size()
        frac=(cnt/tot.reindex(cnt.index)).fillna(0)
        # keep bins where this type dominates locally
        sel=frac[frac>=0.34]
        if not len(sel): continue
        kx=(sel.index//100000)*GRID+GRID/2; ky=(sel.index%100000)*GRID+GRID/2
        ix,iy=to_img(P,kx.to_numpy(),ky.to_numpy())
        base={"Fibroblast cell":0.05,"Ductal cell type 2":0.14}.get(ct,0.30)
        gain={"Fibroblast cell":0.14,"Ductal cell type 2":0.34}.get(ct,0.62)
        a=np.clip(base+gain*sel.to_numpy(),0,0.85)
        sz={"Fibroblast cell":1.0,"Ductal cell type 2":1.2}.get(ct,1.7)
        ax.scatter(ix,iy,s=sz,c=CT[ct],lw=0,alpha=a,rasterized=True,
                   zorder=3 if ct in ("T cell","Macrophage cell") else 2)

def frame(ax,P,img,d):
    ix,iy=to_img(P,d.x.to_numpy(),d.y.to_numpy())
    pad=0.03*(ix.max()-ix.min())
    ax.imshow(img,zorder=0)
    ax.set_xlim(ix.min()-pad,ix.max()+pad); ax.set_ylim(iy.max()+pad,iy.min()-pad)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)

roi=pd.read_csv(D/"audit_v2/roi_top_per_patient.tsv",sep="\t")
for P,tag in [("HC05","P1_wholetissue_HC05_responder"),
              ("HC01","P2_wholetissue_HC01_nonresponder")]:
    img=he(P); ref=load(P,"post_all")
    fig,axs=plt.subplots(1,4,figsize=(7.0,1.95),gridspec_kw={"wspace":0.04})
    for j,(arm,lab) in enumerate([(None,"H&E"),("pre","pre"),("post_whole","whole"),
                                  ("post_all","whole+partial")]):
        ax=axs[j]; frame(ax,P,img,ref)
        if arm is not None: domains(ax,P,load(P,arm))
        ax.set_xlabel(lab,fontsize=7,labelpad=2)
        for _,r in roi[roi.patient==P].head(1).iterrows():
            cs=[to_img(P,r.x0+dx,r.y0+dy) for dx,dy in [(0,0),(400,0),(0,400),(400,400)]]
            c0=min(c[0] for c in cs); c1=max(c[0] for c in cs)
            r0=min(c[1] for c in cs); r1=max(c[1] for c in cs)
            ax.add_patch(Rectangle((c0,r0),c1-c0,r1-r0,fill=False,ec="#000000",lw=1.1,zorder=9))
    save(fig,tag); plt.close(fig)

fig,ax=plt.subplots(figsize=(4.4,0.4)); ax.axis("off")
ax.legend(handles=[Line2D([],[],marker="s",ls="",mfc=CT[c],mec="none",ms=6,
                          label={"Ductal cell type 2":"tumour","Macrophage cell":"TAM",
                                 "Fibroblast cell":"fibroblast"}.get(c,c)) for c in SHOW],
          frameon=False,loc="center",ncol=4,handlelength=1.0)
save(fig,"P1b_legend"); plt.close(fig)
print("[whole-tissue panels done]",flush=True)
