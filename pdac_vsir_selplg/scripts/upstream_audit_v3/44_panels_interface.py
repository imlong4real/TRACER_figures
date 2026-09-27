#!/usr/bin/env python3
"""P7 TIGIT-NECTIN2 interface ROI; P9 VISTA-PSGL1 niche topology (metric + space).

Interface ROIs are chosen reproducibly: among 400 um tiles, the one with the most
ligand+/receptor+ pairs within the prespecified 20 um contact radius (ties broken
by receptor-positive count). All candidate tiles are written out.
"""
import sys; sys.path.insert(0,"/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1/audit_v3")
from style import *
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from pathlib import Path

D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"; AFF=json.load(open(D/"audit_v2/image_affines.json"))
SZ=400.0; R_CONTACT=20.0; R_NICHE=50.0
def to_img(P,x,y):
    a=AFF[P];cx,cy=a["px"],a["py"];s=a["hires_scalef"]
    return (cx[0]*x+cx[1]*y+cx[2])*s,(cy[0]*x+cy[1]*y+cy[2])*s
def he(P):
    """uint8 RGB: mpimg returns float32 RGBA (~400 MB for a 6000 px section) and
    the shared 5 GB login cgroup cannot hold several copies."""
    a=mpimg.imread(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/binned_outputs/square_002um/spatial/tissue_hires_image.png")
    if a.dtype!=np.uint8: a=(np.clip(a[...,:3],0,1)*255).astype(np.uint8)
    else: a=a[...,:3]
    return np.ascontiguousarray(a)
def load(P,arm):
    if arm=="pre": return pd.read_parquet(S/f"{P}_pre_spatial.parquet")
    if arm=="post_whole": return pd.read_parquet(S/f"{P}_post_whole_spatial.parquet")
    return pd.concat([pd.read_parquet(S/f"{P}_post_whole_spatial.parquet"),
                      pd.read_parquet(S/f"{P}_post_partial_spatial.parquet")],ignore_index=True)
def sets(d,lg,lctx,rg,rctx):
    tum=d.cell_type.isin(TUMOR); tam=d.cell_type=="Macrophage cell"; tc=d.cell_type=="T cell"
    ctxm={"TUMOR":tum,"TAM":tam,"T":tc}
    return d[(d[lg]>0)&ctxm[lctx]], d[(d[rg]>0)&ctxm[rctx]]

def pick_roi(P,lg,lctx,rg,rctx,R):
    d=load(P,"post_all"); L,Rr=sets(d,lg,lctx,rg,rctx)
    if len(L)==0 or len(Rr)==0: return None,None
    t=cKDTree(Rr[["x","y"]].to_numpy())
    pairs=t.query_ball_point(L[["x","y"]].to_numpy(),R)
    rows=[]
    for i,(_,row) in enumerate(L.iterrows()):
        if len(pairs[i]):
            rows.append((int(row.x//SZ),int(row.y//SZ),len(pairs[i])))
    if not rows: return None,None
    g=pd.DataFrame(rows,columns=["tx","ty","n"]).groupby(["tx","ty"]).n.sum().reset_index()
    g=g.sort_values("n",ascending=False)
    return g, (float(g.iloc[0].tx*SZ),float(g.iloc[0].ty*SZ))

def draw(ax,P,img,x0,y0,d,L,Rr,lc,rc,R,links=True):
    cs=[to_img(P,x0+dx,y0+dy) for dx,dy in [(0,0),(SZ,0),(0,SZ),(SZ,SZ)]]
    c0=min(c[0] for c in cs);c1=max(c[0] for c in cs);r0=min(c[1] for c in cs);r1=max(c[1] for c in cs)
    ax.imshow(img,zorder=0); ax.set_xlim(c0,c1); ax.set_ylim(r1,r0)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)
    def crop(z): return z[(z.x>=x0)&(z.x<x0+SZ)&(z.y>=y0)&(z.y<y0+SZ)]
    bg=crop(d)
    for ct,col,s in [("Ductal cell type 2",CT["Ductal cell type 2"],1.4),
                     ("Ductal cell type 1",CT["Ductal cell type 1"],1.4),
                     ("Macrophage cell",CT["Macrophage cell"],1.8),
                     ("T cell",CT["T cell"],1.8)]:
        z=bg[bg.cell_type==ct]
        if not len(z): continue
        ix,iy=to_img(P,z.x.to_numpy(),z.y.to_numpy())
        ax.scatter(ix,iy,s=s,c=col,lw=0,alpha=0.42,zorder=2,rasterized=True)
    cl,cr=crop(L),crop(Rr)
    if links and len(cl) and len(cr):
        t=cKDTree(cr[["x","y"]].to_numpy()); segs=[]
        for _,row in cl.iterrows():
            for j in t.query_ball_point([row.x,row.y],R):
                ax1,ay1=to_img(P,row.x,row.y); ax2,ay2=to_img(P,cr.iloc[j].x,cr.iloc[j].y)
                segs.append([(ax1,ay1),(ax2,ay2)])
        if segs: ax.add_collection(LineCollection(segs,colors="#000000",linewidths=0.7,alpha=0.85,zorder=6))
    if len(cl):
        ix,iy=to_img(P,cl.x.to_numpy(),cl.y.to_numpy())
        ax.scatter(ix,iy,s=30,facecolors=lc,edgecolors="white",lw=0.7,zorder=7)
    if len(cr):
        ix,iy=to_img(P,cr.x.to_numpy(),cr.y.to_numpy())
        ax.scatter(ix,iy,s=24,marker="D",facecolors=rc,edgecolors="white",lw=0.6,zorder=6)
    return len(cl),len(cr)

# ---------------- P7 TIGIT-NECTIN2 interface ----------------------------
cand,best=pick_roi("HC08","NECTIN2","TUMOR","TIGIT","T",R_CONTACT)
if cand is not None:
    cand.to_csv(D/"audit_v3/roi_candidates_TIGIT_NECTIN2.tsv",sep="\t",index=False)
    P="HC08"; x0,y0=best; img=he(P)
    fig,axs=plt.subplots(1,3,figsize=(5.4,1.95),gridspec_kw={"wspace":0.04})
    for j,arm in enumerate(ARMS):
        d=load(P,arm); L,Rr=sets(d,"NECTIN2","TUMOR","TIGIT","T")
        nl,nr=draw(axs[j],P,img,x0,y0,d,L,Rr,NECTIN2_C,TIGIT_C,R_CONTACT)
        axs[j].set_xlabel(f"{ARMLAB[arm]}\nNECTIN2$^+$={nl}  TIGIT$^+$={nr}",fontsize=6.5,labelpad=2)
    save(fig,"P7_TIGIT_NECTIN2_interface"); plt.close(fig)

# ---------------- P9a VISTA-PSGL1 enrichment ----------------------------
T=pd.read_csv(D/"audit_v3/topology.tsv",sep="\t")
q=T[(T.axis=="VISTA-PSGL1")&(T.R==50)&(T.status=="ok")]
fig,ax=plt.subplots(figsize=(2.35,2.0))
for p in PTS:
    s=q[q.patient==p].set_index("arm").reindex(ARMS)
    ax.plot([0,1,2],s.enrich_mixed_edges.to_numpy(),marker="o",ms=3.4,lw=1.1,
            color=RC[RESP[p]],alpha=0.9,ls="-" if RESP[p] else (0,(3,1.6)))
ax.axhline(1,color="k",lw=0.6,ls=":")
ax.set_xticks([0,1,2]); ax.set_xticklabels([ARMLAB[a] for a in ARMS])
ax.set_ylabel("VISTA–PSGL-1 edge\nenrichment vs null")
ax.set_xlim(-0.25,2.25)
save(fig,"P9a_vista_psgl1_enrichment"); plt.close(fig)

# ---------------- P9b VISTA-PSGL1 domain in space -----------------------
cand2,best2=pick_roi("HC01","VSIR","TAM","SELPLG","T",R_NICHE)
if cand2 is not None:
    cand2.to_csv(D/"audit_v3/roi_candidates_VISTA_PSGL1.tsv",sep="\t",index=False)
    P="HC01"; x0,y0=best2; img=he(P)
    fig,axs=plt.subplots(1,3,figsize=(5.4,1.95),gridspec_kw={"wspace":0.04})
    for j,arm in enumerate(ARMS):
        d=load(P,arm); L,Rr=sets(d,"VSIR","TAM","SELPLG","T")
        nl,nr=draw(axs[j],P,img,x0,y0,d,L,Rr,CT["Macrophage cell"],CT["T cell"],R_NICHE)
        axs[j].set_xlabel(f"{ARMLAB[arm]}\nVISTA$^+$={nl}  PSGL-1$^+$={nr}",fontsize=6.5,labelpad=2)
    save(fig,"P9b_vista_psgl1_domain_roi"); plt.close(fig)
    # whole-tissue view with the ROI boxed
    fig,ax=plt.subplots(figsize=(2.4,1.95))
    d=load(P,"post_all"); L,Rr=sets(d,"VSIR","TAM","SELPLG","T")
    ix,iy=to_img(P,d.x.to_numpy(),d.y.to_numpy())
    ax.imshow(img,zorder=0)
    pad=0.03*(ix.max()-ix.min())
    ax.set_xlim(ix.min()-pad,ix.max()+pad); ax.set_ylim(iy.max()+pad,iy.min()-pad)
    lx,ly=to_img(P,L.x.to_numpy(),L.y.to_numpy()); ax.scatter(lx,ly,s=2.6,c=CT["Macrophage cell"],lw=0,alpha=0.85,zorder=3)
    rx,ry=to_img(P,Rr.x.to_numpy(),Rr.y.to_numpy()); ax.scatter(rx,ry,s=2.6,c=CT["T cell"],lw=0,alpha=0.85,zorder=3)
    cs=[to_img(P,x0+dx,y0+dy) for dx,dy in [(0,0),(SZ,0),(0,SZ),(SZ,SZ)]]
    c0=min(c[0] for c in cs);c1=max(c[0] for c in cs);r0=min(c[1] for c in cs);r1=max(c[1] for c in cs)
    ax.add_patch(Rectangle((c0,r0),c1-c0,r1-r0,fill=False,ec="#000000",lw=1.1,zorder=9))
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)
    save(fig,"P9c_vista_psgl1_wholetissue"); plt.close(fig)

fig,ax=plt.subplots(figsize=(5.6,0.4)); ax.axis("off")
ax.legend(handles=[Line2D([],[],marker="o",ls="",mfc=NECTIN2_C,mec="white",mew=0.8,ms=6,label="NECTIN2$^+$ tumour"),
                   Line2D([],[],marker="D",ls="",mfc=TIGIT_C,mec="white",mew=0.8,ms=6,label="TIGIT$^+$ T"),
                   Line2D([],[],marker="o",ls="",mfc="none",mec=CT["Macrophage cell"],mew=1.1,ms=6,label="VISTA$^+$ TAM"),
                   Line2D([],[],marker="D",ls="",mfc="none",mec=CT["T cell"],mew=1.1,ms=6,label="PSGL-1$^+$ T"),
                   Line2D([],[],color="#000000",lw=0.9,label="pair within radius")],
          frameon=False,loc="center",ncol=5,handlelength=1.2)
save(fig,"P7b_legend"); plt.close(fig)
print("[interface panels done]",flush=True)
