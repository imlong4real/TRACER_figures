#!/usr/bin/env python3
"""P3/P7: ROI zoom. Vendor polygons pre-TRACER, TRACER entity footprints post.

TRACER footprints are CONVEX HULLS of each entity's assigned transcripts. They are
inferred footprints, not measured membranes, and are drawn as such (thin outline,
no fill for partials). Vendor outlines are true segmentation polygons.

H&E resolution note: the highest-resolution image shipped with these sections is
~4.5 um per pixel, so the underlay shows tissue architecture, not cell membranes.
"""
import sys; sys.path.insert(0,"/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1/audit_v3")
from style import *
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, cKDTree
import pyarrow.parquet as pq
from pathlib import Path

D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"; AFF=json.load(open(D/"audit_v2/image_affines.json"))
SRC=Path("/scratch4/adeshpa6/PDAC_Long/cirro_tracer_seg_production")
SZ=400.0
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

def roi_transcripts(P,x0,y0):
    f=SRC/P/"data/tracer_results/outputs/transcripts_tracer_refined.parquet"
    pf=pq.ParquetFile(f); out=[]
    for i in range(pf.metadata.num_row_groups):
        t=pf.read_row_group(i,columns=["x","y","cell_id","tracer_id","_etype"])
        x=t.column("x").to_numpy(); y=t.column("y").to_numpy()
        m=(x>=x0)&(x<x0+SZ)&(y>=y0)&(y<y0+SZ)
        if m.any():
            out.append(pd.DataFrame({"x":x[m],"y":y[m],
                "cell_id":t.column("cell_id").to_pandas()[m].astype(str).to_numpy(),
                "tracer_id":t.column("tracer_id").to_pandas()[m].astype(str).to_numpy(),
                "etype":t.column("_etype").to_pandas()[m].astype(str).to_numpy()}))
        del t
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def vendor_polys(P,x0,y0):
    """Vendor cell polygons intersecting the ROI, in hires pixels."""
    a=AFF[P]; s=a["hires_scalef"]
    cs=[to_img(P,x0+dx,y0+dy) for dx,dy in [(0,0),(SZ,0),(0,SZ),(SZ,SZ)]]
    c0=min(c[0] for c in cs);c1=max(c[0] for c in cs)
    r0=min(c[1] for c in cs);r1=max(c[1] for c in cs)
    out={}
    with open(f"/scratch4/adeshpa6/PDAC_Long/{P}/data/segmented_outputs/cell_segmentations.geojson") as f:
        g=json.load(f)
    for ft in g["features"]:
        c=np.asarray(ft["geometry"]["coordinates"][0],float)*s
        if c[:,0].max()<c0 or c[:,0].min()>c1 or c[:,1].max()<r0 or c[:,1].min()>r1: continue
        out[int(ft["properties"]["cell_id"])]=c
    return out,(c0,c1,r0,r1)

def hulls(df,key,P):
    polys=[]
    for k,g in df.groupby(key):
        if len(g)<3: continue
        pts=g[["x","y"]].to_numpy()
        try: h=ConvexHull(pts)
        except Exception: continue
        ix,iy=to_img(P,pts[h.vertices,0],pts[h.vertices,1])
        polys.append(np.column_stack([ix,iy]))
    return polys

def frame(ax,img,box):
    c0,c1,r0,r1=box
    ax.imshow(img,zorder=0); ax.set_xlim(c0,c1); ax.set_ylim(r1,r0)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)

roi=pd.read_csv(D/"audit_v2/roi_top_per_patient.tsv",sep="\t").groupby("patient").head(1)
for P,tag in [("HC05","P3_roi_HC05_responder"),("HC01","P3b_roi_HC01_nonresponder")]:
    r=roi[roi.patient==P].iloc[0]; x0,y0=float(r.x0),float(r.y0)
    img=he(P); tx=roi_transcripts(P,x0,y0)
    vp,box=vendor_polys(P,x0,y0)
    pre=pd.read_parquet(S/f"{P}_pre_spatial.parquet")
    pw=pd.read_parquet(S/f"{P}_post_whole_spatial.parquet")
    pp=pd.read_parquet(S/f"{P}_post_partial_spatial.parquet")
    def crop(d): return d[(d.x>=x0)&(d.x<x0+SZ)&(d.y>=y0)&(d.y<y0+SZ)]
    cpre,cpw,cpp=crop(pre),crop(pw),crop(pp)
    tmap={str(e):c for e,c in zip(cpre.entity_id,cpre.cell_type)}
    fig,axs=plt.subplots(1,4,figsize=(7.0,1.95),gridspec_kw={"wspace":0.04})
    frame(axs[0],img,box); axs[0].set_xlabel("H&E",fontsize=7,labelpad=2)
    # pre: vendor polygons filled by transferred type
    ax=axs[1]; frame(ax,img,box)
    verts=[];cols=[]
    for cid,c in vp.items():
        ct=tmap.get(f"cellid_{cid:09d}-1")
        verts.append(c); cols.append(CT.get(ct,"#FFFFFF00") if ct else "#FFFFFF00")
    ax.add_collection(PolyCollection(verts,facecolors=cols,edgecolors="#3A3A3A",
                                     linewidths=0.22,alpha=0.75,zorder=3))
    ax.set_xlabel(f"pre · 10X   T={int((cpre.cell_type=='T cell').sum())}",fontsize=7,labelpad=2)
    # post whole: TRACER hulls
    ax=axs[2]; frame(ax,img,box)
    wmap={str(e):c for e,c in zip(cpw.entity_id,cpw.cell_type)}
    sub=tx[(tx.etype=="cell")&(tx.tracer_id!="-1")]
    verts=[];cols=[]
    for k,g in sub.groupby("tracer_id"):
        if len(g)<3: continue
        pts=g[["x","y"]].to_numpy()
        try: h=ConvexHull(pts)
        except Exception: continue
        ix,iy=to_img(P,pts[h.vertices,0],pts[h.vertices,1])
        verts.append(np.column_stack([ix,iy])); cols.append(CT.get(wmap.get(k),"#FFFFFF00"))
    ax.add_collection(PolyCollection(verts,facecolors=cols,edgecolors="#3A3A3A",
                                     linewidths=0.22,alpha=0.75,zorder=3))
    ax.set_xlabel(f"whole · footprints   T={int((cpw.cell_type=='T cell').sum())}",fontsize=7,labelpad=2)
    # whole + partial
    ax=axs[3]; frame(ax,img,box)
    ax.add_collection(PolyCollection(verts,facecolors=cols,edgecolors="#3A3A3A",
                                     linewidths=0.22,alpha=0.62,zorder=3))
    pmap={str(e):c for e,c in zip(cpp.entity_id,cpp.cell_type)}
    sub=tx[(tx.etype=="partial")&(tx.tracer_id!="-1")]
    pv=[];pc=[]
    for k,g in sub.groupby("tracer_id"):
        if len(g)<3: continue
        pts=g[["x","y"]].to_numpy()
        try: h=ConvexHull(pts)
        except Exception: continue
        ix,iy=to_img(P,pts[h.vertices,0],pts[h.vertices,1])
        pv.append(np.column_stack([ix,iy])); pc.append(CT.get(pmap.get(k),"#FFFFFF00"))
    ax.add_collection(PolyCollection(pv,facecolors="none",edgecolors="#000000",
                                     linewidths=0.22,alpha=0.55,zorder=4))
    nTp=int((cpp.cell_type=='T cell').sum())
    ax.set_xlabel(f"+ partial hulls   +{nTp} T",fontsize=7,labelpad=2)
    save(fig,tag); plt.close(fig)
    print(f"  {P}: vendor polys {len(vp)}, ROI tx {len(tx):,}",flush=True)

fig,ax=plt.subplots(figsize=(6.4,0.4)); ax.axis("off")
ax.legend(handles=[Line2D([],[],marker="s",ls="",mfc=CT[c],mec="#3A3A3A",mew=0.4,ms=6,
                          label={"Ductal cell type 2":"tumour","Macrophage cell":"TAM",
                                 "Fibroblast cell":"fibroblast"}.get(c,c))
                   for c in ["T cell","Macrophage cell","Ductal cell type 2","Fibroblast cell"]]+
                  [Line2D([],[],marker="s",ls="",mfc=CT["Acinar cell"],mec="#3A3A3A",mew=0.4,ms=6,label="acinar"),
                   Line2D([],[],marker="s",ls="",mfc="none",mec="#000000",mew=0.9,ms=6,
                          label="partial (inferred hull)")],
          frameon=False,loc="center",ncol=6,handlelength=1.0)
save(fig,"P3c_legend"); plt.close(fig)
print("[roi panels done]",flush=True)
