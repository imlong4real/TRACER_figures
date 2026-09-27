#!/usr/bin/env python3
"""Quantitative panels: P4 inventory/evaluability, P6 TIGIT-NECTIN2 niche metric."""
import sys; sys.path.insert(0,"/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1/audit_v3")
from style import *
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")

d=pd.read_csv(D/"audit_v3/density_and_evaluable.tsv",sep="\t")
X=[0,1,2]
def paired(ax,col,ylab,logy=False):
    for p in PTS:
        s=d[d.patient==p].set_index("arm").reindex(ARMS)
        ax.plot(X,s[col].to_numpy(),marker="o",ms=3.4,lw=1.1,color=RC[RESP[p]],alpha=0.9,
                ls="-" if RESP[p] else (0,(3,1.6)))
    ax.set_xticks(X); ax.set_xticklabels([ARMLAB[a] for a in ARMS])
    ax.set_ylabel(ylab)
    if logy: ax.set_yscale("log")
    ax.set_xlim(-0.25,2.25)

# ---- P4a T cells / mm2 ; P4b TAM / mm2 ; P4c evaluable axes -------------
for name,col,ylab,logy in [("P4a_T_density","T_per_mm2","T cells mm$^{-2}$",True),
                           ("P4b_TAM_density","TAM_per_mm2","TAMs mm$^{-2}$",True),
                           ("P4c_evaluable_axes","n_evaluable_axes","Checkpoint axes detectable",False)]:
    fig,ax=plt.subplots(figsize=(2.05,1.95))
    paired(ax,col,ylab,logy)
    if col=="n_evaluable_axes":
        ax.set_ylim(-0.4,10.4); ax.set_yticks([0,2,4,6,8,10])
    save(fig,name); plt.close(fig)

# legend chip reused by the quantitative panels
fig,ax=plt.subplots(figsize=(1.5,0.42)); ax.axis("off")
ax.legend(handles=[Line2D([],[],color=RC[1],lw=1.4,ls="-",label="responder"),
                   Line2D([],[],color=RC[0],lw=1.4,ls=(0,(3,1.6)),label="non-responder")],
          frameon=False,loc="center",ncol=2,handlelength=1.8)
save(fig,"P4d_legend"); plt.close(fig)

# ---- P6 TIGIT-NECTIN2 niche metric -------------------------------------
n=pd.read_csv(D/"audit_v3/tigit_nectin2_niche.tsv",sep="\t")
fig,ax=plt.subplots(figsize=(2.35,2.0))
for p in PTS:
    s=n[n.patient==p].set_index("arm").reindex(ARMS)
    ax.plot(X,s.median_dist_NECTIN2_to_TIGIT_um.to_numpy(),marker="o",ms=3.4,lw=1.1,
            color=RC[RESP[p]],alpha=0.9,ls="-" if RESP[p] else (0,(3,1.6)))
ax.set_yscale("log"); ax.set_xticks(X); ax.set_xticklabels([ARMLAB[a] for a in ARMS])
ax.set_ylabel("NECTIN2$^+$ tumour to\nnearest TIGIT$^+$ T (µm)")
ax.set_xlim(-0.25,2.25)
save(fig,"P6a_nectin2_tigit_distance"); plt.close(fig)

fig,ax=plt.subplots(figsize=(2.35,2.0))
for p in PTS:
    s=n[n.patient==p].set_index("arm").reindex(ARMS)
    ax.plot(X,s.pct_NECTIN2_with_TIGIT_within20.to_numpy(),marker="o",ms=3.4,lw=1.1,
            color=RC[RESP[p]],alpha=0.9,ls="-" if RESP[p] else (0,(3,1.6)))
ax.set_xticks(X); ax.set_xticklabels([ARMLAB[a] for a in ARMS])
ax.set_ylabel("NECTIN2$^+$ tumour with\nTIGIT$^+$ T ≤20 µm (%)")
ax.set_xlim(-0.25,2.25)
save(fig,"P6b_nectin2_tigit_pct20"); plt.close(fig)

# ---- P8 patient x axis category heatmap ---------------------------------
ae=pd.read_csv(D/"audit_v3/axis_evaluability.tsv",sep="\t")
piv=ae.pivot_table(index=["axis","patient"],columns="arm",values="evaluable")
anc=ae.pivot_table(index=["axis","patient"],columns="arm",values="n_anchor_pos")
tgt=ae.pivot_table(index=["axis","patient"],columns="arm",values="n_target_pos")
def cat(row,a,t):
    def b(v): return bool(v) and not pd.isna(v)
    pre,wh,al=b(row.get("pre")),b(row.get("post_whole")),b(row.get("post_all"))
    if not (pre or wh or al): return 4                      # unsupported
    if pre and wh:
        ap=a.get("pre",0); ap=0 if pd.isna(ap) else ap
        aw=a.get("post_whole",0); aw=0 if pd.isna(aw) else aw
        return 1 if aw>=2*max(ap,1) else 0                  # strengthened / visible pre
    if (not pre) and wh: return 2                           # enabled post-whole
    if (not pre) and (not wh) and al: return 3              # enabled by partials
    if pre and (not wh) and (not al): return 0              # visible pre only
    return 4
axes_order=(ae[ae.arm=="post_all"].groupby("axis").n_anchor_pos.median()
            .sort_values(ascending=False).index.tolist())
M=np.zeros((len(axes_order),len(PTS)),int)
for i,ax_ in enumerate(axes_order):
    for j,p in enumerate(PTS):
        M[i,j]=cat(piv.loc[(ax_,p)],anc.loc[(ax_,p)],tgt.loc[(ax_,p)])
from matplotlib.colors import ListedColormap,BoundaryNorm
cmap=ListedColormap(CATC); norm=BoundaryNorm(np.arange(-0.5,5.5,1),cmap.N)
fig,ax=plt.subplots(figsize=(3.05,2.7))
ax.imshow(M,cmap=cmap,norm=norm,aspect="auto")
ax.set_xticks(range(len(PTS)))
ax.set_xticklabels(PTS,rotation=0)
for j,p in enumerate(PTS): ax.get_xticklabels()[j].set_color(RC[RESP[p]])
ax.set_yticks(range(len(axes_order))); ax.set_yticklabels(axes_order,fontsize=6)
ax.set_xticks(np.arange(-0.5,len(PTS),1),minor=True)
ax.set_yticks(np.arange(-0.5,len(axes_order),1),minor=True)
ax.grid(which="minor",color="w",lw=1.0); ax.tick_params(which="minor",length=0)
for sp in ax.spines.values(): sp.set_visible(False)
save(fig,"P8_axis_category_heatmap"); plt.close(fig)

fig,ax=plt.subplots(figsize=(4.6,0.4)); ax.axis("off")
ax.legend(handles=[plt.Rectangle((0,0),1,1,fc=CATC[i],ec="none",label=c) for i,c in enumerate(CATS)],
          frameon=False,loc="center",ncol=5,handlelength=1.1,columnspacing=1.0)
save(fig,"P8b_legend"); plt.close(fig)
print("[quant panels done]",flush=True)
