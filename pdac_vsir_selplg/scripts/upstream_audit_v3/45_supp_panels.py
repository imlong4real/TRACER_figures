#!/usr/bin/env python3
"""Supplementary panels: signed-distance boundary gradient, response screen,
null comparison, radius sensitivity, ROI candidates, label-confidence."""
import sys; sys.path.insert(0,"/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1/audit_v3")
from style import *
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, binary_closing
from pathlib import Path
D=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")
S=D/"audit_v2/spatial"
def load(P,arm):
    if arm=="pre": return pd.read_parquet(S/f"{P}_pre_spatial.parquet")
    if arm=="post_whole": return pd.read_parquet(S/f"{P}_post_whole_spatial.parquet")
    return pd.concat([pd.read_parquet(S/f"{P}_post_whole_spatial.parquet"),
                      pd.read_parquet(S/f"{P}_post_partial_spatial.parquet")],ignore_index=True)

# ---- S1 signed-distance T-cell gradient ---------------------------------
G=25.0; BINS=np.arange(-300,301,25)
rows=[]
for P in PTS:
    ref=load(P,"post_all")
    nx=int(np.ceil(ref.x.max()/G))+1; ny=int(np.ceil(ref.y.max()/G))+1
    for arm in ARMS:
        d=load(P,arm)
        tum=d[d.cell_type.isin(TUMOR)]
        M=np.zeros((nx,ny),bool)
        M[np.floor(tum.x/G).astype(int),np.floor(tum.y/G).astype(int)]=True
        M=binary_closing(M,structure=np.ones((3,3)))
        din=distance_transform_edt(M)*G; dout=distance_transform_edt(~M)*G
        signed=np.where(M,-din,dout)      # negative inside tumour domain
        tc=d[d.cell_type=="T cell"]
        if not len(tc): continue
        sv=signed[np.floor(tc.x/G).astype(int),np.floor(tc.y/G).astype(int)]
        # occupancy-normalised density: T cells per occupied grid cell in each band
        occ=signed[np.floor(d.x/G).astype(int),np.floor(d.y/G).astype(int)]
        hT,_=np.histogram(sv,bins=BINS); hA,_=np.histogram(occ,bins=BINS)
        fr=np.where(hA>=20,hT/np.maximum(hA,1),np.nan)   # drop low-support bands
        rows.append(pd.DataFrame({"patient":P,"arm":arm,
            "d":0.5*(BINS[:-1]+BINS[1:]),"frac":fr,"n_bins":hA}))
    print(f"[grad] {P}",flush=True)
GR=pd.concat(rows,ignore_index=True); GR.to_csv(D/"audit_v3/boundary_gradient.tsv",sep="\t",index=False)
fig,axs=plt.subplots(1,3,figsize=(6.0,1.9),sharey=True,gridspec_kw={"wspace":0.12})
for j,arm in enumerate(ARMS):
    ax=axs[j]
    for p in PTS:
        s=GR[(GR.patient==p)&(GR.arm==arm)]
        ax.plot(s.d,s.frac,lw=1.0,color=RC[RESP[p]],alpha=0.9,ls="-" if RESP[p] else (0,(3,1.6)))
    ax.axvline(0,color="k",lw=0.6,ls=":")
    ax.set_xlabel(f"{ARMLAB[arm]}\nsigned distance (µm)")
    if j==0: ax.set_ylabel("T cells per\noccupied bin")
save(fig,"S1_boundary_gradient",outdir="panels_v3/supp"); plt.close(fig)

# ---- S2 response screen at chance --------------------------------------
mult=json.load(open(D/"audit_v2/multiplicity_audit.json"))
b=pd.read_csv(D/"tables/responder_vs_nonresponder_candidate_biomarkers.tsv",sep="\t")
fig,ax=plt.subplots(figsize=(2.1,1.9))
ax.bar([0],[mult["n_at_floor"]],0.55,color="#0072B2")
ax.bar([1],[mult["expected_at_floor_under_null"]],0.55,color="#BDBDBD")
for i,v in enumerate([mult["n_at_floor"],mult["expected_at_floor_under_null"]]):
    ax.annotate(f"{v:.0f}",(i,v),textcoords="offset points",xytext=(0,2),ha="center",fontsize=6)
ax.set_xticks([0,1]); ax.set_xticklabels(["observed","chance"])
ax.set_ylabel("features fully\nseparating 3-vs-3"); ax.set_ylim(0,mult["n_at_floor"]*1.25)
save(fig,"S2_response_screen_chance",outdir="panels_v3/supp"); plt.close(fig)

# ---- S3 global vs block null -------------------------------------------
c=pd.read_csv(D/"tables/checkpoint_niche_features_all.tsv",sep="\t")
c=c[(c.analysis_set=="all")&(c.metric=="nearest_distance_mean_um")&(c.arm=="post_whole")]
c=c.dropna(subset=["global_marker_perm_p","block_marker_perm_p"])
fig,ax=plt.subplots(figsize=(2.3,2.1))
ax.scatter(c.global_marker_perm_p,c.block_marker_perm_p,s=8,
           c=[RC[RESP[p]] for p in c.patient],alpha=0.7,lw=0)
ax.plot([0,1],[0,1],color="k",lw=0.6,ls=":")
ax.axvline(0.05,color="#999999",lw=0.5); ax.axhline(0.05,color="#999999",lw=0.5)
ax.set_xlabel("global null p"); ax.set_ylabel("500 µm block null p")
save(fig,"S3_null_comparison",outdir="panels_v3/supp"); plt.close(fig)

# ---- S4 topology radius sensitivity ------------------------------------
T=pd.read_csv(D/"audit_v3/topology.tsv",sep="\t")
fig,axs=plt.subplots(1,2,figsize=(4.6,1.9),gridspec_kw={"wspace":0.42})
for k,ax_ in enumerate(["VISTA-PSGL1","TIGIT-NECTIN2 (tumour)"]):
    ax=axs[k]
    q=T[(T.axis==ax_)&(T.arm=="post_all")&(T.status=="ok")]
    for p in PTS:
        s=q[q.patient==p].sort_values("R")
        ax.plot(s.R,s.frac_in_mixed,marker="o",ms=2.4,lw=0.9,color=RC[RESP[p]],
                alpha=0.9,ls="-" if RESP[p] else (0,(3,1.6)))
    ax.set_xlabel("radius (µm)")
    ax.set_ylabel("fraction in mixed\ncomponent" if k==0 else "")
save(fig,"S4_topology_radius",outdir="panels_v3/supp"); plt.close(fig)

# ---- S5 label confidence ------------------------------------------------
lt=pd.read_csv(D/"tables/frozen_label_transfer.tsv",sep="\t")
fig,ax=plt.subplots(figsize=(2.6,1.9))
for i,arm in enumerate(["pre","post_whole","post_partial"]):
    s=lt[lt.arm==arm].set_index("patient").reindex(PTS)
    ax.bar(np.arange(6)+(i-1)*0.27,s.median_confidence,0.26,
           color=["#BDBDBD","#0072B2","#56B4E9"][i],label={"pre":"pre","post_whole":"whole","post_partial":"partial"}[arm])
ax.set_xticks(range(6)); ax.set_xticklabels(PTS,fontsize=6); ax.set_ylim(0,1.3)
ax.set_ylabel("median label confidence")
ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.03),handlelength=1.0)
save(fig,"S5_label_confidence",outdir="panels_v3/supp"); plt.close(fig)

# ---- S6 ROI candidate distribution -------------------------------------
roi=pd.read_csv(D/"audit_v2/roi_candidates_all.tsv",sep="\t")
fig,ax=plt.subplots(figsize=(2.6,1.9))
for p in PTS:
    s=roi[(roi.patient==p)&roi.candidate].sort_values("score",ascending=False)
    ax.plot(np.arange(1,len(s)+1),s.score.to_numpy(),lw=1.0,color=RC[RESP[p]],
            alpha=0.9,ls="-" if RESP[p] else (0,(3,1.6)))
ax.set_yscale("symlog",linthresh=1); ax.set_xlabel("ROI rank")
ax.set_ylabel("ROI selection score")
save(fig,"S6_roi_candidates",outdir="panels_v3/supp"); plt.close(fig)
print("[supp panels done]",flush=True)
