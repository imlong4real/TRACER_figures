"""Shared style for every PDAC panel. Arimo (metric Arial), Okabe-Ito palette."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
from matplotlib import rcParams
for _f in Path("/home/lyuan13/scr4_adeshpa6/tracer_campaign/fonts").glob("Arimo-*.ttf"):
    _fm.fontManager.addfont(str(_f))
rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arimo","Arial","DejaVu Sans"],
 "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.size":7,
 "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
 "xtick.labelsize":6.5,"ytick.labelsize":6.5,"axes.labelsize":7.5,"legend.fontsize":6.5,
 "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200})

PTS=["HC01","HC03","HC04","HC05","HC07","HC08"]
RESP={"HC05":1,"HC07":1,"HC08":1,"HC01":0,"HC03":0,"HC04":0}
ARMS=["pre","post_whole","post_all"]
ARMLAB={"pre":"pre","post_whole":"whole","post_all":"whole+partial"}
# Okabe-Ito
CT={"T cell":"#0072B2","Macrophage cell":"#D55E00","Ductal cell type 1":"#E69F00",
    "Ductal cell type 2":"#E69F00","Fibroblast cell":"#BDBDBD","Stellate cell":"#CFCFCF",
    "B cell":"#56B4E9","Endothelial cell":"#8C8C8C","Acinar cell":"#F0E442",
    "Endocrine cell":"#AAAAAA"}
TUMOR=["Ductal cell type 1","Ductal cell type 2"]
NECTIN2_C="#009E73"     # ligand-positive tumour (filled; high contrast on pink H&E)
TIGIT_C="#000000"       # receptor-positive T
EDGE_C="#CC79A7"
RC={1:"#009E73",0:"#D55E00"}
ARMC={"pre":"#BDBDBD","post_whole":"#0072B2","post_all":"#56B4E9"}
CATS=["visible pre","strengthened post","enabled post-whole","enabled by partials","unsupported"]
CATC=["#F0E442","#009E73","#0072B2","#56B4E9","#EEEEEE"]

def save(fig,name,outdir="panels_v3"):
    from pathlib import Path
    d=Path("/scratch4/adeshpa6/PDAC_Long/analysis/PDAC6_TRACER_discovery_v1")/outdir
    d.mkdir(parents=True,exist_ok=True)
    for ext,dpi in (("pdf",600),("png",400),("svg",600)):
        fig.savefig(d/f"{name}.{ext}",dpi=dpi,bbox_inches="tight")
    print(f"  wrote {name}.{{pdf,png,svg}}",flush=True)
