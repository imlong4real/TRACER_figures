#!/usr/bin/env python
"""r01 - extract the Cellular Localization channels and autofluorescence
background channels that the earlier protein prep omitted.

The 23-plex Xenium protein panel was already extracted (ch0008-ch0030 + ch0032
alphaSMA).  The four "Cellular Localization" markers requested are imaged in a
separate staining round:

    alphaSMA      ch0032   (already present)
    Vimentin      ch0031
    CD45          ch0033
    E-cadherin    ch0034

Four autofluorescence background channels (ch0004-ch0007, one per spectral
band) are also extracted.  They are not markers: they are the negative control
that lets us ask whether residual protein signal inside a reconstructed partial
cell is real staining or simply a locally bright / autofluorescent patch of
tissue.  No previous analysis used them.
"""
from __future__ import annotations
import json, time, zipfile
from pathlib import Path
import numpy as np, tifffile

BASE   = Path("/scratch4/adeshpa6/xenium_rcc")
ZIP    = BASE / "Xenium_V1_Human_Kidney_FFPE_Protein_updated_outs.zip"
TIFDIR = BASE / "rcc_analysis/protein/tif"
DSDIR  = BASE / "rcc_analysis/protein/downsampled"
MANP   = BASE / "rcc_analysis/protein/protein_manifest.json"
PIXEL_UM, LEVEL = 0.2125, 3

NEW = {
    "vimentin":   "ch0031_vimentin",
    "cd45":       "ch0033_cd45",
    "e-cadherin": "ch0034_e-cadherin",
    "af_blue":    "ch0004_blue_background",
    "af_green":   "ch0005_green_background",
    "af_yellow":  "ch0006_yellow_background",
    "af_red":     "ch0007_red_background",
}


def log(m): print(f"[r01] {m}", flush=True)


def read_plane(path: Path, level: int) -> np.ndarray:
    with tifffile.TiffFile(str(path), is_ome=False) as tf:
        s = tf.series[0]
        arr = np.squeeze(s.levels[min(level, len(s.levels) - 1)].asarray())
    if arr.ndim == 3:
        arr = arr[int(arr.reshape(arr.shape[0], -1).max(1).argmax())]
    return arr


def main() -> int:
    TIFDIR.mkdir(parents=True, exist_ok=True); DSDIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANP.read_text())
    t0 = time.time()
    z = zipfile.ZipFile(ZIP)
    members = {Path(n).name: n for n in z.namelist() if n.endswith(".ome.tif")}

    for marker, chbase in NEW.items():
        fname = f"{chbase}.ome.tif"
        dst = TIFDIR / fname
        if not dst.exists():
            member = members.get(fname)
            assert member, f"{fname} not in zip"
            log(f"extracting {member}")
            with z.open(member) as src, open(dst, "wb") as out:
                while (chunk := src.read(1 << 24)):
                    out.write(chunk)
        plane = read_plane(dst, LEVEL)
        np.save(DSDIR / f"{marker}_L{LEVEL}.npy", plane.astype(np.uint16))
        man["markers"][marker] = {
            "channel_file": fname, "shape": list(plane.shape),
            "um_per_px": PIXEL_UM * (2 ** LEVEL),
            "p99": float(np.percentile(plane, 99)),
            "max": int(plane.max()), "level": LEVEL,
            "role": "autofluorescence" if marker.startswith("af_")
                    else "cellular_localization"}
        log(f"{marker:12s} shape={plane.shape} p99={man['markers'][marker]['p99']:.0f} "
            f"max={man['markers'][marker]['max']} ({time.time()-t0:.0f}s)")

    MANP.write_text(json.dumps(man, indent=2))
    log(f"DONE {time.time()-t0:.0f}s -- manifest now has {len(man['markers'])} channels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
