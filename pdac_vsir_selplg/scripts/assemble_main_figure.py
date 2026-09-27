#!/usr/bin/env python3
"""Assemble the frozen PDAC panel set in the a-p order in FIGURE_ORDER.md.

This script performs layout only. It does not recompute or alter a panel.
Multiple current panel variants belonging to one manuscript panel are grouped
within the same labelled tile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


GROUPS = [
    ("a", ["M1_xenium_transcript_fate_alluvial.png"], "vertical"),
    ("b", ["M2_xenium_transcript_fate_accounting.png"], "vertical"),
    ("c", ["M3_xenium_3d_roi.png"], "vertical"),
    ("d", ["M4_xenium_roi_immune_resolved.png"], "vertical"),
    ("e", ["M5_xenium_admixture_cleanup.png"], "vertical"),
    ("f", ["M6_xenium_vsig4_specificity.png"], "vertical"),
    ("g", ["M7_xenium_immune_recovery.png"], "vertical"),
    ("h", ["M15_hd_roi_HC05_responder.png", "M15b_hd_roi_HC01_nonresponder.png",
           "M15c_legend_celltype.png"], "vertical"),
    ("i", ["M8_hd_axes_detectable.png", "M0_legend_response.png"], "vertical"),
    ("j", ["M9_hd_vista_enrichment_forest.png"], "vertical"),
    ("k", ["M11_hd_lineage_identity.png", "M11b_hd_reference_similarity.png"], "vertical"),
    ("l", ["M10_hd_vista_partial_contribution.png"], "vertical"),
    ("m", ["M12_hd_vista_roi_HC01_nonresponder.png", "M12_hd_vista_roi_HC08_responder.png",
           "M12b_legend_vista.png"], "vertical"),
    ("n", ["M13_hd_vista_wholetissue_HC08_responder.png",
           "M13_hd_vista_wholetissue_HC01_nonresponder.png"], "horizontal"),
    ("o", ["M14_hd_domain_structure.png"], "vertical"),
    ("p", ["M16_hd_wholetissue_prepost_HC05_responder.png",
           "M16_hd_wholetissue_prepost_HC01_nonresponder.png",
           "M16b_legend_domains.png"], "vertical"),
]


def fit(im: Image.Image, box: tuple[int, int]) -> Image.Image:
    out = ImageOps.contain(im.convert("RGB"), box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", box, "white")
    canvas.paste(out, ((box[0] - out.width) // 2, (box[1] - out.height) // 2))
    return canvas


def grouped(images: list[Image.Image], mode: str, box: tuple[int, int]) -> Image.Image:
    gap = 18
    if len(images) == 1:
        return fit(images[0], box)
    if mode == "horizontal":
        each = ((box[0] - gap * (len(images) - 1)) // len(images), box[1])
        fitted = [fit(im, each) for im in images]
        out = Image.new("RGB", box, "white")
        x = 0
        for im in fitted:
            out.paste(im, (x, 0)); x += each[0] + gap
        return out
    each = (box[0], (box[1] - gap * (len(images) - 1)) // len(images))
    fitted = [fit(im, each) for im in images]
    out = Image.new("RGB", box, "white")
    y = 0
    for im in fitted:
        out.paste(im, (0, y)); y += each[1] + gap
    return out


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    campaign = repository.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=Path,
                        default=campaign / "final/panels/pdac_main")
    parser.add_argument("--output", type=Path,
                        default=campaign / "final/panels/Fig_PDAC_VSIR_SELPLG.png")
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    missing = [name for _, names, _ in GROUPS for name in names
               if not (args.panels / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing panel files: " + ", ".join(missing))

    tile_w, tile_h, outer, gap, label_h = 2200, 1600, 70, 45, 105
    rows = (len(GROUPS) + args.columns - 1) // args.columns
    canvas = Image.new("RGB", (outer * 2 + args.columns * tile_w + (args.columns - 1) * gap,
                               outer * 2 + rows * tile_h + (rows - 1) * gap), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 72) if font_path.exists() else ImageFont.load_default()

    for index, (letter, names, mode) in enumerate(GROUPS):
        row, col = divmod(index, args.columns)
        x = outer + col * (tile_w + gap)
        y = outer + row * (tile_h + gap)
        ims = [Image.open(args.panels / name) for name in names]
        panel = grouped(ims, mode, (tile_w, tile_h - label_h))
        canvas.paste(panel, (x, y + label_h))
        draw.text((x + 8, y), letter, fill="#111111", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(450, 450), optimize=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
