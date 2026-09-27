"""Streaming reader for 10x `cell_segmentations.geojson`.

The file is a single-line FeatureCollection of ~10^5 polygons (50-100 MB of
text, ~2 GB once materialised as Python objects), which will not fit in the
shared 5 GB login cgroup. This walks the text one buffered chunk at a time,
uses each feature's FIRST vertex as a cheap locality proxy, and json-parses
only the few polygons whose bounding box can reach the requested window.
"""
from __future__ import annotations
import json, re
import numpy as np

DELIM = '{"type": "Feature"'
FIRST_XY = re.compile(r'\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)')


def _balanced(s: str) -> str | None:
    """Substring of s covering one balanced {...} object (coords hold no braces)."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[:i + 1]
    return None


def polys_in_window(path, x0, x1, y0, y1, pad=80.0, chunk=8 << 20):
    """{cell_id: (n,2) float array} for polygons near the window, full-res pixels.

    `pad` covers the offset between a polygon's first vertex and its extent.
    """
    out = {}
    buf = ""
    first = True
    with open(path) as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            buf += data
            parts = buf.split(DELIM)
            buf = parts.pop()
            if first:
                parts = parts[1:]
                first = False
            for p in parts:
                _take(DELIM + p, out, x0, x1, y0, y1, pad)
    tail = buf.split(DELIM)
    if first:
        tail = tail[1:]
    for p in tail:
        _take(DELIM + p, out, x0, x1, y0, y1, pad)
    return out


def _take(s, out, x0, x1, y0, y1, pad):
    m = FIRST_XY.search(s)
    if m is None:
        return
    fx, fy = float(m.group(1)), float(m.group(2))
    if fx < x0 - pad or fx > x1 + pad or fy < y0 - pad or fy > y1 + pad:
        return
    body = _balanced(s)
    if body is None:
        return
    try:
        ft = json.loads(body)
    except Exception:
        return
    c = np.asarray(ft["geometry"]["coordinates"][0], float)
    out[int(ft["properties"]["cell_id"])] = c
