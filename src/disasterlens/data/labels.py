"""xBD-style label parsing and mask rasterization.

ASSUMPTION (verify on a real file in Stage 2): each label JSON looks like
  {"features": {"xy": [{"properties": {"feature_type": "building",
                                       "subtype": "minor-damage", "uid": "..."},
                        "wkt": "POLYGON ((x y, x y, ...))"}, ...],
                "lng_lat": [... same order, WKT in lon/lat ...]},
   "metadata": {...}}
Pre-disaster files have no "subtype". Unknown subtypes raise, they are never guessed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Mask encoding used throughout DisasterLens.
BACKGROUND = 0
DAMAGE_CLASSES = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}
CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
IGNORE = 255  # "un-classified" buildings: excluded from loss and metrics

_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_RING = re.compile(r"\(([^()]+)\)")


@dataclass(frozen=True)
class Building:
    uid: str
    damage: int                                   # 1..4, or IGNORE
    rings_xy: list[np.ndarray]                    # pixel coordinates, (N, 2); first ring = outer
    rings_lnglat: list[np.ndarray] | None = None  # lon/lat if present (approximate, see docs)


def parse_wkt_polygon(wkt: str) -> list[np.ndarray]:
    """Parse 'POLYGON ((...), (...))' into a list of (N, 2) float arrays."""
    text = wkt.strip()
    if not text.upper().startswith("POLYGON"):
        raise ValueError(f"Only POLYGON is supported, got: {text[:40]!r}")
    rings = []
    for body in _RING.findall(text):
        pts = [tuple(map(float, re.findall(_NUM, pair))) for pair in body.split(",")]
        if any(len(p) < 2 for p in pts):
            raise ValueError(f"Malformed coordinate pair in WKT: {body[:60]!r}")
        rings.append(np.array([p[:2] for p in pts], dtype=np.float64))
    if not rings:
        raise ValueError(f"No rings found in WKT: {text[:60]!r}")
    return rings


def load_buildings(label_path: str | Path, post_disaster: bool = True) -> list[Building]:
    """Read buildings from one label JSON. Raises FileNotFoundError / ValueError explicitly."""
    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        xy = data["features"]["xy"]
    except KeyError as exc:
        raise ValueError(f"{path.name}: missing features/xy (schema differs from assumption)") from exc
    lnglat = data["features"].get("lng_lat")
    if lnglat is not None and len(lnglat) != len(xy):
        lnglat = None  # do not pair mismatched lists; geographic view will fall back to image coords

    buildings = []
    for i, feat in enumerate(xy):
        props = feat.get("properties", {})
        if props.get("feature_type", "building") != "building":
            continue
        if post_disaster:
            subtype = props.get("subtype")
            if subtype == "un-classified":
                damage = IGNORE
            elif subtype in DAMAGE_CLASSES:
                damage = DAMAGE_CLASSES[subtype]
            else:
                raise ValueError(f"{path.name}: unknown damage subtype {subtype!r}")
        else:
            damage = DAMAGE_CLASSES["no-damage"]  # pre-disaster: footprint only
        buildings.append(Building(
            uid=str(props.get("uid", f"b{i}")),
            damage=damage,
            rings_xy=parse_wkt_polygon(feat["wkt"]),
            rings_lnglat=parse_wkt_polygon(lnglat[i]["wkt"]) if lnglat else None,
        ))
    return buildings


def rasterize_mask(buildings: list[Building], height: int, width: int) -> np.ndarray:
    """Damage mask (H, W) uint8: 0 background, 1..4 damage class, 255 ignore.

    Holes (inner rings) are cut out. Buildings are drawn in order; later ones win overlaps.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in buildings:
        outer = np.round(b.rings_xy[0]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [outer], int(b.damage))
        for hole in b.rings_xy[1:]:
            pts = np.round(hole).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], BACKGROUND)
    return mask
