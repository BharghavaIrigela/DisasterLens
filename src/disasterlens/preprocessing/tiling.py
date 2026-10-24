"""Tiling with explicit source-image coordinates (so tile results map back exactly)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x: int  # left edge in the source image (pixels)
    y: int  # top edge
    size: int

    def to_image_xy(self, tx: float, ty: float) -> tuple[float, float]:
        """Convert a point in tile pixels to source-image pixels."""
        return self.x + tx, self.y + ty


def _starts(length: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:  # last tile is shifted back so the edge is covered
        starts.append(length - size)
    return starts


def tile_grid(height: int, width: int, size: int = 512, overlap: int = 64) -> list[Tile]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    if size > min(height, width):
        raise ValueError(f"tile size {size} larger than image {width}x{height}")
    stride = size - overlap
    return [Tile(x, y, size) for y in _starts(height, size, stride) for x in _starts(width, size, stride)]


def crop(image: np.ndarray, tile: Tile) -> np.ndarray:
    return image[tile.y:tile.y + tile.size, tile.x:tile.x + tile.size]
