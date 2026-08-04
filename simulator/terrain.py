"""Read-only terrain queries over the exported Dota heightmap.

The heightmap is an exported, normalized raster rather than the Dota navigation
mesh.  It is useful for ground height, terrain slope, and keeping simulated
actors inside the map bounds; it cannot establish exact Dota walkability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TerrainHeightmap:
    """Bilinear world-coordinate sampling of a normalized terrain raster."""

    normalized_heights: np.ndarray
    map_min: float
    map_max: float
    raw_height_min: float
    raw_height_max: float

    @classmethod
    def from_data_directory(cls, directory: Path) -> "TerrainHeightmap":
        """Load ``dota_heightmap.npy`` and its required provenance metadata."""
        heights_path = directory / "dota_heightmap.npy"
        metadata_path = directory / "dota_heightmap.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.asarray(np.load(heights_path), dtype=np.float32)
        if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
            raise ValueError("heightmap must be a finite two-dimensional grid")
        return cls(
            normalized_heights=values,
            map_min=float(metadata["map_min"]),
            map_max=float(metadata["map_max"]),
            raw_height_min=float(metadata["raw_height_min"]),
            raw_height_max=float(metadata["raw_height_max"]),
        )

    @property
    def cell_size(self) -> float:
        return (self.map_max - self.map_min) / (self.normalized_heights.shape[0] - 1)

    @property
    def raw_height_span(self) -> float:
        return self.raw_height_max - self.raw_height_min

    def contains(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float32)
        return ((points[..., 0] >= self.map_min) & (points[..., 0] <= self.map_max)
                & (points[..., 1] >= self.map_min) & (points[..., 1] <= self.map_max))

    def height(self, xy: np.ndarray) -> np.ndarray:
        """Return bilinearly interpolated *raw* world height for world ``x, y``."""
        points = np.asarray(xy, dtype=np.float32)
        if points.shape[-1] != 2:
            raise ValueError("terrain coordinates must end in (x, y)")
        clipped = np.clip(points, self.map_min, self.map_max)
        height, width = self.normalized_heights.shape
        u = (clipped[..., 0] - self.map_min) / (self.map_max - self.map_min) * (width - 1)
        v = (clipped[..., 1] - self.map_min) / (self.map_max - self.map_min) * (height - 1)
        x0, y0 = np.floor(u).astype(np.intp), np.floor(v).astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        tx, ty = u - x0, v - y0
        top = self.normalized_heights[y0, x0] * (1.0 - tx) + self.normalized_heights[y0, x1] * tx
        bottom = self.normalized_heights[y1, x0] * (1.0 - tx) + self.normalized_heights[y1, x1] * tx
        normalized = top * (1.0 - ty) + bottom * ty
        return (self.raw_height_min + normalized * self.raw_height_span).astype(np.float32)

    def slope(self, xy: np.ndarray) -> np.ndarray:
        """Return local rise/run estimated with centered terrain differences."""
        points = np.asarray(xy, dtype=np.float32)
        delta = self.cell_size
        x_delta = np.zeros_like(points); x_delta[..., 0] = delta
        y_delta = np.zeros_like(points); y_delta[..., 1] = delta
        dzdx = (self.height(points + x_delta) - self.height(points - x_delta)) / (2.0 * delta)
        dzdy = (self.height(points + y_delta) - self.height(points - y_delta)) / (2.0 * delta)
        return np.hypot(dzdx, dzdy).astype(np.float32)

    def traversable(self, start_xy: np.ndarray, end_xy: np.ndarray, max_slope: float) -> bool:
        """Conservative ground-only movement check along one proposed segment."""
        start = np.asarray(start_xy, dtype=np.float32)
        end = np.asarray(end_xy, dtype=np.float32)
        if not bool(self.contains(start)) or not bool(self.contains(end)):
            return False
        distance = float(np.linalg.norm(end - start))
        samples = max(2, int(np.ceil(distance / max(self.cell_size / 2.0, 1.0))) + 1)
        points = np.linspace(start, end, samples, dtype=np.float32)
        heights = self.height(points)
        horizontal = max(distance / (samples - 1), 1e-6)
        segment_slope = np.abs(np.diff(heights)) / horizontal
        return bool(np.all(segment_slope <= max_slope) and np.all(self.slope(points) <= max_slope))

