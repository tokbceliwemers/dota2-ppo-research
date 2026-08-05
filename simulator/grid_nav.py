"""Read the shipped Dota GridNav (`.gnv`) walkability grid.

This is the original static grid-navigation asset, not a triangle mesh.  The
format is intentionally parsed only as far as its self-describing header and
one byte per cell allow.  The low bit is the traversability bit: it is present
throughout the playable interior and absent in the outer blocked region.  The
remaining flag bits are retained raw and are not assigned game-specific names.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MAGIC = 0xFADEBEAD
_HEADER = struct.Struct("<Ifffiiii")
_WALKABLE_BIT = np.uint8(0x01)


@dataclass(frozen=True)
class GridNavigation:
    """One-byte-per-cell Source GridNav data in Dota world coordinates."""

    cell_size: float
    cell_center_x: float
    cell_center_y: float
    width: int
    height: int
    origin_cell_x: int
    origin_cell_y: int
    flags: np.ndarray  # [row_y, column_x], uint8

    @classmethod
    def from_file(cls, path: Path) -> "GridNavigation":
        payload = path.read_bytes()
        if len(payload) < _HEADER.size:
            raise ValueError("GridNav file is smaller than its header")
        magic, cell_size, center_x, center_y, width, height, origin_x, origin_y = _HEADER.unpack_from(payload)
        if magic != _MAGIC:
            raise ValueError(f"unsupported GridNav magic {magic:#x}")
        if cell_size <= 0 or width <= 0 or height <= 0:
            raise ValueError("GridNav header has non-positive dimensions")
        expected = _HEADER.size + width * height
        if len(payload) != expected:
            raise ValueError(f"GridNav size mismatch: expected {expected}, got {len(payload)}")
        flags = np.frombuffer(payload, dtype=np.uint8, offset=_HEADER.size).copy().reshape(height, width)
        return cls(cell_size, center_x, center_y, width, height, origin_x, origin_y, flags)

    @property
    def world_min(self) -> np.ndarray:
        return np.array((self.origin_cell_x * self.cell_size, self.origin_cell_y * self.cell_size), dtype=np.float32)

    @property
    def world_max(self) -> np.ndarray:
        return self.world_min + np.array((self.width * self.cell_size, self.height * self.cell_size), dtype=np.float32)

    def _indices(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=np.float32)
        if points.shape[-1] != 2:
            raise ValueError("navigation coordinates must end in (x, y)")
        minimum = self.world_min
        x = np.floor((points[..., 0] - minimum[0]) / self.cell_size).astype(np.intp)
        y = np.floor((points[..., 1] - minimum[1]) / self.cell_size).astype(np.intp)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        return x, y, valid

    def raw_flags(self, xy: np.ndarray) -> np.ndarray:
        x, y, valid = self._indices(xy)
        result = np.zeros_like(x, dtype=np.uint8)
        result[valid] = self.flags[y[valid], x[valid]]
        return result

    def is_walkable(self, xy: np.ndarray) -> np.ndarray:
        """Return the conservative low-bit traversability result for each point."""
        return (self.raw_flags(xy) & _WALKABLE_BIT) != 0

    def traversable(self, start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
        """Require every crossed GridNav cell to have the walkable bit set."""
        start = np.asarray(start_xy, dtype=np.float32)
        end = np.asarray(end_xy, dtype=np.float32)
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(distance / max(self.cell_size / 2.0, 1.0))) + 1)
        points = np.linspace(start, end, count, dtype=np.float32)
        return bool(np.all(self.is_walkable(points)))

