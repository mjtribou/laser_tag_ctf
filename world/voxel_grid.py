"""Voxel grid storage with fast indexed access."""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

Face = Tuple[int, int, int]

FACE_DIRECTIONS: Tuple[Face, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def neighbors(x: int, y: int, z: int) -> Iterator[Face]:
    """Yield face-adjacent neighbor coordinates for ``(x, y, z)``."""
    for dx, dy, dz in FACE_DIRECTIONS:
        yield x + dx, y + dy, z + dz


class VoxelGrid:
    """Dense voxel storage optimized for tight loops."""

    __slots__ = ("size_x", "size_y", "size_z", "_default", "_data")

    def __init__(self, size_xyz: Sequence[int], default_block: int = 0) -> None:
        if len(size_xyz) != 3:
            raise ValueError("size_xyz must contain three integers")
        sx, sy, sz = (int(axis) for axis in size_xyz)
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError("grid dimensions must be positive")

        self.size_x = sx
        self.size_y = sy
        self.size_z = sz
        self._default = int(default_block)

        volume = sx * sy * sz
        self._data = [self._default] * volume

    # Internal utilities -------------------------------------------------
    def _index(self, x: int, y: int, z: int) -> int:
        if not (0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z):
            raise IndexError("voxel coordinates out of range")
        return (z * self.size_y + y) * self.size_x + x

    # API ----------------------------------------------------------------
    def get(self, x: int, y: int, z: int) -> int:
        return self._data[self._index(x, y, z)]

    def set(self, x: int, y: int, z: int, block_id: int) -> None:
        self._data[self._index(x, y, z)] = int(block_id)

    def is_air(self, x: int, y: int, z: int) -> bool:
        return self.get(x, y, z) == 0

    def bounds(self) -> Tuple[Face, Face]:
        return (0, 0, 0), (self.size_x - 1, self.size_y - 1, self.size_z - 1)
