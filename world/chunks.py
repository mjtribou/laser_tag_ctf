"""Chunk indexing utilities for voxel grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Iterator, Tuple

from world.voxel_grid import VoxelGrid


@dataclass(frozen=True)
class ChunkKey:
    x: int
    y: int
    z: int


class ChunkIndex:
    """Partition a voxel grid into axis-aligned chunks."""

    def __init__(self, grid: VoxelGrid, chunk_size: Tuple[int, int, int] = (16, 16, 16)) -> None:
        if len(chunk_size) != 3:
            raise ValueError("chunk_size must be a 3-tuple")
        if any(cs <= 0 for cs in chunk_size):
            raise ValueError("chunk dimensions must be positive")
        self.grid = grid
        self.chunk_size = tuple(int(v) for v in chunk_size)
        self._chunks = self._build_index()

    def _build_index(self) -> Tuple[int, int, int]:
        max_x = (self.grid.size_x + self.chunk_size[0] - 1) // self.chunk_size[0]
        max_y = (self.grid.size_y + self.chunk_size[1] - 1) // self.chunk_size[1]
        max_z = (self.grid.size_z + self.chunk_size[2] - 1) // self.chunk_size[2]
        return max_x, max_y, max_z

    def iter_chunk_keys(self) -> Iterator[ChunkKey]:
        max_x, max_y, max_z = self._chunks
        for cx in range(max_x):
            for cy in range(max_y):
                for cz in range(max_z):
                    yield ChunkKey(cx, cy, cz)

    def _chunk_bounds(self, key: ChunkKey) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        start_x = key.x * self.chunk_size[0]
        start_y = key.y * self.chunk_size[1]
        start_z = key.z * self.chunk_size[2]

        end_x = min(start_x + self.chunk_size[0], self.grid.size_x)
        end_y = min(start_y + self.chunk_size[1], self.grid.size_y)
        end_z = min(start_z + self.chunk_size[2], self.grid.size_z)
        return (start_x, start_y, start_z), (end_x, end_y, end_z)

    def iter_blocks_in_chunk(self, key: ChunkKey) -> Iterator[Tuple[int, int, int, int]]:
        (sx, sy, sz), (ex, ey, ez) = self._chunk_bounds(key)
        for x in range(sx, ex):
            for y in range(sy, ey):
                for z in range(sz, ez):
                    block = self.grid.get(x, y, z)
                    if block != 0:
                        yield x, y, z, block

    def world_bounds_in_chunk(self, key: ChunkKey) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return self._chunk_bounds(key)
