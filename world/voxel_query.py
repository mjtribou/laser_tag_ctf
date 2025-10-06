"""Utility helpers for voxel-based occupancy queries."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Set, Tuple


GridIndex = Tuple[int, int, int]


class VoxelQuery:
    """Spatial queries against a set of solid voxel indices."""

    __slots__ = (
        "_solid",
        "_cube",
        "_inv_cube",
        "_bounds",
    )

    def __init__(self, solid_voxels: Set[GridIndex], cube_size: float) -> None:
        self._solid = solid_voxels
        self._cube = float(cube_size) if cube_size > 0 else 1.0
        self._inv_cube = 1.0 / self._cube

        if solid_voxels:
            xs = [ix for ix, _, _ in solid_voxels]
            ys = [iy for _, iy, _ in solid_voxels]
            zs = [iz for _, _, iz in solid_voxels]
            self._bounds = (
                (min(xs), max(xs)),
                (min(ys), max(ys)),
                (min(zs), max(zs)),
            )
        else:
            self._bounds = ((0, -1), (0, -1), (0, -1))

    # ------------------------------------------------------------------
    # Basic conversions
    def _cell_index(self, x: float, y: float, z: float) -> GridIndex:
        idx_x = math.floor(x * self._inv_cube)
        idx_y = math.floor(y * self._inv_cube)
        idx_z = math.floor(z * self._inv_cube)
        return (idx_x, idx_y, idx_z)

    def _cell_bounds(self, cell: GridIndex) -> Tuple[float, float, float, float, float, float]:
        ix, iy, iz = cell
        cube = self._cube
        return (
            ix * cube,
            (ix + 1) * cube,
            iy * cube,
            (iy + 1) * cube,
            iz * cube,
            (iz + 1) * cube,
        )

    def _within_bounds(self, cell: GridIndex) -> bool:
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = self._bounds
        ix, iy, iz = cell
        return (min_x <= ix <= max_x) and (min_y <= iy <= max_y) and (min_z <= iz <= max_z)

    # ------------------------------------------------------------------
    # Queries
    def point_inside(self, x: float, y: float, z: float, margin: float = 0.0) -> bool:
        if not self._solid:
            return False
        cell = self._cell_index(x, y, z)
        if cell not in self._solid:
            return False
        if margin <= 0.0:
            return True

        min_x, max_x, min_y, max_y, min_z, max_z = self._cell_bounds(cell)
        return (
            (min_x + margin) <= x <= (max_x - margin)
            and (min_y + margin) <= y <= (max_y - margin)
            and (min_z + margin) <= z <= (max_z - margin)
        )

    def ray_hits_solid(
        self,
        start: Sequence[float],
        end: Sequence[float],
        *,
        ignore_start: bool = True,
        ignore_end: bool = True,
        step_fraction: float = 0.125,
    ) -> bool:
        if not self._solid:
            return False

        sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
        ex, ey, ez = float(end[0]), float(end[1]), float(end[2])

        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return False

        start_cell = self._cell_index(sx, sy, sz)
        end_cell = self._cell_index(ex, ey, ez)

        step_fraction = max(1e-2, float(step_fraction))
        step = self._cube * step_fraction
        steps = max(1, int(math.ceil(length / step)))

        inv_steps = 1.0 / steps
        for i in range(1, steps + 1):
            if ignore_end and i == steps:
                break
            t = i * inv_steps
            px = sx + dx * t
            py = sy + dy * t
            pz = sz + dz * t
            cell = self._cell_index(px, py, pz)
            if ignore_start and cell == start_cell:
                continue
            if ignore_end and cell == end_cell:
                continue
            if not self._within_bounds(cell):
                continue
            if cell in self._solid:
                return True
        return False


def build_voxel_query(solid_voxels: Set[GridIndex], cube_size: float) -> Optional[VoxelQuery]:
    if not solid_voxels:
        return None
    return VoxelQuery(solid_voxels, cube_size)
