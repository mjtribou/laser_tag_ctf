"""Helpers for adapting authored maps into voxel grids."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from game.map_gen import load_from_file
from world.voxel_grid import VoxelGrid


@dataclass(frozen=True)
class BlockDef:
    material: str
    atlas_tile: str
    opaque: bool = True


BlockRegistry = Dict[int, BlockDef]


def _pos_to_index(pos: float, cube_size: float) -> int:
    # Shift from cube center to voxel origin before resolving to an index.
    scaled = (pos - 0.5 * cube_size) / cube_size
    # Guard against float drift around integers.
    return int(math.floor(scaled + 1e-6))


def _compute_bounds(indices: Iterable[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    min_x = min_y = min_z = None
    max_x = max_y = max_z = None
    for x, y, z in indices:
        min_x = x if min_x is None else min(min_x, x)
        min_y = y if min_y is None else min(min_y, y)
        min_z = z if min_z is None else min(min_z, z)
        max_x = x if max_x is None else max(max_x, x)
        max_y = y if max_y is None else max(max_y, y)
        max_z = z if max_z is None else max(max_z, z)
    if min_x is None:
        raise ValueError("Map contained no solid blocks")
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def load_map_to_voxels(json_path: str) -> Tuple[VoxelGrid, BlockRegistry]:
    """Load a map JSON file into a dense voxel grid."""
    mapdata = load_from_file(json_path)
    cube = float(getattr(mapdata, "cube_size", 1.0) or 1.0)
    if cube <= 0.0:
        raise ValueError("cube_size must be positive")

    # Convert block centers into integer voxel indices.
    indices = [
        (
            _pos_to_index(block.pos[0], cube),
            _pos_to_index(block.pos[1], cube),
            _pos_to_index(block.pos[2], cube),
        )
        for block in mapdata.blocks
    ]
    if not indices:
        raise ValueError("Map contained no voxels to populate")

    min_bounds, max_bounds = _compute_bounds(indices)

    size_x = max_bounds[0] - min_bounds[0] + 1
    size_y = max_bounds[1] - min_bounds[1] + 1
    size_z = max_bounds[2] - min_bounds[2] + 1

    grid = VoxelGrid((size_x, size_y, size_z))
    block_id = 1

    for (ix, iy, iz) in indices:
        grid.set(ix - min_bounds[0], iy - min_bounds[1], iz - min_bounds[2], block_id)

    registry: BlockRegistry = {
        block_id: BlockDef(material="stone", atlas_tile="stone", opaque=True)
    }

    # Sanity: ensure counts align with authored cube total.
    solid_count = len(indices)
    filled = 0
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                if grid.get(x, y, z) != 0:
                    filled += 1
    if filled != solid_count:
        raise AssertionError("Voxel population mismatch vs block count")

    return grid, registry
