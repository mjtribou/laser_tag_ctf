from __future__ import annotations
import math
from typing import List, Tuple, Dict, Set

# Reuse Block and MapData for convenience
try:
    from .map_gen import Block, MapData
except Exception:  # minimal fallback for type checking
    from dataclasses import dataclass
    @dataclass
    class Block:  # type: ignore
        pos: Tuple[float, float, float]
        size: Tuple[float, float, float]
        box_type: int = 0
    @dataclass
    class MapData:  # type: ignore
        blocks: List[Block]
        bounds: Tuple[float, float]
        cube_size: float


def _grid_index_from_world(x: float, y: float, z: float, origin_x: float, origin_y: float, cube: float) -> Tuple[int, int, int]:
    """Map world-space cube center to integer voxel indices (ix, iy, k).

    Assumes cubes are axis-aligned and centered at (ix+0.5, iy+0.5, k+0.5) * cube,
    with world floor at z=0 and XY origin at (-size/2, -size/2).
    """
    ix = int(math.floor((x - origin_x) / cube))
    iy = int(math.floor((y - origin_y) / cube))
    k = int(math.floor(z / cube))
    return ix, iy, k


def merge_voxel_cubes_2d_per_level(mapdata: MapData) -> List[Block]:
    """
    Merge adjacent 1x1x1 voxel cubes into larger AABBs, per Z-level (thickness = 1 cube).

    Strategy:
      1) Build occupancy sets per k (vertical level).
      2) For each level, greedily tile the XY grid using maximal rectangles
         of occupied cells that are not yet covered.

    Returns a new list of Block entries representing merged AABBs for physics.
    Visuals should keep using the original mapdata.blocks.
    """
    size_x, size_y = mapdata.bounds
    cube = float(getattr(mapdata, "cube_size", 1.0) or 1.0)
    origin_x = -0.5 * size_x
    origin_y = -0.5 * size_y
    nx = max(1, int(round(size_x / cube)))
    ny = max(1, int(round(size_y / cube)))

    # Build per-level occupancy: k -> set of (ix, iy)
    levels: Dict[int, Set[Tuple[int, int]]] = {}
    for b in getattr(mapdata, "blocks", []) or []:
        sx, sy, sz = b.size
        # only process 1x1x1 voxels to avoid accidentally absorbing walls/floor (which are added separately)
        if not (abs(sx - cube) < 1e-5 and abs(sy - cube) < 1e-5 and abs(sz - cube) < 1e-5):
            continue
        ix, iy, k = _grid_index_from_world(b.pos[0], b.pos[1], b.pos[2], origin_x, origin_y, cube)
        if 0 <= ix < nx and 0 <= iy < ny and k >= 0:
            levels.setdefault(k, set()).add((ix, iy))

    merged: List[Block] = []

    # Greedy rectangle tiling per Z level
    for k, occ_set in levels.items():
        if not occ_set:
            continue

        # Build occupancy and visited arrays for this level
        occ = [[False for _ in range(ny)] for _ in range(nx)]
        used = [[False for _ in range(ny)] for _ in range(nx)]
        for (ix, iy) in occ_set:
            if 0 <= ix < nx and 0 <= iy < ny:
                occ[ix][iy] = True

        for iy in range(ny):
            ix = 0
            while ix < nx:
                # Skip already covered or empty
                if (not occ[ix][iy]) or used[ix][iy]:
                    ix += 1
                    continue

                # Compute maximal horizontal run starting here (row-wise)
                run_w = 1
                while (ix + run_w) < nx and occ[ix + run_w][iy] and not used[ix + run_w][iy]:
                    run_w += 1

                # Compute maximal vertical run starting here (col-wise)
                run_h = 1
                while (iy + run_h) < ny and occ[ix][iy + run_h] and not used[ix][iy + run_h]:
                    run_h += 1

                # Candidate A: width-first rectangle, then extend down constrained by width
                wA = run_w
                hA = 1
                while (iy + hA) < ny:
                    ok = True
                    for tx in range(ix, ix + wA):
                        if (not occ[tx][iy + hA]) or used[tx][iy + hA]:
                            ok = False
                            break
                    if not ok:
                        break
                    hA += 1

                # Candidate B: height-first rectangle, then extend right constrained by height
                hB = run_h
                wB = 1
                while (ix + wB) < nx:
                    ok = True
                    for ty in range(iy, iy + hB):
                        if (not occ[ix + wB][ty]) or used[ix + wB][ty]:
                            ok = False
                            break
                    if not ok:
                        break
                    wB += 1

                areaA = wA * hA
                areaB = wB * hB

                # Pick better candidate; if areas tie, prefer the "taller" one to reduce
                # seams along long outer walls (helps L-shapes).
                if areaB > areaA or (areaA == areaB and hB >= wB and not (hA >= wA and hB == wB == 1)):
                    w, h = wB, hB
                else:
                    w, h = wA, hA

                # Mark used
                for ty in range(iy, iy + h):
                    for tx in range(ix, ix + w):
                        used[tx][ty] = True

                # Emit merged slab for this rectangle at level k (thickness 1 cube)
                cx = origin_x + (ix + 0.5 * w) * cube
                cy = origin_y + (iy + 0.5 * h) * cube
                cz = (k + 0.5) * cube
                sx = w * cube
                sy = h * cube
                sz = cube
                merged.append(Block(pos=(cx, cy, cz), size=(sx, sy, sz), box_type=0))

                # Continue after this rectangle
                ix += w

    return merged


def merge_voxel_cubes_full(mapdata: MapData) -> List[Block]:
    """
    Extended merge that first merges per-level rectangles and then optionally
    stacks identical XY rectangles across consecutive k-levels into taller boxes.

    This preserves exact occupancy (no filling gaps) while further reducing seams.
    """
    # First, get per-level rectangles
    slabs = merge_voxel_cubes_2d_per_level(mapdata)
    if not slabs:
        return slabs

    size_x, size_y = mapdata.bounds
    cube = float(getattr(mapdata, "cube_size", 1.0) or 1.0)
    origin_x = -0.5 * size_x
    origin_y = -0.5 * size_y

    # Index by (x0,x1,y0,y1,k) using grid coordinates to recognize identical XY footprints
    def rect_to_key(b: Block) -> Tuple[int, int, int, int, int]:
        # Recover grid-aligned integer bounds from world center+size
        cx, cy, cz = b.pos
        sx, sy, sz = b.size
        # Convert to min/max ix/iy (inclusive-exclusive) and k
        # Because these rectangles are aligned to the grid and integral multiples of cube,
        # floating point rounding should be stable with small eps.
        eps = 1e-6
        x0 = int(round(((cx - 0.5 * sx) - origin_x + eps) / cube))
        x1 = int(round(((cx + 0.5 * sx) - origin_x - eps) / cube))
        y0 = int(round(((cy - 0.5 * sy) - origin_y + eps) / cube))
        y1 = int(round(((cy + 0.5 * sy) - origin_y - eps) / cube))
        k  = int(math.floor(cz / cube))
        return x0, x1, y0, y1, k

    # Build lookup from footprint -> set of ks present
    by_rect: Dict[Tuple[int, int, int, int], Set[int]] = {}
    rect_from_key: Dict[Tuple[int, int, int, int, int], Block] = {}
    for b in slabs:
        x0, x1, y0, y1, k = rect_to_key(b)
        by_rect.setdefault((x0, x1, y0, y1), set()).add(k)
        rect_from_key[(x0, x1, y0, y1, k)] = b

    merged: List[Block] = []

    # For each unique XY footprint, combine consecutive k runs
    for xy_key, ks in by_rect.items():
        x0, x1, y0, y1 = xy_key
        sorted_ks = sorted(ks)
        if not sorted_ks:
            continue
        run_start = sorted_ks[0]
        prev = run_start
        for kk in sorted_ks[1:] + [None]:
            if kk is not None and kk == prev + 1:
                prev = kk
                continue
            # Close current run [run_start .. prev]
            height_cubes = (prev - run_start + 1)
            # Build world AABB
            cx = origin_x + (x0 + x1) * 0.5 * cube
            cy = origin_y + (y0 + y1) * 0.5 * cube
            cz = (run_start + 0.5 * height_cubes) * cube
            sx = (x1 - x0) * cube
            sy = (y1 - y0) * cube
            sz = height_cubes * cube
            merged.append(Block(pos=(cx, cy, cz), size=(sx, sy, sz), box_type=0))
            # Start new run
            if kk is None:
                break
            run_start = kk
            prev = kk

    return merged


def build_merged_colliders(mapdata: MapData, strategy: str = "stack") -> List[Block]:
    """Public API: return merged AABBs suitable for Bullet static world.

    strategy:
      - "level" → merge per Z level only (thickness = 1 cube slabs)
      - "stack" → also merge identical XY slabs across consecutive levels
    """
    if strategy == "level":
        return merge_voxel_cubes_2d_per_level(mapdata)
    return merge_voxel_cubes_full(mapdata)
