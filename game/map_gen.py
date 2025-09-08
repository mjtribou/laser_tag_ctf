# game/map_gen.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
import math

@dataclass
class Block:
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]

@dataclass
class MapData:
    blocks: List[Block]
    red_base: Tuple[float, float, float]
    blue_base: Tuple[float, float, float]
    red_flag_stand: Tuple[float, float, float]
    blue_flag_stand: Tuple[float, float, float]
    bounds: Tuple[float, float]  # (x_size, z_size) — Panda uses X right, Y forward, Z up
    cube_size: float = 1.0       # new: grid cell size (meters)
    agent_radius: float = 0.5    # new: used by navgrid if present

def _mirror_ix(ix: int, nx: int, symmetry: str) -> Optional[int]:
    """
    Mirror an x-index into the opposite half.
    - even: center seam is between columns (nx/2 -1) and (nx/2)
    - odd : the center column mirrors onto itself
    Returns None if we should *not* duplicate (center on odd).
    """
    mid = nx // 2
    j = (nx - 1) - ix
    if symmetry == "odd" and ix == mid:
        return None
    return j

def generate(seed: int,
             size_x: float,
             size_z: float,
             cubes: Optional[Dict] = None) -> MapData:
    """
    Minecraft-style voxel map:
    - Axis-aligned cubes (1×1×1 by default), stacked up to max_height.
    - Symmetry across X=0 with 'even' or 'odd' parity.
    - Spawn areas cleared around each team base.
    """
    cfg = {
        "size": 1.0,               # meters per cube
        "max_height": 3,           # max vertical cubes in a stack
        "fill_prob": 0.09,         # base chance per ground cell on authored half
        "stack_prob": 0.55,        # probability to add the next cube in the stack
        "symmetry": "even",        # 'even' or 'odd'
        "spawn_clear_radius": 6,   # cubes; cleared around each base center
    }
    if isinstance(cubes, dict):
        cfg.update(cubes)

    cube = float(cfg["size"])
    max_h = int(cfg["max_height"])
    p0 = float(cfg["fill_prob"])
    p_stack = float(cfg["stack_prob"])
    symmetry = str(cfg["symmetry"]).lower()
    spawn_clear_r = int(round(cfg["spawn_clear_radius"]))

    rng = random.Random(seed)

    # Discrete grid covering the arena in XY (Panda: Y is "forward").
    nx = max(1, int(math.floor(size_x / cube)))
    ny = max(1, int(math.floor(size_z / cube)))

    origin_x = -size_x * 0.5
    origin_y = -size_z * 0.5

    def world_to_ix(x: float) -> int:
        return int(math.floor((x - origin_x) / cube))

    def world_to_iy(y: float) -> int:
        return int(math.floor((y - origin_y) / cube))

    # Author only the "left" half (smaller ix side), then mirror.
    # We'll consider "left" to be ix in [0 .. mid) for even; [0 .. mid] for odd.
    authored_half: set[Tuple[int, int, int]] = set()

    mid = nx // 2
    if symmetry == "even":
        left_range = range(0, mid)
    else:  # 'odd'
        left_range = range(0, mid + 1)  # include center column

    # Ground-layer placement then vertical stacking
    for iy in range(ny):
        for ix in left_range:
            if rng.random() < p0:
                h = 1
                while (h < max_h) and (rng.random() < p_stack):
                    h += 1
                for k in range(h):  # k = 0 at floor
                    authored_half.add((ix, iy, k))

    # Mirror into the opposite half
    full_grid: set[Tuple[int, int, int]] = set(authored_half)
    for (ix, iy, k) in list(authored_half):
        j = _mirror_ix(ix, nx, symmetry)
        if j is not None:
            full_grid.add((j, iy, k))

    # Compute base positions in world space (along X, centered in Y)
    base_offset = size_x * 0.35
    base_y = 0.0
    red_base = (-base_offset, base_y, 0.0)
    blue_base = ( base_offset, base_y, 0.0)

    # Clear spawn bubbles (3D columns) around each base.
    def clear_spawn(center_xy: Tuple[float, float]):
        cx, cy = center_xy
        cx_i = world_to_ix(cx)
        cy_i = world_to_iy(cy)
        to_remove = []
        r2 = float(spawn_clear_r * spawn_clear_r)
        for (ix, iy, k) in full_grid:
            dx = ix - cx_i
            dy = iy - cy_i
            if (dx * dx + dy * dy) <= r2:
                to_remove.append((ix, iy, k))
        for key in to_remove:
            full_grid.discard(key)

    clear_spawn((red_base[0], red_base[1]))
    clear_spawn((blue_base[0], blue_base[1]))

    # Collapse vertical stacks (same ix,iy with consecutive k) into single boxes
    blocks: List[Block] = []
    for iy in range(ny):
        for ix in range(nx):
            k = 0
            while k < max_h:
                if (ix, iy, k) not in full_grid:
                    k += 1
                    continue
                # run length in k
                k0 = k
                while (k < max_h) and ((ix, iy, k) in full_grid):
                    k += 1
                height_cubes = k - k0
                cz = (k0 + height_cubes * 0.5) * cube
                cx = origin_x + (ix + 0.5) * cube
                cy = origin_y + (iy + 0.5) * cube
                blocks.append(Block(
                    pos=(cx, cy, cz),
                    size=(cube, cube, height_cubes * cube)
                ))
            # k continues from last value per loop; fine

    # Flag stands: put at base XY, top at floor
    red_flag_stand = (red_base[0], red_base[1], 0.0)
    blue_flag_stand = (blue_base[0], blue_base[1], 0.0)

    return MapData(
        blocks=blocks,
        red_base=red_base,
        blue_base=blue_base,
        red_flag_stand=red_flag_stand,
        blue_flag_stand=blue_flag_stand,
        bounds=(size_x, size_z),
        cube_size=cube,
        agent_radius=0.5  # matches 1-cube width
    )
