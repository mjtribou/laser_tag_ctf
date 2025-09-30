# game/map_gen.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union
import random, math, json, os
from .bunkers import BunkerLibrary, place_bunkers

@dataclass
class Block:
    pos: Tuple[float, float, float]
    size: Tuple[float, float, float]
    # Visual type for selecting texture tiles at the client (0..N-1)
    box_type: int = 0

@dataclass
class MapData:
    blocks: List[Block]
    red_base: Tuple[float, float, float]
    blue_base: Tuple[float, float, float]
    red_flag_stand: Tuple[float, float, float]
    blue_flag_stand: Tuple[float, float, float]
    # Center-flag CTF support: neutral flag stand at arena center
    neutral_flag_stand: Tuple[float, float, float]
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

    # Emit one block per cube (no vertical stack collapsing).
    # Positions are at cell centers; size is exactly one cube.
    blocks: List[Block] = []
    for (ix, iy, k) in sorted(full_grid):
        cx = origin_x + (ix + 0.5) * cube
        cy = origin_y + (iy + 0.5) * cube
        cz = (k + 0.5) * cube
        # Assign a deterministic type per cell using the RNG and grid indices
        # to provide visual variety while remaining stable for a given seed.
        t = (ix * 73856093 ^ iy * 19349663 ^ k * 83492791 ^ int(seed)) & 0xFFFFFFFF
        box_type = (t % 4)
        blocks.append(Block(pos=(cx, cy, cz), size=(cube, cube, cube), box_type=box_type))

    # Flag stands: put at base XY, top at floor
    red_flag_stand = (red_base[0], red_base[1], 0.0)
    blue_flag_stand = (blue_base[0], blue_base[1], 0.0)
    # Single-flag mode: neutral flag at arena center on the floor
    neutral_flag_stand = (0.0, 0.0, 0.0)

    mapdata = MapData(
        blocks=blocks,
        red_base=red_base,
        blue_base=blue_base,
        red_flag_stand=red_flag_stand,
        blue_flag_stand=blue_flag_stand,
        neutral_flag_stand=neutral_flag_stand,
        bounds=(size_x, size_z),
        cube_size=cube,
        agent_radius=0.5  # matches 1-cube width
    )

    # --- prefab bunkers ---
    try:
        # Default knobs
        symmetry_cfg  = (cubes or {}).get("symmetry", "even")  # "even" mirrors across X=0
        spawn_keepout = float((cubes or {}).get("spawn_clear_radius", 6.0))
        bunkers_path  = "configs/bunkers.json"  # override via config if you like

        # If a richer config file exists, allow overriding defaults
        import json, os
        if os.path.exists("configs/defaults.json"):
            _cfg = json.load(open("configs/defaults.json", "r"))
            if "bunkers" in _cfg:
                bunkers_path  = _cfg["bunkers"].get("file", bunkers_path)
                symmetry_cfg  = _cfg["bunkers"].get("symmetry", symmetry_cfg)
                spawn_keepout = float(_cfg["bunkers"].get("keepout_radius", spawn_keepout))
                count_min     = int(_cfg["bunkers"].get("count_min", 8))
                count_max     = int(_cfg["bunkers"].get("count_max", 12))
            else:
                count_min, count_max = 8, 12
        else:
            count_min, count_max = 8, 12

        if os.path.exists(bunkers_path):
            lib = BunkerLibrary.from_json(bunkers_path, default_cube=cube)
            # Seeded RNG for determinism from map seed
            rng = random.Random(int(seed))
            placed = place_bunkers(
                mapdata, lib,
                count_range=(count_min, count_max),
                symmetry=symmetry_cfg,
                keepout_radius=spawn_keepout,
                rng=rng
            )
            print(f"[map_gen] bunkers placed: {placed} instances")
    except Exception as e:
        # Non-fatal (so you can ship without a bunkers file)
        print(f"[map_gen] bunkers skipped: {e}")
        pass

    return mapdata


# ---------- Serialization helpers ----------

Number = Union[int, float]


def _encode_number(value: Any) -> Number:
    """Return ints for whole numbers to keep JSON tidy, otherwise floats."""
    try:
        fval = float(value)
    except Exception as exc:
        raise TypeError(f"Value '{value}' must be numeric") from exc
    if math.isfinite(fval) and abs(fval - round(fval)) < 1e-9:
        return int(round(fval))
    return fval

def _to_tuple(values: Any, expected_len: int, name: str, default: Optional[Tuple[float, ...]] = None) -> Tuple[float, ...]:
    """Convert a sequence into a tuple of floats of the given length."""
    if values is None:
        if default is None:
            raise ValueError(f"Missing required field '{name}'")
        return tuple(float(v) for v in default)
    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        raise ValueError(f"Field '{name}' must be a sequence of length {expected_len}")
    try:
        return tuple(float(values[i]) for i in range(expected_len))
    except Exception as exc:
        raise ValueError(f"Field '{name}' must contain numeric values") from exc


def mapdata_to_dict(mapdata: MapData) -> Dict[str, Any]:
    """Convert a MapData instance into a JSON-serializable dictionary."""
    return {
        "version": 1,
        "bounds": [_encode_number(v) for v in mapdata.bounds],
        "cube_size": _encode_number(getattr(mapdata, "cube_size", 1.0)),
        "agent_radius": _encode_number(getattr(mapdata, "agent_radius", 0.5)),
        "red_base": [_encode_number(v) for v in mapdata.red_base],
        "blue_base": [_encode_number(v) for v in mapdata.blue_base],
        "red_flag_stand": [_encode_number(v) for v in mapdata.red_flag_stand],
        "blue_flag_stand": [_encode_number(v) for v in mapdata.blue_flag_stand],
        "neutral_flag_stand": [_encode_number(v) for v in mapdata.neutral_flag_stand],
        "blocks": [
            {
                "pos": [_encode_number(v) for v in block.pos],
                "size": [_encode_number(v) for v in block.size],
                "box_type": int(getattr(block, "box_type", 0)),
            }
            for block in mapdata.blocks
        ],
    }


def mapdata_from_dict(data: Dict[str, Any]) -> MapData:
    """Create a MapData instance from a dictionary (inverse of mapdata_to_dict)."""
    if not isinstance(data, dict):
        raise TypeError("Map data must be a JSON object/dict")

    bounds = _to_tuple(data.get("bounds"), 2, "bounds")
    cube_size = float(data.get("cube_size", 1.0))
    agent_radius = float(data.get("agent_radius", 0.5))

    default_red = (-bounds[0] * 0.35, 0.0, 0.0)
    default_blue = (bounds[0] * 0.35, 0.0, 0.0)

    red_base = _to_tuple(data.get("red_base"), 3, "red_base", default_red)
    blue_base = _to_tuple(data.get("blue_base"), 3, "blue_base", default_blue)
    red_flag_stand = _to_tuple(data.get("red_flag_stand"), 3, "red_flag_stand", red_base)
    blue_flag_stand = _to_tuple(data.get("blue_flag_stand"), 3, "blue_flag_stand", blue_base)
    neutral_flag_stand = _to_tuple(data.get("neutral_flag_stand"), 3, "neutral_flag_stand", (0.0, 0.0, 0.0))

    blocks_data = data.get("blocks", [])
    if not isinstance(blocks_data, list):
        raise ValueError("Field 'blocks' must be a list")

    def _expand_to_cubes(center: Tuple[float, float, float],
                         size: Tuple[float, float, float],
                         cube: float) -> List[Tuple[float, float, float]]:
        counts: List[int] = []
        for axis in range(3):
            dim = size[axis]
            if dim <= 0.0:
                raise ValueError("Block dimensions must be positive")
            count_f = dim / cube
            count = int(round(count_f))
            if count < 1 or abs(count_f - count) > 1e-5:
                raise ValueError(
                    "Block size must be an integer multiple of cube_size; "
                    f"got {dim} on axis {axis} with cube_size {cube}"
                )
            counts.append(count)

        origin = (
            center[0] - size[0] * 0.5,
            center[1] - size[1] * 0.5,
            center[2] - size[2] * 0.5,
        )

        cubes: List[Tuple[float, float, float]] = []
        for ix in range(counts[0]):
            cx = origin[0] + (ix + 0.5) * cube
            for iy in range(counts[1]):
                cy = origin[1] + (iy + 0.5) * cube
                for iz in range(counts[2]):
                    cz = origin[2] + (iz + 0.5) * cube
                    cubes.append((cx, cy, cz))
        return cubes

    blocks: Dict[Tuple[int, int, int], Block] = {}

    for idx, b in enumerate(blocks_data):
        if not isinstance(b, dict):
            raise ValueError(f"Block #{idx} must be an object")
        pos = _to_tuple(b.get("pos"), 3, f"blocks[{idx}].pos")
        size_default = (cube_size, cube_size, cube_size)
        size = _to_tuple(b.get("size"), 3, f"blocks[{idx}].size", size_default)
        try:
            box_type = int(b.get("box_type", 0))
        except Exception as exc:
            raise ValueError(f"blocks[{idx}].box_type must be an integer") from exc

        for cube_pos in _expand_to_cubes(pos, size, cube_size):
            key = (
                int(round(cube_pos[0] / cube_size)),
                int(round(cube_pos[1] / cube_size)),
                int(round(cube_pos[2] / cube_size)),
            )
            if key not in blocks:
                blocks[key] = Block(pos=cube_pos, size=(cube_size, cube_size, cube_size), box_type=box_type)

    block_list = [blocks[key] for key in sorted(blocks.keys())]

    return MapData(
        blocks=block_list,
        red_base=red_base,
        blue_base=blue_base,
        red_flag_stand=red_flag_stand,
        blue_flag_stand=blue_flag_stand,
        neutral_flag_stand=neutral_flag_stand,
        bounds=bounds,
        cube_size=cube_size,
        agent_radius=agent_radius,
    )


def save_to_file(mapdata: MapData, path: str, *, indent: int = 2) -> None:
    """Serialize MapData to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapdata_to_dict(mapdata), f, indent=indent)


def load_from_file(path: str) -> MapData:
    """Load MapData from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return mapdata_from_dict(data)
