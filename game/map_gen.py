# game/map_gen.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union
import math
import json

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
                         cube: float) -> List[Tuple[Tuple[float, float, float], Tuple[int, int, int]]]:
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

        cubes: List[Tuple[Tuple[float, float, float], Tuple[int, int, int]]] = []
        base_ix = int(round(origin[0] / cube))
        base_iy = int(round(origin[1] / cube))
        base_iz = int(round(origin[2] / cube))
        for ix in range(counts[0]):
            cx = origin[0] + (ix + 0.5) * cube
            for iy in range(counts[1]):
                cy = origin[1] + (iy + 0.5) * cube
                for iz in range(counts[2]):
                    cz = origin[2] + (iz + 0.5) * cube
                    key = (base_ix + ix, base_iy + iy, base_iz + iz)
                    cubes.append(((cx, cy, cz), key))
        return cubes

    def _normalize_box_type(value: Any, path: str) -> int:
        try:
            return int(value)
        except Exception as exc:
            raise ValueError(f"{path}.box_type must be an integer") from exc

    def _resolve_box_type(node: Dict[str, Any], path: str, inherited: Optional[int]) -> int:
        if "box_type" in node:
            return _normalize_box_type(node["box_type"], path)
        if inherited is not None:
            return inherited
        return 0

    VoxelMap = Dict[Tuple[int, int, int], Tuple[Tuple[float, float, float], int]]

    def _combine_union(children: List[VoxelMap]) -> VoxelMap:
        combined: VoxelMap = {}
        for child in children:
            for key, value in child.items():
                if key not in combined:
                    combined[key] = value
        return combined

    def _combine_intersection(children: List[VoxelMap]) -> VoxelMap:
        if not children:
            return {}
        intersect: VoxelMap = dict(children[0])
        for child in children[1:]:
            keys = set(intersect.keys()) & set(child.keys())
            intersect = {k: intersect[k] for k in keys}
        return intersect

    def _combine_subtraction(children: List[VoxelMap]) -> VoxelMap:
        if not children:
            return {}
        base: VoxelMap = dict(children[0])
        for child in children[1:]:
            for key in child.keys():
                base.pop(key, None)
        return base

    def _parse_voxel_node(node: Any, path: str, inherited_box_type: Optional[int]) -> VoxelMap:
        if not isinstance(node, dict):
            raise ValueError(f"{path} must be an object")

        node_box_type = _resolve_box_type(node, path, inherited_box_type)

        if "pos" in node or "size" in node:
            pos = _to_tuple(node.get("pos"), 3, f"{path}.pos")
            size_default = (cube_size, cube_size, cube_size)
            size = _to_tuple(node.get("size"), 3, f"{path}.size", size_default)
            voxels: VoxelMap = {}
            for cube_pos, key in _expand_to_cubes(pos, size, cube_size):
                voxels[key] = (cube_pos, node_box_type)
            return voxels

        op_val = node.get("op") or node.get("operation") or node.get("type")
        if not isinstance(op_val, str):
            raise ValueError(f"{path} must define 'pos'/'size' or 'op'")
        op = op_val.lower()

        if "children" in node:
            child_key = "children"
        elif "shapes" in node:
            child_key = "shapes"
        else:
            raise ValueError(f"{path} must provide a 'children' or 'shapes' list for op '{op}'")

        children_raw = node.get(child_key)
        if not isinstance(children_raw, list) or not children_raw:
            raise ValueError(f"{path}.{child_key} must be a non-empty list")

        child_voxels = [
            _parse_voxel_node(child, f"{path}.{child_key}[{idx}]", node_box_type)
            for idx, child in enumerate(children_raw)
        ]

        if op in ("union", "add", "merge"):
            return _combine_union(child_voxels)
        if op in ("intersect", "intersection", "and"):
            return _combine_intersection(child_voxels)
        if op in ("subtract", "difference", "minus", "without"):
            return _combine_subtraction(child_voxels)

        raise ValueError(f"{path}.op '{op}' is not supported; expected union/intersect/subtract")

    blocks: Dict[Tuple[int, int, int], Block] = {}

    for idx, node in enumerate(blocks_data):
        voxels = _parse_voxel_node(node, f"blocks[{idx}]", None)
        for key, (cube_pos, box_type) in voxels.items():
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
