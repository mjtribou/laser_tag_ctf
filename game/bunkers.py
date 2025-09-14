# game/bunkers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json, math, random, os

# Reuse your Block dataclass from map_gen
try:
    from .map_gen import Block
except Exception:
    # Lightweight fallback for type hints if importing during static checks
    from dataclasses import dataclass
    @dataclass
    class Block:
        pos: Tuple[float, float, float]
        size: Tuple[float, float, float]
        box_type: int = 0

# ---- Data types -------------------------------------------------------------

@dataclass
class VoxelShape:
    name: str
    voxels: List[Tuple[int,int,int]]         # [(x,y,z)] in cube units, Panda axes: X right, Y fwd, Z up
    allowed_rotations: List[int]             # subset of [0,90,180,270]
    anchor: str = "center_bottom"            # how (x,y,z) are anchored when placed
    max_per_map: int = 999
    clear_radius: float = 0.0                # keep-out radius (XY) around shape center (meters)
    tags: Optional[List[str]] = None

class BunkerLibrary:
    def __init__(self, cube_size: float = 1.0):
        self.cube_size = float(cube_size)
        self.shapes: Dict[str, VoxelShape] = {}

    @staticmethod
    def from_json(path: str, default_cube: float = 1.0) -> "BunkerLibrary":
        with open(path, "r") as f:
            data = json.load(f)
        cube = float(data.get("cube_size", default_cube))
        lib = BunkerLibrary(cube)
        for s in data.get("shapes", []):
            shape = VoxelShape(
                name=s["name"],
                voxels=[tuple(map(int, v)) for v in s["voxels"]],
                allowed_rotations=[int(r) for r in s.get("allowed_rotations", [0,90,180,270])],
                anchor=s.get("anchor", "center_bottom"),
                max_per_map=int(s.get("max_per_map", 999)),
                clear_radius=float(s.get("clear_radius", 0.0)),
                tags=list(s.get("tags", [])) if s.get("tags") else None,
            )
            lib.shapes[shape.name] = shape
        return lib

# ---- Helpers ----------------------------------------------------------------

def _rot_xy(ix: int, iy: int, rot_deg: int) -> Tuple[int,int]:
    r = rot_deg % 360
    if r == 0:   return ix, iy
    if r == 90:  return -iy, ix
    if r == 180: return -ix, -iy
    if r == 270: return iy, -ix
    raise ValueError("Rotation must be one of 0,90,180,270")

def _bbox(voxels: List[Tuple[int,int,int]]) -> Tuple[int,int,int,int,int,int]:
    xs = [v[0] for v in voxels]; ys = [v[1] for v in voxels]; zs = [v[2] for v in voxels]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

def _anchor_offset(voxels: List[Tuple[int,int,int]], anchor: str) -> Tuple[float,float,float]:
    x0,x1,y0,y1,z0,z1 = _bbox(voxels)
    if anchor == "center_bottom":
        ax = 0.5 * (x0 + x1)
        ay = 0.5 * (y0 + y1)
        az = float(z0)   # sit on floor
        return ax, ay, az
    # Extendable for other anchors if you need them later
    return 0.0, 0.0, float(z0)

def _overlap_aabb(ca, sa, cb, sb) -> bool:
    ax,ay,az = ca; asx,asy,asz = sa
    bx,by,bz = cb; bsx,bsy,bsz = sb
    return (abs(ax-bx) <= (asx+bsx)*0.5 and
            abs(ay-by) <= (asy+bsy)*0.5 and
            abs(az-bz) <= (asz+bsz)*0.5)

def _distance2(x0,y0,x1,y1) -> float:
    dx, dy = x1-x0, y1-y0
    return dx*dx + dy*dy

def _mirror_rot_z(rot_deg: int) -> int:
    # Mirror across X=0 plane after rotation: equivalent to rot -> (180 - rot) mod 360
    return (180 - (rot_deg % 360)) % 360

# ---- Core: instantiate & place ---------------------------------------------

def instantiate_blocks(shape: VoxelShape,
                       world_xy: Tuple[float,float],
                       rot_deg: int,
                       cube: float,
                       mirror_x: bool = False) -> List[Block]:
    """
    Convert a VoxelShape to concrete Block(1x1x1) list at a world XY, rotated about Z.
    Anchor 'center_bottom' puts the shape centered in XY, sitting on z=0 floor.
    """
    # Rotate voxels first
    vox_r = [(_rot_xy(ix, iy, rot_deg) + (iz,)) for (ix,iy,iz) in shape.voxels]
    # Mirror across X if requested (reflect x)
    if mirror_x:
        vox_r = [((-ix), iy, iz) for (ix,iy,iz) in vox_r]

    ax, ay, az = _anchor_offset(vox_r, shape.anchor)
    cx, cy = world_xy

    blocks: List[Block] = []
    for (ix, iy, iz) in vox_r:
        wx = cx + (ix - ax) * cube
        wy = cy + (iy - ay) * cube
        wz = (iz + 0.5 - az) * cube   # each cube center at (k + 0.5) * cube
        # Default bunker cube type 0 for a subtle visual distinction
        blocks.append(Block(pos=(wx, wy, wz), size=(cube, cube, cube), box_type=0))
    return blocks

def place_bunkers(mapdata,
                  lib: BunkerLibrary,
                  count_range: Tuple[int,int] = (10, 16),
                  symmetry: str = "even",             # "even" mirrors across X=0; "odd" also allows a few centered
                  keepout_radius: float = 6.0,
                  rng: Optional[random.Random] = None,
                  allowed_tags: Optional[List[str]] = None) -> int:
    """
    Mutates mapdata.blocks by inserting bunker blocks.
    - Symmetry across X=0 (left/right halves) to match base placement.
    - Honors keepout radius around both bases & centerline.
    - Avoids overlap with existing blocks.
    Returns number of bunker *instances* (one instance = one shape, not counting its mirror).
    """
    if rng is None:
        rng = random.Random()

    size_x, size_y = mapdata.bounds
    cube = float(getattr(mapdata, "cube_size", lib.cube_size) or 1.0)
    origin_x = -0.5 * size_x
    origin_y = -0.5 * size_y
    nx = max(1, int(round(size_x / cube)))
    ny = max(1, int(round(size_y / cube)))

    # Build a list of candidate grid cells on the RIGHT half (x >= 0), leaving margin from edges & center line
    margin = max(2, int(round(2.0 / cube)))  # ~2 m
    ix_mid = nx // 2
    candidates: List[Tuple[int,int]] = []
    for iy in range(margin, ny - margin):
        for ix in range(ix_mid + margin, nx - margin):
            x = origin_x + (ix + 0.5) * cube
            y = origin_y + (iy + 0.5) * cube
            # Keep away from bases and center line; use map config spawn clear radius as a guide
            if _distance2(x, y, mapdata.red_base[0], mapdata.red_base[1]) < keepout_radius**2:
                continue
            if _distance2(x, y, mapdata.blue_base[0], mapdata.blue_base[1]) < keepout_radius**2:
                continue
            # a little gap from the mirror line
            if abs(x) < (keepout_radius * 0.5):
                continue
            candidates.append((ix, iy))

    rng.shuffle(candidates)
    want = rng.randint(count_range[0], count_range[1])

    # Optionally filter shapes by tag
    shapes = list(lib.shapes.values())
    if allowed_tags:
        shapes = [s for s in shapes if any(t in (s.tags or []) for t in allowed_tags)]
    if not shapes:
        return 0

    # Collision checker against current blocks
    def collides(new_blocks: List[Block]) -> bool:
        for nb in new_blocks:
            for eb in getattr(mapdata, "blocks", []):
                if _overlap_aabb(nb.pos, nb.size, eb.pos, eb.size):
                    return True
        return False

    placed = 0
    used_per_shape: Dict[str,int] = {}

    for (ix, iy) in candidates:
        if placed >= want:
            break
        shape = rng.choice(shapes)
        if used_per_shape.get(shape.name, 0) >= shape.max_per_map:
            continue
        rot = rng.choice(shape.allowed_rotations)

        # World center from grid
        wx = origin_x + (ix + 0.5) * cube
        wy = origin_y + (iy + 0.5) * cube

        # Primary placement (right side)
        inst_a = instantiate_blocks(shape, (wx, wy), rot_deg=rot, cube=cube, mirror_x=False)
        if shape.clear_radius > 0.0:
            # quick keep-out around this instance center
            if _distance2(wx, wy, mapdata.red_base[0], mapdata.red_base[1]) < shape.clear_radius**2:
                continue
            if _distance2(wx, wy, mapdata.blue_base[0], mapdata.blue_base[1]) < shape.clear_radius**2:
                continue
        if collides(inst_a):
            continue

        # Mirror placement (left side), across X=0
        wx_m = -wx
        rot_m = _mirror_rot_z(rot)
        inst_b = instantiate_blocks(shape, (wx_m, wy), rot_deg=rot_m, cube=cube, mirror_x=True)
        if collides(inst_b):
            continue

        # Commit both
        mapdata.blocks.extend(inst_a)
        mapdata.blocks.extend(inst_b)
        used_per_shape[shape.name] = used_per_shape.get(shape.name, 0) + 2
        placed += 1

    # Optional: a few “centerline” pieces for ODD symmetry
    if symmetry.lower().startswith("odd"):
        tries = 0
        while placed < want and tries < 40:
            tries += 1
            # pick near center line x≈0
            ix = ix_mid
            iy = rng.randint(margin, ny - margin - 1)
            wx = origin_x + (ix + 0.5) * cube
            wy = origin_y + (iy + 0.5) * cube
            shape = rng.choice(shapes)
            rot  = rng.choice(shape.allowed_rotations)
            inst = instantiate_blocks(shape, (wx, wy), rot_deg=rot, cube=cube, mirror_x=False)
            if collides(inst):
                continue
            mapdata.blocks.extend(inst)
            placed += 1

    return placed
