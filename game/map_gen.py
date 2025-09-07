# game/map_gen.py
from dataclasses import dataclass
from typing import List, Tuple
import random

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
    bounds: Tuple[float, float]  # (x_size, z_size)

def generate(seed: int, size_x: float, size_z: float) -> MapData:
    rng = random.Random(seed)
    blocks: List[Block] = []

    # Symmetric procedural blocks: place on one half, mirror to the other
    lanes = 5
    for i in range(45):
        x = rng.uniform(-size_x*0.4, size_x*0.4)
        z = rng.uniform(-size_z*0.45, size_z*0.45)
        w = rng.uniform(2.0, 6.0)
        h = rng.uniform(2.0, 3.5)
        d = rng.uniform(2.0, 6.0)
        # avoid exact mid lane sometimes to create lanes
        blocks.append(Block((x, h/2, z), (w, h, d)))

    # Clear two base zones
    base_offset = size_x*0.35
    base_z = 0.0
    # Add some base cover
    for side in (-1, 1):
        xcenter = side*base_offset
        for i in range(6):
            bx = xcenter + rng.uniform(-6, 6)
            bz = rng.uniform(-8, 8)
            bw = rng.uniform(2.0, 5.0)
            bh = rng.uniform(1.6, 2.5)
            bd = rng.uniform(2.0, 5.0)
            blocks.append(Block((bx, bh/2, bz), (bw, bh, bd)))

    red_base = (-base_offset, 0.0, base_z)
    blue_base = ( base_offset, 0.0, base_z)
    red_flag_stand = (red_base[0], 0.0, red_base[2])
    blue_flag_stand = (blue_base[0], 0.0, blue_base[2])

    return MapData(blocks=blocks, red_base=red_base, blue_base=blue_base,
                   red_flag_stand=red_flag_stand, blue_flag_stand=blue_flag_stand,
                   bounds=(size_x, size_z))
