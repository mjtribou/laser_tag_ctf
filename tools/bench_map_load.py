"""Benchmark map loading and chunk meshing."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple, Set

from panda3d.core import loadPrcFileData

# Configure Panda3D for headless/offscreen use before importing ShowBase.
loadPrcFileData("bench", "window-type offscreen")
loadPrcFileData("bench", "audio-library-name null")
loadPrcFileData("bench", "sync-video false")

from direct.showbase.ShowBase import ShowBase

from engine.config import get as engine_config_get
from game.map_gen import load_from_file as load_map_from_file
from render.chunk_mesher import ChunkMesher
from world.map_adapter import load_map_to_voxels
from world.chunks import ChunkIndex, ChunkKey


def _resolve_map_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate = os.path.join("configs", "maps", path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Map file '{path}' not found (checked current directory and configs/maps)")


def _load_atlas_assets(loader) -> Tuple[object, Dict[str, Tuple[float, float, float, float]]]:
    path = engine_config_get("world.texture_atlas.path", "assets/atlas/atlas.png")
    meta_path = engine_config_get("world.texture_atlas.meta", "assets/atlas/atlas_meta.json")

    texture = None
    try:
        texture = loader.loadTexture(path)
    except Exception as exc:
        print(f"[bench] atlas texture load failed: {exc}")

    meta: Dict[str, Tuple[float, float, float, float]] = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            raw = __import__("json").load(handle)
        if isinstance(raw, dict):
            for name, coords in raw.items():
                try:
                    if isinstance(coords, dict):
                        u0 = float(coords.get("u0", 0.0))
                        v0 = float(coords.get("v0", 0.0))
                        u1 = float(coords.get("u1", 1.0))
                        v1 = float(coords.get("v1", 1.0))
                    elif isinstance(coords, (list, tuple)) and len(coords) == 4:
                        u0, v0, u1, v1 = map(float, coords)
                    else:
                        continue
                    meta[str(name)] = (u0, v0, u1, v1)
                except Exception:
                    continue
    except Exception as exc:
        print(f"[bench] atlas meta load failed: {exc}")

    return texture, meta


def _voxel_origin_indices(map_blocks, cube_size: float) -> Tuple[int, int, int]:
    origin = [0, 0, 0]
    first = True
    for block in map_blocks:
        indices = []
        for axis in range(3):
            idx = int(round((block.pos[axis] - 0.5 * cube_size) / cube_size))
            indices.append(idx)
        if first:
            origin = indices
            first = False
        else:
            for axis in range(3):
                if indices[axis] < origin[axis]:
                    origin[axis] = indices[axis]
    return tuple(origin)


def _iter_chunk_blocks(
    chunk_index: ChunkIndex,
    origin_indices: Tuple[int, int, int],
) -> Iterable[Tuple[ChunkKey, List[Tuple[int, int, int, int]]]]:
    for key in chunk_index.iter_chunk_keys():
        blocks: List[Tuple[int, int, int, int]] = []
        for x, y, z, block_id in chunk_index.iter_blocks_in_chunk(key):
            wx = x + origin_indices[0]
            wy = y + origin_indices[1]
            wz = z + origin_indices[2]
            blocks.append((wx, wy, wz, block_id))
        if blocks:
            yield key, blocks


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark map chunk meshing")
    parser.add_argument("--map", required=True, help="Map json filename or path")
    parser.add_argument("--chunk", nargs=3, type=int, metavar=("X", "Y", "Z"), default=[16, 16, 16])
    parser.add_argument("--greedy", action="store_true", help="Force greedy meshing on")
    parser.add_argument("--no-greedy", action="store_true", help="Force greedy meshing off")
    parser.add_argument("--atlas", action="store_true", help="Force atlas usage on")
    parser.add_argument("--no-atlas", action="store_true", help="Force atlas usage off")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    map_path = _resolve_map_path(args.map)
    chunk_size = tuple(max(1, int(v)) for v in args.chunk)

    t_start = time.perf_counter()
    mapdata = load_map_from_file(map_path)
    grid, registry = load_map_to_voxels(map_path)
    load_time = time.perf_counter() - t_start

    cube_size = float(getattr(mapdata, "cube_size", 1.0) or 1.0)
    origin_indices = _voxel_origin_indices(mapdata.blocks, cube_size)

    chunk_index = ChunkIndex(grid, chunk_size=chunk_size)

    use_atlas = bool(engine_config_get("world.texture_atlas.enabled", False))
    if args.atlas:
        use_atlas = True
    if args.no_atlas:
        use_atlas = False

    greedy_cfg = engine_config_get("world.chunking.greedy", False)
    if isinstance(greedy_cfg, str):
        greedy_enabled = greedy_cfg.strip().lower() not in ("", "0", "false", "no")
    else:
        greedy_enabled = bool(greedy_cfg)
    if args.greedy:
        greedy_enabled = True
    if args.no_greedy:
        greedy_enabled = False

    base = ShowBase()

    atlas_texture = None
    atlas_meta: Dict[str, Tuple[float, float, float, float]] = {}
    if use_atlas:
        atlas_texture, atlas_meta = _load_atlas_assets(base.loader)
        if atlas_texture is None or not atlas_meta:
            print("[bench] Atlas assets missing; disabling atlas usage")
            use_atlas = False

    chunk_blocks = list(_iter_chunk_blocks(chunk_index, origin_indices))

    solid_voxels: Set[Tuple[int, int, int]] = set()
    for _key, blocks in chunk_blocks:
        for wx, wy, wz, block_id in blocks:
            if block_id != 0:
                block_def = registry.get(block_id)
                opaque = True if block_def is None else getattr(block_def, "opaque", True)
                if opaque:
                    solid_voxels.add((wx, wy, wz))

    def is_solid(wx: int, wy: int, wz: int) -> bool:
        return (wx, wy, wz) in solid_voxels

    mesher = ChunkMesher(
        registry,
        use_atlas,
        cube_size=cube_size,
        atlas_meta=atlas_meta,
        atlas_texture=atlas_texture,
        greedy=greedy_enabled,
    )

    total_vertices = 0
    total_geoms = 0
    chunk_count = 0

    t_mesh_start = time.perf_counter()
    for key, blocks in chunk_blocks:
        (start_x, start_y, start_z), _ = chunk_index.world_bounds_in_chunk(key)
        chunk_origin = (
            start_x + origin_indices[0],
            start_y + origin_indices[1],
            start_z + origin_indices[2],
        )
        node = mesher.build_geomnode(blocks, chunk_origin, chunk_size, solid_lookup=is_solid)
        geom_node = node.node()
        if geom_node.getNumGeoms() == 0:
            continue
        chunk_count += 1
        total_geoms += geom_node.getNumGeoms()
        for gi in range(geom_node.getNumGeoms()):
            geom = geom_node.getGeom(gi)
            vdata = geom.getVertexData()
            if vdata is not None:
                total_vertices += vdata.getNumRows()
    meshing_time = time.perf_counter() - t_mesh_start

    print("Map:", map_path)
    print("Chunk size:", chunk_size)
    print(f"Load time: {load_time:.3f} s")
    print(f"Meshing time: {meshing_time:.3f} s")
    print("Chunks:", chunk_count)
    print("Geoms:", total_geoms)
    print("Vertices:", total_vertices)

    base.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
