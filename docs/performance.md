# Performance Notes

## Why chunk meshing matters

Lasertag environments can contain tens of thousands of cubes. Instancing each cube as a separate Panda3D `GeomNode` drives render cost through:

- **Draw submissions**: every node/geom combination makes the driver do work. Reducing 10k cubes to a few hundred chunk meshes drops CPU overhead dramatically.
- **Cull traversal**: Panda still walks the scene graph each frame. Fewer nodes means cheaper culling and state changes.
- **Vertex processing**: combining faces allows greedy merges (top faces first) to shrink vertex counts without losing detail on flat surfaces.

## Config flags

Engine-side toggles live in `config/engine.json` and are read via `engine.config.get(...)`:

- `world.chunking.enabled` — switch between legacy per-cube instancing and chunk meshing (`client.py:568`).
- `world.chunking.size` — chunk dimensions in voxels; benchmark with `tools/bench_map_load.py`.
- `world.render_distance_chunks` — how many chunk shells (Chebyshev metric) stay active (`world/chunk_manager.py:21`).
- `world.chunking.update_hz` — debounce rate for chunk activation (default 8 Hz).
- `world.chunking.greedy` — enable greedy top-face merging when atlases are disabled (`render/chunk_mesher.py:188`).
- `world.texture_atlas.enabled` plus `world.texture_atlas.path/meta` — enable per-tile UV lookup (`render/chunk_mesher.py:256`).

## Atlas format

The atlas metadata is a JSON dictionary mapping tile names to UV bounds:

```json
{
  "stone": { "u0": 0.0, "v0": 0.0, "u1": 0.25, "v1": 0.25 },
  "metal": [0.25, 0.0, 0.5, 0.25]
}
```

Both object and list shorthand are supported (`tools/bench_map_load.py:46`). The atlas image should live at `world.texture_atlas.path` and contain the tiles arranged in a grid.

## Targets

Aim to keep visible chunk meshes under ~200 geoms per frame. Use `PERF_DUMP=1` or `tools/bench_map_load.py` to monitor:

- Meshing time (`tools/bench_map_load.py` output).
- Geoms vs. vertices (`client.py:1000` perf log, `render/chunk_mesher.py`).
- Active chunk counts (`world/chunk_manager.py:96`).

If counts creep above target, reduce `world.chunking.size` or increase `world.render_distance_chunks` to trade geometry density vs. streaming.
