import json
from game.map_gen import load_from_file
from world.map_adapter import BlockDef, load_map_to_voxels


def _count_solid(grid):
    mins, maxs = grid.bounds()
    solid = 0
    for x in range(mins[0], maxs[0] + 1):
        for y in range(mins[1], maxs[1] + 1):
            for z in range(mins[2], maxs[2] + 1):
                if grid.get(x, y, z) != 0:
                    solid += 1
    return solid


def test_block_demo_voxel_counts():
    grid, registry, nav = load_map_to_voxels("configs/maps/block_demo.json")
    assert registry[1] == BlockDef(material="stone", atlas_tile="stone")
    assert not nav.nodes and not nav.links

    mapdata = load_from_file("configs/maps/block_demo.json")
    assert _count_solid(grid) == len(mapdata.blocks)


def test_csg_showcase_voxel_counts():
    grid, _, nav = load_map_to_voxels("configs/maps/csg_showcase.json")
    mapdata = load_from_file("configs/maps/csg_showcase.json")
    assert _count_solid(grid) == len(mapdata.blocks)
    # ensure grid is non-trivial in size
    _, maxs = grid.bounds()
    assert maxs[0] > 0 and maxs[1] > 0 and maxs[2] >= 0
    assert isinstance(nav.nodes, dict)


def test_nav_nodes_parse(tmp_path):
    sample = {
        "bounds": [10, 10],
        "cube_size": 1,
        "agent_radius": 0.5,
        "red_base": [-3, 0, 0],
        "blue_base": [3, 0, 0],
        "red_flag_stand": [-3, 0, 0],
        "blue_flag_stand": [3, 0, 0],
        "neutral_flag_stand": [0, 0, 0],
        "blocks": [
            {"pos": [0, 0, 0.5], "size": [1, 1, 1], "box_type": 0},
            {"pos": [2, 0, 0.5], "size": [1, 1, 1], "box_type": 0},
        ],
        "nav": {
            "nodes": [
                {"id": "alpha", "pos": [0, 0, 0], "type": "cover", "tags": ["defend", "spawn"], "facing": [1, 0, 0], "radius": 1.5},
                {"id": "beta", "pos": [4, 0, 0], "tags": ["attack"]},
            ],
            "links": [
                {"from": "alpha", "to": "beta", "weight": 2.0, "bidirectional": False},
            ],
        },
    }

    map_path = tmp_path / "nav_demo.json"
    with map_path.open("w", encoding="utf-8") as handle:
        json.dump(sample, handle)

    grid, registry, nav = load_map_to_voxels(str(map_path))

    # Baseline voxel sanity
    assert _count_solid(grid) == 2
    assert 1 in registry and registry[1].opaque

    assert "alpha" in nav.nodes and "beta" in nav.nodes
    alpha = nav.nodes["alpha"]
    assert alpha.kind == "cover"
    assert set(alpha.tags) == {"defend", "spawn"}
    assert nav.links and nav.links[0].source == "alpha" and nav.links[0].target == "beta"
