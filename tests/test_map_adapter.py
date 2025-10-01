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
    grid, registry = load_map_to_voxels("configs/maps/block_demo.json")
    assert registry[1] == BlockDef(material="stone", atlas_tile="stone")

    mapdata = load_from_file("configs/maps/block_demo.json")
    assert _count_solid(grid) == len(mapdata.blocks)


def test_csg_showcase_voxel_counts():
    grid, _ = load_map_to_voxels("configs/maps/csg_showcase.json")
    mapdata = load_from_file("configs/maps/csg_showcase.json")
    assert _count_solid(grid) == len(mapdata.blocks)
    # ensure grid is non-trivial in size
    _, maxs = grid.bounds()
    assert maxs[0] > 0 and maxs[1] > 0 and maxs[2] >= 0
