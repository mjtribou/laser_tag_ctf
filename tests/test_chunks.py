from world.chunks import ChunkIndex
from world.map_adapter import load_map_to_voxels


def test_chunk_iteration_covers_voxels_once():
    grid, _ = load_map_to_voxels("configs/maps/block_demo.json")
    index = ChunkIndex(grid, chunk_size=(4, 4, 4))

    seen = set()
    for key in index.iter_chunk_keys():
        for x, y, z, block in index.iter_blocks_in_chunk(key):
            assert block != 0
            assert (x, y, z) not in seen
            seen.add((x, y, z))
    # Compare with brute-force count
    brute = set()
    mins, maxs = grid.bounds()
    for x in range(mins[0], maxs[0] + 1):
        for y in range(mins[1], maxs[1] + 1):
            for z in range(mins[2], maxs[2] + 1):
                if grid.get(x, y, z) != 0:
                    brute.add((x, y, z))
    assert seen == brute
