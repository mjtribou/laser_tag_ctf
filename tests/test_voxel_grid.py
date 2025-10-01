from world.voxel_grid import FACE_DIRECTIONS, VoxelGrid, neighbors

import pytest


def test_default_fill_and_get():
    grid = VoxelGrid((2, 3, 4))
    for x in range(2):
        for y in range(3):
            for z in range(4):
                assert grid.get(x, y, z) == 0
                assert grid.is_air(x, y, z)


def test_set_and_is_air():
    grid = VoxelGrid((4, 4, 4))
    grid.set(1, 2, 3, 5)
    assert grid.get(1, 2, 3) == 5
    assert not grid.is_air(1, 2, 3)
    grid.set(1, 2, 3, 0)
    assert grid.is_air(1, 2, 3)


def test_bounds_and_index_errors():
    grid = VoxelGrid((2, 2, 2))
    assert grid.bounds() == ((0, 0, 0), (1, 1, 1))
    with pytest.raises(IndexError):
        grid.get(2, 0, 0)
    with pytest.raises(IndexError):
        grid.set(-1, 0, 0, 1)


def test_face_constants_and_neighbors():
    assert len(FACE_DIRECTIONS) == 6
    origin_neighbors = set(neighbors(0, 0, 0))
    expected = {
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    }
    assert origin_neighbors == expected


def test_large_access_pattern():
    grid = VoxelGrid((16, 16, 16))
    for i in range(16):
        grid.set(i, i % 16, i % 16, i)
        assert grid.get(i, i % 16, i % 16) == i
