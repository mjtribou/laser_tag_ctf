from render.chunk_mesher import BlockDef, ChunkMesher


def _collect_triangles(node):
    geom_node = node.node()
    tri_count = 0
    for i in range(geom_node.getNumGeoms()):
        geom = geom_node.getGeom(i)
        for p in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(p)
            tri_count += prim.getNumPrimitives()
    return tri_count


def test_single_cube_produces_six_quads():
    mesher = ChunkMesher({1: BlockDef("stone", "stone")}, use_atlas=False)
    node = mesher.build_geomnode([(1, 1, 1, 1)], (0, 0, 0), (3, 3, 3))
    tri_count = _collect_triangles(node)
    assert tri_count == 12
    assert node.node().getGeom(0).getVertexData().getNumRows() == 24


def test_solid_three_cube_only_shell():
    blocks = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                blocks.append((x, y, z, 1))
    mesher = ChunkMesher({1: BlockDef("stone", "stone")}, use_atlas=False)
    node = mesher.build_geomnode(blocks, (0, 0, 0), (3, 3, 3))
    tri_count = _collect_triangles(node)
    # Surface area of 3x3x3 cube: 6 faces * 3*3 quads each = 54 quads = 108 triangles
    assert tri_count == 108
