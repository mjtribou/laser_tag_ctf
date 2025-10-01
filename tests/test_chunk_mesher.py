from panda3d.core import GeomVertexReader, Texture

from render.chunk_mesher import ChunkMesher
from world.map_adapter import BlockDef


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
    mesher = ChunkMesher({1: BlockDef("stone", "stone")}, use_atlas=False, cube_size=1.0)
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
    mesher = ChunkMesher({1: BlockDef("stone", "stone")}, use_atlas=False, cube_size=1.0)
    node = mesher.build_geomnode(blocks, (0, 0, 0), (3, 3, 3))
    tri_count = _collect_triangles(node)
    # Surface area of 3x3x3 cube: 6 faces * 3*3 quads each = 54 quads = 108 triangles
    assert tri_count == 108


def test_atlas_uv_lookup():
    meta = {"stone": (0.25, 0.5, 0.5, 0.75)}
    texture = Texture()
    mesher = ChunkMesher(
        {1: BlockDef("stone", "stone")},
        use_atlas=True,
        cube_size=1.0,
        atlas_meta=meta,
        atlas_texture=texture,
    )
    node = mesher.build_geomnode([(0, 0, 0, 1)], (0, 0, 0), (1, 1, 1))
    vdata = node.node().getGeom(0).getVertexData()
    rdr = GeomVertexReader(vdata, "texcoord")
    coords = [tuple(round(c, 5) for c in rdr.getData2f()) for _ in range(4)]
    assert coords == [(0.25, 0.5), (0.5, 0.5), (0.5, 0.75), (0.25, 0.75)]
    textures = node.findAllTextures()
    assert textures.getNumTextures() == 1
