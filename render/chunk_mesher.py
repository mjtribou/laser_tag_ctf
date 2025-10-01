"""Chunk meshing helpers to emit Panda3D geometry per chunk."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    Texture,
    TextureStage,
    Material,
)

from world.map_adapter import BlockDef


class ChunkMesher:
    """Generate opaque cube geometry for chunked voxel data."""

    _uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    _face_defs = (
        # dir, vertex offsets, normal
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), (1, 0, 0)),
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), (-1, 0, 0)),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), (0, 1, 0)),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), (0, -1, 0)),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), (0, 0, 1)),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), (0, 0, -1)),
    )

    _TEXTURE_STAGE = TextureStage("chunk_atlas")
    _TEXTURE_STAGE.setMode(TextureStage.M_modulate)

    def __init__(
        self,
        block_registry: Dict[int, BlockDef],
        use_atlas: bool,
        cube_size: float = 1.0,
        atlas_meta: Dict[str, Tuple[float, float, float, float]] | None = None,
        atlas_texture: Texture | None = None,
    ) -> None:
        self.block_registry = block_registry
        self.use_atlas = bool(use_atlas)
        self._format = GeomVertexFormat.getV3n3t2()
        self._scale = float(cube_size) if cube_size else 1.0
        self._atlas_meta = atlas_meta or {}
        self._atlas_texture = atlas_texture
        self._uv_cache: Dict[int, Tuple[Tuple[float, float], ...]] = {}
        self._material = None
        if self.use_atlas and atlas_texture is not None:
            mat = Material()
            mat.setDiffuse((1, 1, 1, 1))
            mat.setAmbient((1, 1, 1, 1))
            self._material = mat

    def _build_block_map(self, chunk_blocks_iter: Iterable[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], int]:
        block_map: Dict[Tuple[int, int, int], int] = {}
        for x, y, z, block_id in chunk_blocks_iter:
            block_map[(int(x), int(y), int(z))] = int(block_id)
        return block_map

    def build_geomnode(
        self,
        chunk_blocks_iter: Iterable[Tuple[int, int, int, int]],
        chunk_origin_xyz: Tuple[int, int, int],
        chunk_size_xyz: Tuple[int, int, int],
    ) -> NodePath:
        """Emit a `GeomNode` containing only the exposed faces for this chunk."""
        block_map = self._build_block_map(chunk_blocks_iter)
        geom_node = GeomNode("chunk")
        if not block_map:
            return NodePath(geom_node)

        vdata = GeomVertexData("chunk", self._format, Geom.UHStatic)
        vwriter = GeomVertexWriter(vdata, "vertex")
        nwriter = GeomVertexWriter(vdata, "normal")
        twriter = GeomVertexWriter(vdata, "texcoord")
        prim = GeomTriangles(Geom.UHStatic)

        ox, oy, oz = (
            float(chunk_origin_xyz[0]) * self._scale,
            float(chunk_origin_xyz[1]) * self._scale,
            float(chunk_origin_xyz[2]) * self._scale,
        )

        vertex_index = 0
        for (wx, wy, wz), block_id in block_map.items():
            if block_id == 0:
                continue
            uv_set = self._uvs_for_block(block_id)
            local_base = (
                float(wx) * self._scale - ox,
                float(wy) * self._scale - oy,
                float(wz) * self._scale - oz,
            )
            for face_dir, offsets, normal in self._face_defs:
                nx = wx + face_dir[0]
                ny = wy + face_dir[1]
                nz = wz + face_dir[2]
                if (nx, ny, nz) in block_map:
                    continue
                for (dx, dy, dz), uv in zip(offsets, uv_set):
                    vx = local_base[0] + dx * self._scale
                    vy = local_base[1] + dy * self._scale
                    vz = local_base[2] + dz * self._scale
                    vwriter.addData3(vx, vy, vz)
                    nwriter.addData3(*normal)
                    twriter.addData2(*uv)
                prim.addVertices(vertex_index, vertex_index + 1, vertex_index + 2)
                prim.closePrimitive()
                prim.addVertices(vertex_index, vertex_index + 2, vertex_index + 3)
                prim.closePrimitive()
                vertex_index += 4

        if prim.getNumPrimitives() == 0:
            return NodePath(geom_node)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node.addGeom(geom)
        node = NodePath(geom_node)
        if self.use_atlas and self._atlas_texture is not None:
            node.setTexture(self._TEXTURE_STAGE, self._atlas_texture, 1)
            if self._material is not None:
                node.setMaterial(self._material, 1)
        return node

    def _uvs_for_block(self, block_id: int) -> Tuple[Tuple[float, float], ...]:
        if not self.use_atlas:
            return self._uvs
        if block_id in self._uv_cache:
            return self._uv_cache[block_id]
        tile_name = None
        block_def = self.block_registry.get(block_id)
        if block_def is not None:
            tile_name = getattr(block_def, "atlas_tile", None)
        coords = self._atlas_meta.get(tile_name) if tile_name else None
        if coords is None:
            self._uv_cache[block_id] = self._uvs
            return self._uvs
        u0, v0, u1, v1 = coords
        uv_set = ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
        self._uv_cache[block_id] = uv_set
        return uv_set
