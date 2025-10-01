"""Chunk meshing helpers to emit Panda3D geometry per chunk."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Set

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
        greedy: bool = False,
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
        self.greedy = bool(greedy)

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
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]] = None,
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

        chunk_origin_world = (
            float(chunk_origin_xyz[0]) * self._scale,
            float(chunk_origin_xyz[1]) * self._scale,
            float(chunk_origin_xyz[2]) * self._scale,
        )

        greedy_enabled = self.greedy

        if greedy_enabled:
            vertex_index = self._emit_with_greedy(
                block_map,
                chunk_origin_world,
                vwriter,
                nwriter,
                twriter,
                prim,
                triangles_out,
            )
        else:
            vertex_index = self._emit_simple(
                block_map,
                chunk_origin_world,
                vwriter,
                nwriter,
                twriter,
                prim,
                triangles_out,
            )

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
        node.setPos(*chunk_origin_world)
        return node

    def _emit_simple(
        self,
        block_map: Dict[Tuple[int, int, int], int],
        chunk_origin_world: Tuple[float, float, float],
        vwriter: GeomVertexWriter,
        nwriter: GeomVertexWriter,
        twriter: GeomVertexWriter,
        prim: GeomTriangles,
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
    ) -> int:
        vertex_index = 0
        for (wx, wy, wz), block_id in block_map.items():
            if block_id == 0:
                continue
            uv_set = self._uvs_for_block(block_id)
            world_base = (
                float(wx) * self._scale,
                float(wy) * self._scale,
                float(wz) * self._scale,
            )
            local_base = (
                world_base[0] - chunk_origin_world[0],
                world_base[1] - chunk_origin_world[1],
                world_base[2] - chunk_origin_world[2],
            )
            for face_dir, offsets, normal in self._face_defs:
                nx = wx + face_dir[0]
                ny = wy + face_dir[1]
                nz = wz + face_dir[2]
                if (nx, ny, nz) in block_map:
                    continue
                verts_local: List[Tuple[float, float, float]] = []
                verts_world: List[Tuple[float, float, float]] = []
                for (dx, dy, dz), uv in zip(offsets, uv_set):
                    vx = local_base[0] + dx * self._scale
                    vy = local_base[1] + dy * self._scale
                    vz = local_base[2] + dz * self._scale
                    verts_local.append((vx, vy, vz))
                    verts_world.append((chunk_origin_world[0] + vx,
                                         chunk_origin_world[1] + vy,
                                         chunk_origin_world[2] + vz))
                vertex_index = self._write_face(
                    verts_local,
                    verts_world,
                    normal,
                    uv_set,
                    vwriter,
                    nwriter,
                    twriter,
                    prim,
                    vertex_index,
                    triangles_out,
                )
        return vertex_index

    def _emit_with_greedy(
        self,
        block_map: Dict[Tuple[int, int, int], int],
        chunk_origin_world: Tuple[float, float, float],
        vwriter: GeomVertexWriter,
        nwriter: GeomVertexWriter,
        twriter: GeomVertexWriter,
        prim: GeomTriangles,
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
    ) -> int:
        vertex_index, emitted_top = self._emit_greedy_top_faces(
            block_map,
            chunk_origin_world,
            vwriter,
            nwriter,
            twriter,
            prim,
            triangles_out,
        )
        vertex_index = self._emit_remaining_faces(
            block_map,
            chunk_origin_world,
            vwriter,
            nwriter,
            twriter,
            prim,
            triangles_out,
            vertex_index,
            emitted_top,
        )
        return vertex_index

    def _emit_greedy_top_faces(
        self,
        block_map: Dict[Tuple[int, int, int], int],
        chunk_origin_world: Tuple[float, float, float],
        vwriter: GeomVertexWriter,
        nwriter: GeomVertexWriter,
        twriter: GeomVertexWriter,
        prim: GeomTriangles,
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
    ) -> Tuple[int, Set[Tuple[int, int, int]]]:
        # TODO: Extend greedy merging to the remaining face directions when necessary.
        layers: Dict[int, Dict[Tuple[int, int], Tuple[int, Tuple[Tuple[float, float], ...]]]] = {}
        for (wx, wy, wz), block_id in block_map.items():
            if block_id == 0:
                continue
            if (wx, wy, wz + 1) in block_map:
                continue
            uv_set = self._uvs_for_block(block_id)
            layers.setdefault(wz, {})[(wx, wy)] = (block_id, uv_set)

        emitted: Set[Tuple[int, int, int]] = set()
        vertex_index = 0
        normal = (0.0, 0.0, 1.0)
        step = self._scale

        for z in sorted(layers.keys()):
            grid = layers[z]
            used: Set[Tuple[int, int]] = set()
            for (x, y) in sorted(grid.keys()):
                if (x, y) in used:
                    continue
                block_id, uv_base = grid[(x, y)]
                width = 1
                while True:
                    cell = (x + width, y)
                    data = grid.get(cell)
                    if data is None or data[0] != block_id or data[1] != uv_base or cell in used:
                        break
                    width += 1

                height = 1
                while True:
                    row_y = y + height
                    row_ok = True
                    for dx in range(width):
                        cell = (x + dx, row_y)
                        data = grid.get(cell)
                        if data is None or data[0] != block_id or data[1] != uv_base or cell in used:
                            row_ok = False
                            break
                    if not row_ok:
                        break
                    height += 1

                for dx in range(width):
                    for dy in range(height):
                        used.add((x + dx, y + dy))
                        emitted.add((x + dx, y + dy, z))

                x0 = x * step
                x1 = (x + width) * step
                y0 = y * step
                y1 = (y + height) * step
                z_top = (z + 1) * step

                verts_local = [
                    (x0 - chunk_origin_world[0], y0 - chunk_origin_world[1], z_top - chunk_origin_world[2]),
                    (x1 - chunk_origin_world[0], y0 - chunk_origin_world[1], z_top - chunk_origin_world[2]),
                    (x1 - chunk_origin_world[0], y1 - chunk_origin_world[1], z_top - chunk_origin_world[2]),
                    (x0 - chunk_origin_world[0], y1 - chunk_origin_world[1], z_top - chunk_origin_world[2]),
                ]
                verts_world = [
                    (x0, y0, z_top),
                    (x1, y0, z_top),
                    (x1, y1, z_top),
                    (x0, y1, z_top),
                ]

                if self.use_atlas:
                    scaled_uv = uv_base
                else:
                    u0, v0 = uv_base[0]
                    u1, _ = uv_base[1]
                    _, v1 = uv_base[2]
                    du = (u1 - u0) if abs(u1 - u0) > 1e-8 else 1.0
                    dv = (v1 - v0) if abs(v1 - v0) > 1e-8 else 1.0
                    scaled_uv = (
                        (u0, v0),
                        (u0 + du * width, v0),
                        (u0 + du * width, v0 + dv * height),
                        (u0, v0 + dv * height),
                    )

                vertex_index = self._write_face(
                    verts_local,
                    verts_world,
                    normal,
                    scaled_uv,
                    vwriter,
                    nwriter,
                    twriter,
                    prim,
                    vertex_index,
                    triangles_out,
                )

        return vertex_index, emitted

    def _emit_remaining_faces(
        self,
        block_map: Dict[Tuple[int, int, int], int],
        chunk_origin_world: Tuple[float, float, float],
        vwriter: GeomVertexWriter,
        nwriter: GeomVertexWriter,
        twriter: GeomVertexWriter,
        prim: GeomTriangles,
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
        vertex_index: int,
        skip_top: Set[Tuple[int, int, int]],
    ) -> int:
        for (wx, wy, wz), block_id in block_map.items():
            if block_id == 0:
                continue
            uv_set = self._uvs_for_block(block_id)
            world_base = (
                float(wx) * self._scale,
                float(wy) * self._scale,
                float(wz) * self._scale,
            )
            local_base = (
                world_base[0] - chunk_origin_world[0],
                world_base[1] - chunk_origin_world[1],
                world_base[2] - chunk_origin_world[2],
            )
            for face_dir, offsets, normal in self._face_defs:
                if face_dir == (0, 0, 1) and (wx, wy, wz) in skip_top:
                    continue
                nx = wx + face_dir[0]
                ny = wy + face_dir[1]
                nz = wz + face_dir[2]
                if (nx, ny, nz) in block_map:
                    continue
                verts_local: List[Tuple[float, float, float]] = []
                verts_world: List[Tuple[float, float, float]] = []
                for (dx, dy, dz), uv in zip(offsets, uv_set):
                    vx = local_base[0] + dx * self._scale
                    vy = local_base[1] + dy * self._scale
                    vz = local_base[2] + dz * self._scale
                    verts_local.append((vx, vy, vz))
                    verts_world.append((chunk_origin_world[0] + vx,
                                         chunk_origin_world[1] + vy,
                                         chunk_origin_world[2] + vz))
                vertex_index = self._write_face(
                    verts_local,
                    verts_world,
                    normal,
                    uv_set,
                    vwriter,
                    nwriter,
                    twriter,
                    prim,
                    vertex_index,
                    triangles_out,
                )
        return vertex_index

    def _write_face(
        self,
        verts_local: List[Tuple[float, float, float]],
        verts_world: List[Tuple[float, float, float]],
        normal: Tuple[float, float, float],
        uv_set: Tuple[Tuple[float, float], ...],
        vwriter: GeomVertexWriter,
        nwriter: GeomVertexWriter,
        twriter: GeomVertexWriter,
        prim: GeomTriangles,
        vertex_index: int,
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
    ) -> int:
        for (vx, vy, vz), uv in zip(verts_local, uv_set):
            vwriter.addData3(vx, vy, vz)
            nwriter.addData3(*normal)
            twriter.addData2(*uv)
        prim.addVertices(vertex_index, vertex_index + 1, vertex_index + 2)
        prim.closePrimitive()
        prim.addVertices(vertex_index, vertex_index + 2, vertex_index + 3)
        prim.closePrimitive()
        if triangles_out is not None:
            triangles_out.append((verts_world[0], verts_world[1], verts_world[2]))
            triangles_out.append((verts_world[0], verts_world[2], verts_world[3]))
        return vertex_index + 4

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node.addGeom(geom)
        node = NodePath(geom_node)
        if self.use_atlas and self._atlas_texture is not None:
            node.setTexture(self._TEXTURE_STAGE, self._atlas_texture, 1)
            if self._material is not None:
                node.setMaterial(self._material, 1)
        node.setPos(*chunk_origin_world)
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
