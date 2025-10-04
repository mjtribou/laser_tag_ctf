"""Chunk meshing helpers to emit Panda3D geometry per chunk."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set, NamedTuple

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
    MaterialAttrib,
    TextureAttrib,
    CullFaceAttrib,
    TransparencyAttrib,
    RenderState,
)

from world.map_adapter import BlockDef


class _FaceConfig(NamedTuple):
    direction: Tuple[int, int, int]
    axis: int
    positive: bool
    u_axis: int
    v_axis: int
    offsets: Tuple[Tuple[int, int, int], ...]
    normal: Tuple[int, int, int]
    u_bits: Tuple[int, int, int, int]
    v_bits: Tuple[int, int, int, int]
    idx00: int
    idx10: int
    idx11: int
    idx01: int


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
        self.greedy = bool(greedy) and not self.use_atlas
        self._render_state = self._build_render_state()
        self._face_configs: List[_FaceConfig] = []
        if self.greedy:
            for direction, offsets, normal in self._face_defs:
                axis = 0 if direction[0] != 0 else 1 if direction[1] != 0 else 2
                positive = direction[axis] > 0
                axes = [0, 1, 2]
                axes.remove(axis)
                u_axis, v_axis = axes
                u_bits = tuple(offset[u_axis] for offset in offsets)
                v_bits = tuple(offset[v_axis] for offset in offsets)
                idx00 = self._find_corner_index(u_bits, v_bits, 0, 0)
                idx10 = self._find_corner_index(u_bits, v_bits, 1, 0)
                idx11 = self._find_corner_index(u_bits, v_bits, 1, 1)
                idx01 = self._find_corner_index(u_bits, v_bits, 0, 1)
                cfg = _FaceConfig(
                    direction=direction,
                    axis=axis,
                    positive=positive,
                    u_axis=u_axis,
                    v_axis=v_axis,
                    offsets=offsets,
                    normal=normal,
                    u_bits=u_bits,
                    v_bits=v_bits,
                    idx00=idx00,
                    idx10=idx10,
                    idx11=idx11,
                    idx01=idx01,
                )
                self._face_configs.append(cfg)

    @staticmethod
    def _find_corner_index(
        u_bits: Tuple[int, int, int, int],
        v_bits: Tuple[int, int, int, int],
        target_u: int,
        target_v: int,
    ) -> int:
        for idx, (ub, vb) in enumerate(zip(u_bits, v_bits)):
            if ub == target_u and vb == target_v:
                return idx
        raise ValueError("Quad vertex mapping missing expected corner")

    def _build_block_map(self, chunk_blocks_iter: Iterable[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], int]:
        block_map: Dict[Tuple[int, int, int], int] = {}
        for x, y, z, block_id in chunk_blocks_iter:
            bid = int(block_id)
            if not self._is_opaque(bid):
                continue
            block_map[(int(x), int(y), int(z))] = bid
        return block_map

    def build_geomnode(
        self,
        chunk_blocks_iter: Iterable[Tuple[int, int, int, int]],
        chunk_origin_xyz: Tuple[int, int, int],
        chunk_size_xyz: Tuple[int, int, int],
        *,
        solid_lookup: Optional[Callable[[int, int, int], bool]] = None,
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
                solid_lookup,
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
                solid_lookup,
                triangles_out,
            )

        if prim.getNumPrimitives() == 0:
            return NodePath(geom_node)

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node.addGeom(geom)
        node = NodePath(geom_node)
        node.setState(self._render_state)
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
        solid_lookup: Optional[Callable[[int, int, int], bool]],
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
                if self._has_solid_neighbor(nx, ny, nz, block_map, solid_lookup):
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
        solid_lookup: Optional[Callable[[int, int, int], bool]],
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
    ) -> int:
        vertex_index = 0
        for config in self._face_configs:
            vertex_index = self._emit_greedy_direction(
                block_map,
                chunk_origin_world,
                vwriter,
                nwriter,
                twriter,
                prim,
                solid_lookup,
                triangles_out,
                vertex_index,
                config,
            )
        return vertex_index

    def _emit_greedy_direction(
        self,
        block_map: Dict[Tuple[int, int, int], int],
        chunk_origin_world: Tuple[float, float, float],
        vwriter: GeomVertexWriter,
        nwriter: GeomVertexWriter,
        twriter: GeomVertexWriter,
        prim: GeomTriangles,
        solid_lookup: Optional[Callable[[int, int, int], bool]],
        triangles_out: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]],
        vertex_index: int,
        config: _FaceConfig,
    ) -> int:
        planes: Dict[int, Dict[Tuple[int, int], Tuple[int, Tuple[Tuple[float, float], ...]]]] = {}
        dir_vec = config.direction
        axis = config.axis
        u_axis = config.u_axis
        v_axis = config.v_axis
        for coords, block_id in block_map.items():
            if block_id == 0:
                continue
            nx = coords[0] + dir_vec[0]
            ny = coords[1] + dir_vec[1]
            nz = coords[2] + dir_vec[2]
            if self._has_solid_neighbor(nx, ny, nz, block_map, solid_lookup):
                continue
            plane_coord = coords[axis] + (1 if config.positive else 0)
            u_coord = coords[u_axis]
            v_coord = coords[v_axis]
            uv_set = self._uvs_for_block(block_id)
            planes.setdefault(plane_coord, {})[(u_coord, v_coord)] = (block_id, uv_set)

        step = self._scale
        for plane in sorted(planes.keys()):
            grid = planes[plane]
            used: Set[Tuple[int, int]] = set()
            for (u, v) in sorted(grid.keys()):
                if (u, v) in used:
                    continue
                block_id, uv_base = grid[(u, v)]
                width = 1
                while True:
                    cell = (u + width, v)
                    data = grid.get(cell)
                    if data is None or data[0] != block_id or data[1] != uv_base or cell in used:
                        break
                    width += 1

                height = 1
                while True:
                    row_v = v + height
                    row_ok = True
                    for du in range(width):
                        cell = (u + du, row_v)
                        data = grid.get(cell)
                        if data is None or data[0] != block_id or data[1] != uv_base or cell in used:
                            row_ok = False
                            break
                    if not row_ok:
                        break
                    height += 1

                for du in range(width):
                    for dv in range(height):
                        used.add((u + du, v + dv))

                axis_world = plane * step
                u0 = u * step
                u1 = (u + width) * step
                v0 = v * step
                v1 = (v + height) * step

                verts_world: List[Tuple[float, float, float]] = []
                verts_local: List[Tuple[float, float, float]] = []
                for idx in range(4):
                    coord = [0.0, 0.0, 0.0]
                    coord[axis] = axis_world
                    coord[u_axis] = u0 if config.u_bits[idx] == 0 else u1
                    coord[v_axis] = v0 if config.v_bits[idx] == 0 else v1
                    world = (coord[0], coord[1], coord[2])
                    local = (
                        coord[0] - chunk_origin_world[0],
                        coord[1] - chunk_origin_world[1],
                        coord[2] - chunk_origin_world[2],
                    )
                    verts_world.append(world)
                    verts_local.append(local)

                if self.use_atlas:
                    scaled_uv = uv_base
                else:
                    uv00 = uv_base[config.idx00]
                    uv10 = uv_base[config.idx10]
                    uv01 = uv_base[config.idx01]
                    uv_u_vec = (uv10[0] - uv00[0], uv10[1] - uv00[1])
                    uv_v_vec = (uv01[0] - uv00[0], uv01[1] - uv00[1])
                    scaled_uv_list: List[Tuple[float, float]] = []
                    for ub, vb in zip(config.u_bits, config.v_bits):
                        u_factor = width if ub else 0
                        v_factor = height if vb else 0
                        u_val = uv00[0] + uv_u_vec[0] * u_factor + uv_v_vec[0] * v_factor
                        v_val = uv00[1] + uv_u_vec[1] * u_factor + uv_v_vec[1] * v_factor
                        scaled_uv_list.append((u_val, v_val))
                    scaled_uv = tuple(scaled_uv_list)

                vertex_index = self._write_face(
                    verts_local,
                    verts_world,
                    config.normal,
                    scaled_uv,
                    vwriter,
                    nwriter,
                    twriter,
                    prim,
                    vertex_index,
                    triangles_out,
                )

        return vertex_index

    def _build_render_state(self) -> RenderState:
        attribs = []
        attribs.append(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
        attribs.append(TransparencyAttrib.make(TransparencyAttrib.M_none))
        if self._material is not None:
            attribs.append(MaterialAttrib.make(self._material))
        if self.use_atlas and self._atlas_texture is not None:
            tex_attr = TextureAttrib.make()
            tex_attr = tex_attr.add_on_stage(self._TEXTURE_STAGE, self._atlas_texture)
            attribs.append(tex_attr)
        return RenderState.make(*attribs)

    def _has_solid_neighbor(
        self,
        nx: int,
        ny: int,
        nz: int,
        block_map: Dict[Tuple[int, int, int], int],
        solid_lookup: Optional[Callable[[int, int, int], bool]],
    ) -> bool:
        if (nx, ny, nz) in block_map:
            return True
        if solid_lookup is None:
            return False
        try:
            return bool(solid_lookup(nx, ny, nz))
        except Exception:
            return False

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

    def _is_opaque(self, block_id: int) -> bool:
        block = self.block_registry.get(block_id)
        if block is None:
            return True
        return getattr(block, "opaque", True)
