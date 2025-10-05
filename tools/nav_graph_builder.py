"""Generate navigation nodes/links using a 3×3 corner kernel on the voxel map."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game.map_gen import load_from_file
from world.map_adapter import load_map_to_voxels
from world.voxel_grid import VoxelGrid


@dataclass
class SynthNode:
    node_id: str
    pos: Tuple[float, float, float]
    kind: str
    tags: Tuple[str, ...]
    facing: Tuple[float, float, float]
    radius: float


@dataclass
class SynthLink:
    source: str
    target: str
    weight: float
    bidirectional: bool = True


class NavBuilder:
    def __init__(
        self,
        grid: VoxelGrid,
        origin_indices: Tuple[int, int, int],
        cube_size: float,
        agent_radius: float,
        bounds: Sequence[float],
        *,
        link_radius: float = 18.0,
    ) -> None:
        self.grid = grid
        self.origin = origin_indices
        self.cube = float(cube_size)
        self.agent_radius = max(0.1, agent_radius)
        self.bounds_x = float(bounds[0]) if bounds else 0.0
        self.link_radius = max(4.0, link_radius)

        self.corner_offset = max(0.35, self.agent_radius * 0.8)
        self.height_offset = max(0.1, self.agent_radius * 0.2)

        self.nodes: List[SynthNode] = []
        self.links: List[SynthLink] = []
        self._dedupe_keys: set[Tuple[int, int, int, int]] = set()

        self._solid = self._collect_solids()

    def build(self) -> None:
        walkable = self._collect_walkable_cells()
        self._emit_corner_nodes(walkable)
        self._connect_nodes()

    def _collect_solids(self) -> set[Tuple[int, int, int]]:
        solids: set[Tuple[int, int, int]] = set()
        for x in range(self.grid.size_x):
            for y in range(self.grid.size_y):
                for z in range(self.grid.size_z):
                    if not self.grid.is_air(x, y, z):
                        solids.add((x, y, z))
        return solids

    def _is_solid(self, x: int, y: int, z: int) -> bool:
        if not (0 <= x < self.grid.size_x and 0 <= y < self.grid.size_y and 0 <= z < self.grid.size_z):
            return False
        return (x, y, z) in self._solid

    def _collect_walkable_cells(self) -> List[Tuple[int, int, int]]:
        walkable: List[Tuple[int, int, int]] = []
        for x in range(self.grid.size_x):
            for y in range(self.grid.size_y):
                for z in range(1, self.grid.size_z):
                    if not self.grid.is_air(x, y, z):
                        continue
                    if self.grid.is_air(x, y, z - 1):
                        continue
                    if z + 1 < self.grid.size_z and not self.grid.is_air(x, y, z + 1):
                        continue
                    walkable.append((x, y, z))
        return walkable

    def _emit_corner_nodes(self, walkable: Iterable[Tuple[int, int, int]]) -> None:
        for x, y, z in walkable:
            has_n = self._has_wall(x, y + 1, z)
            has_s = self._has_wall(x, y - 1, z)
            has_e = self._has_wall(x + 1, y, z)
            has_w = self._has_wall(x - 1, y, z)

            free_ne = self._corner_open(x + 1, y + 1, z)
            free_nw = self._corner_open(x - 1, y + 1, z)
            free_se = self._corner_open(x + 1, y - 1, z)
            free_sw = self._corner_open(x - 1, y - 1, z)

            if has_n:
                if free_ne:
                    self._create_wall_node(x, y, z, dx=0, dy=1, corner_tag="corner:NE")
                if free_nw:
                    self._create_wall_node(x, y, z, dx=0, dy=1, corner_tag="corner:NW")
            if has_s:
                if free_se:
                    self._create_wall_node(x, y, z, dx=0, dy=-1, corner_tag="corner:SE")
                if free_sw:
                    self._create_wall_node(x, y, z, dx=0, dy=-1, corner_tag="corner:SW")
            if has_e:
                if free_ne:
                    self._create_wall_node(x, y, z, dx=1, dy=0, corner_tag="corner:NE")
                if free_se:
                    self._create_wall_node(x, y, z, dx=1, dy=0, corner_tag="corner:SE")
            if has_w:
                if free_nw:
                    self._create_wall_node(x, y, z, dx=-1, dy=0, corner_tag="corner:NW")
                if free_sw:
                    self._create_wall_node(x, y, z, dx=-1, dy=0, corner_tag="corner:SW")

    def _has_wall(self, x: int, y: int, z: int) -> bool:
        return any(self._is_solid(x, y, z + dz) for dz in (0, 1, 2))

    def _corner_open(self, x: int, y: int, z: int) -> bool:
        return all(not self._is_solid(x, y, z + dz) for dz in (0, 1))

    def _create_wall_node(self, x: int, y: int, z: int, dx: int, dy: int, corner_tag: Optional[str]) -> None:
        dir_idx = (dx + 1) * 3 + (dy + 1)
        key = (x, y, z, dir_idx, corner_tag)
        if key in self._dedupe_keys:
            return

        world_pos = self._wall_world_position(x, y, z, dx, dy)
        if self._point_in_wall(world_pos):
            return

        facing = self._compute_facing_vector(dx, dy)
        tags = self._wall_tags(world_pos[0], dx, dy, corner_tag)

        node = SynthNode(
            node_id=f"corner_{len(self.nodes)}",
            pos=world_pos,
            kind="cover",
            tags=tags,
            facing=facing,
            radius=max(self.agent_radius + 0.1, 0.5),
        )
        self.nodes.append(node)
        self._dedupe_keys.add(key)

    def _wall_world_position(self, x: int, y: int, z: int, dx: int, dy: int) -> Tuple[float, float, float]:
        if dx > 0:
            plane_x = (self.origin[0] + x + 1) * self.cube
            pos_x = plane_x - self.corner_offset
        elif dx < 0:
            plane_x = (self.origin[0] + x) * self.cube
            pos_x = plane_x + self.corner_offset
        else:
            pos_x = (self.origin[0] + x + 0.5) * self.cube

        if dy > 0:
            plane_y = (self.origin[1] + y + 1) * self.cube
            pos_y = plane_y - self.corner_offset
        elif dy < 0:
            plane_y = (self.origin[1] + y) * self.cube
            pos_y = plane_y + self.corner_offset
        else:
            pos_y = (self.origin[1] + y + 0.5) * self.cube

        floor_plane = (self.origin[2] + z) * self.cube
        pos_z = floor_plane + self.height_offset
        return (pos_x, pos_y, pos_z)

    def _point_in_wall(self, world_pos: Tuple[float, float, float]) -> bool:
        gx, gy, gz = self._world_to_grid_float(world_pos)
        ix = int(round(gx))
        iy = int(round(gy))
        iz = max(1, int(round(gz)))
        if iz < self.grid.size_z and self._is_solid(ix, iy, iz):
            return True
        if iz + 1 < self.grid.size_z and self._is_solid(ix, iy, iz + 1):
            return True
        return False

    def _compute_facing_vector(self, dx: int, dy: int) -> Tuple[float, float, float]:
        vx = float(dx)
        vy = float(dy)
        length = math.hypot(vx, vy)
        if length < 1e-6:
            return (0.0, 1.0, 0.0)
        return (vx / length, vy / length, 0.0)

    def _wall_tags(self, world_x: float, dx: int, dy: int, corner_tag: Optional[str]) -> Tuple[str, ...]:
        tags = {"cover", "peek", self._team_tag(world_x)}
        if dx > 0:
            tags.add("side:east")
        elif dx < 0:
            tags.add("side:west")
        if dy > 0:
            tags.add("side:north")
        elif dy < 0:
            tags.add("side:south")
        if corner_tag:
            tags.add(corner_tag)
        return tuple(sorted(tags))

    def _team_tag(self, world_x: float) -> str:
        threshold = max(5.0, 0.15 * self.bounds_x)
        if world_x <= -threshold:
            return "team:red_area"
        if world_x >= threshold:
            return "team:blue_area"
        return "team:neutral_area"

    def _connect_nodes(self) -> None:
        links: List[SynthLink] = []
        for i, a in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                b = self.nodes[j]
                dist = self._distance(a.pos, b.pos)
                if dist > self.link_radius:
                    continue
                if self._segment_intersects_solid(a.pos, b.pos):
                    continue
                links.append(SynthLink(source=a.node_id, target=b.node_id, weight=dist, bidirectional=True))
        self.links = links

    def _segment_intersects_solid(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> bool:
        sx, sy, sz = self._world_to_grid_float(start)
        ex, ey, ez = self._world_to_grid_float(end)

        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        steps = max(abs(dx), abs(dy), abs(dz))
        steps = max(1, int(math.ceil(steps) * 6))

        for step in range(1, steps):
            t = step / steps
            gx = sx + dx * t
            gy = sy + dy * t
            gz = sz + dz * t
            ix = int(round(gx))
            iy = int(round(gy))
            iz = int(round(gz))
            if self._is_solid(ix, iy, iz):
                return True
        return False

    def _world_to_grid_float(self, pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = pos
        ox, oy, oz = self.origin
        gx = (x / self.cube) - ox - 0.5
        gy = (y / self.cube) - oy - 0.5
        gz = (z / self.cube) - oz - 0.5
        return gx, gy, gz

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


def render_topdown(
    builder: NavBuilder,
    nodes: Sequence[SynthNode],
    links: Sequence[SynthLink],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    scale = 12
    width = builder.grid.size_x * scale
    height = builder.grid.size_y * scale
    img = Image.new("RGB", (width, height), (26, 26, 28))
    draw = ImageDraw.Draw(img)

    # Precompute tallest solid for each column for height-based coloring
    column_height: Dict[Tuple[int, int], int] = {}
    max_height = 0
    for x in range(builder.grid.size_x):
        for y in range(builder.grid.size_y):
            top = -1
            for z in range(builder.grid.size_z - 1, -1, -1):
                if builder._is_solid(x, y, z):
                    top = z
                    break
            column_height[(x, y)] = top
            if top > max_height:
                max_height = top

    # Draw solid columns with height-based shading
    for x in range(builder.grid.size_x):
        for y in range(builder.grid.size_y):
            top = column_height[(x, y)]
            if top >= 0:
                t_norm = (top + 1) / max(1, max_height + 1)
                intensity = int(60 + 140 * t_norm)
                color = (intensity, intensity, intensity)
            else:
                color = (32, 32, 36)
            x0 = x * scale
            y0 = height - (y + 1) * scale
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=color)

    # Overlay grid lines
    grid_color = (0, 0, 0)
    for x in range(builder.grid.size_x + 1):
        px = x * scale
        draw.line([(px, 0), (px, height)], fill=grid_color, width=1)
    for y in range(builder.grid.size_y + 1):
        py = height - y * scale
        draw.line([(0, py), (width, py)], fill=grid_color, width=1)

    # Helper to convert a world position into pixel coordinates
    def world_to_pixel(pos: Tuple[float, float, float]) -> Tuple[int, int]:
        gx, gy, _ = builder._world_to_grid_float(pos)
        px = int(round(gx)) * scale + scale // 2
        py = height - (int(round(gy)) * scale + scale // 2)
        return px, py

    node_lookup: Dict[str, SynthNode] = {node.node_id: node for node in nodes}

    # Draw links first (under nodes)
    for link in links:
        a = node_lookup.get(link.source)
        b = node_lookup.get(link.target)
        if not a or not b:
            continue
        ax, ay = world_to_pixel(a.pos)
        bx, by = world_to_pixel(b.pos)
        draw.line([ax, ay, bx, by], fill=(90, 120, 170), width=1)

    # Draw nodes
    node_radius = max(4, scale // 2)
    corner_colors = {
        "corner:NE": (120, 220, 120),
        "corner:SE": (240, 200, 120),
        "corner:SW": (220, 120, 120),
        "corner:NW": (120, 160, 240),
    }
    for node in nodes:
        px, py = world_to_pixel(node.pos)
        r = node_radius
        fill = None
        for corner_tag, color in corner_colors.items():
            if corner_tag in node.tags:
                fill = color
                break
        if fill is None:
            if "team:red_area" in node.tags:
                fill = (220, 90, 90)
            elif "team:blue_area" in node.tags:
                fill = (90, 160, 230)
            else:
                fill = (200, 200, 200)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=fill, outline=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _round_tuple(values: Iterable[float], digits: int = 3) -> List[float]:
    return [round(float(v), digits) for v in values]


def _node_to_dict(node: SynthNode) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "id": node.node_id,
        "pos": _round_tuple(node.pos),
        "type": node.kind,
    }
    if node.tags:
        payload["tags"] = list(node.tags)
    if node.facing is not None:
        payload["facing"] = _round_tuple(node.facing)
    if abs(node.radius - 1.0) > 1e-6:
        payload["radius"] = round(node.radius, 3)
    return payload


def _link_to_dict(link: SynthLink) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "from": link.source,
        "to": link.target,
        "weight": round(link.weight, 3),
    }
    if not link.bidirectional:
        payload["bidirectional"] = False
    return payload


def _compute_origin_indices(mapdata) -> Tuple[int, int, int]:
    cube = float(getattr(mapdata, "cube_size", 1.0) or 1.0)
    min_x = min_y = min_z = None
    for block in getattr(mapdata, "blocks", []) or []:
        ix = int(math.floor(((block.pos[0] - 0.5 * cube) / cube) + 1e-6))
        iy = int(math.floor(((block.pos[1] - 0.5 * cube) / cube) + 1e-6))
        iz = int(math.floor(((block.pos[2] - 0.5 * cube) / cube) + 1e-6))
        min_x = ix if min_x is None else min(min_x, ix)
        min_y = iy if min_y is None else min(min_y, iy)
        min_z = iz if min_z is None else min(min_z, iz)
    if min_x is None:
        return (0, 0, 0)
    return (min_x, min_y, min_z)


def build_nav_graph(
    map_path: Path,
    *,
    include_existing: bool = False,
    link_radius: float = 18.0,
    render_path: Optional[Path] = None,
) -> Dict[str, List[Dict[str, object]]]:
    mapdata = load_from_file(str(map_path))
    grid, _registry, existing_graph = load_map_to_voxels(str(map_path))
    origin = _compute_origin_indices(mapdata)

    builder = NavBuilder(
        grid=grid,
        origin_indices=origin,
        cube_size=getattr(mapdata, "cube_size", 1.0),
        agent_radius=getattr(mapdata, "agent_radius", 0.5) or 0.5,
        bounds=getattr(mapdata, "bounds", (0.0, 0.0)),
        link_radius=link_radius,
    )
    builder.build()

    raw_nodes = list(builder.nodes)
    raw_links = list(builder.links)

    if render_path:
        try:
            render_topdown(builder, raw_nodes, raw_links, Path(render_path))
            print(f"[render] wrote {render_path}")
        except Exception as exc:
            print(f"[render] failed: {exc}")

    nodes = [_node_to_dict(node) for node in raw_nodes]
    links = [_link_to_dict(link) for link in raw_links]

    if include_existing and existing_graph is not None:
        nodes.extend(
            {
                "id": node.node_id,
                "pos": _round_tuple(node.pos),
                "type": node.kind,
                "tags": list(getattr(node, "tags", ())),
                "facing": _round_tuple(getattr(node, "facing", (0.0, 1.0, 0.0))),
                "radius": round(getattr(node, "radius", 1.0), 3),
            }
            for node in existing_graph.nodes.values()
        )
        links.extend(
            {
                "from": getattr(link, "source", getattr(link, "from", "")),
                "to": getattr(link, "target", getattr(link, "to", "")),
                "weight": round(float(getattr(link, "weight", 1.0) or 1.0), 3),
            }
            for link in existing_graph.links
        )

    return {"nodes": nodes, "links": links}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nav graph from voxel corners.")
    parser.add_argument("map", type=Path, help="Path to a map JSON file")
    parser.add_argument("--output", type=Path, help="Optional destination JSON file")
    parser.add_argument(
        "--update-map",
        action="store_true",
        help="Write the map JSON with the synthesized nav graph embedded (requires --output)",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Append any existing nav nodes/links from the source map",
    )
    parser.add_argument(
        "--link-radius",
        type=float,
        default=18.0,
        help="Maximum distance (m) for auto-generated links",
    )
    parser.add_argument(
        "--render",
        type=Path,
        help="Optional PNG path for a top-down visualization",
    )

    args = parser.parse_args()

    nav_graph = build_nav_graph(
        args.map,
        include_existing=args.include_existing,
        link_radius=args.link_radius,
        render_path=args.render,
    )

    if args.output:
        if args.update_map:
            with open(args.map, "r", encoding="utf-8") as handle:
                map_dict = json.load(handle)
            map_dict["nav"] = nav_graph
            args.output.write_text(json.dumps(map_dict, indent=2))
        else:
            args.output.write_text(json.dumps(nav_graph, indent=2))
    else:
        print(json.dumps(nav_graph, indent=2))


if __name__ == "__main__":
    main()
