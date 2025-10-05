"""Utility to synthesize navigation nodes/links for block-based maps.

Usage:
    python tools/nav_graph_builder.py configs/maps/ai_test.json --output nav_auto.json

The tool analyzes axis-aligned block obstacles in the map and places cover nodes
around each qualifying block. It emits nav nodes and links compatible with the
`game.map_gen` TacticalGraph schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

@dataclass
class Block:
    """Simple wrapper to make obstacle calculations easier."""

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    box_type: int

    @property
    def half_extents(self) -> Tuple[float, float, float]:
        return (self.size[0] * 0.5, self.size[1] * 0.5, self.size[2] * 0.5)

    @property
    def base_z(self) -> float:
        return self.center[2] - self.half_extents[2]


def _is_floor(block: Block, *, floor_height: float = 1.05) -> bool:
    return block.size[2] <= floor_height


def _point_inside_block(pos: Tuple[float, float, float], block: Block, padding: float = 0.0) -> bool:
    px, py, pz = pos
    cx, cy, cz = block.center
    hx, hy, hz = block.half_extents
    return (
        abs(px - cx) <= hx + padding
        and abs(py - cy) <= hy + padding
        and abs(pz - cz) <= hz + padding
    )


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
        agent_radius: float,
        bounds: Sequence[float],
        *,
        cover_offset: Optional[float] = None,
        corner_gap: Optional[float] = None,
        link_radius: float = 18.0,
    ) -> None:
        self.agent_radius = max(0.1, agent_radius)
        self.cover_offset = cover_offset if cover_offset is not None else self.agent_radius + 0.75
        self.corner_gap = corner_gap if corner_gap is not None else max(0.75, self.agent_radius * 0.9)
        self.link_radius = max(4.0, link_radius)
        self.bounds_x = float(bounds[0]) if bounds else 0.0

        self.blocks: List[Block] = []
        self.nodes: List[SynthNode] = []
        self.links: List[SynthLink] = []

    def add_block(self, block: Block) -> None:
        self.blocks.append(block)

    # --- node synthesis -------------------------------------------------

    def build(self) -> None:
        ring_registry: Dict[int, List[SynthNode]] = {}
        for index, block in enumerate(self.blocks):
            if _is_floor(block) or block.size[2] < 1.5:
                continue
            hx, hy, hz = block.half_extents
            if hx < 1.0 or hy < 1.0:
                continue

            ring_nodes = self._build_nodes_for_block(index, block)
            if not ring_nodes:
                continue
            ring_registry[index] = ring_nodes
            self.nodes.extend(ring_nodes)

        for block_index, ring_nodes in ring_registry.items():
            self._connect_ring_neighbors(ring_nodes)

        self._connect_proximity_links(self.nodes)

    # --- helpers --------------------------------------------------------

    def _segment_block_intersection(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        *,
        margin: float = 0.1,
    ) -> bool:
        """Check if the segment from start→end intersects any non-floor block."""

        sx, sy, sz = start
        ex, ey, ez = end
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz

        for block in self.blocks:
            if _is_floor(block):
                continue
            cx, cy, cz = block.center
            hx, hy, hz = block.half_extents
            hx += margin
            hy += margin
            hz += margin

            tx_min, tx_max = self._axis_interval(sx, dx, cx, hx)
            ty_min, ty_max = self._axis_interval(sy, dy, cy, hy)
            tz_min, tz_max = self._axis_interval(sz, dz, cz, hz)

            t_enter = max(tx_min, ty_min, tz_min)
            t_exit = min(tx_max, ty_max, tz_max)

            if t_enter <= t_exit and t_exit >= 0.0 and t_enter <= 1.0:
                # Ignore the case where the only overlap is due to an endpoint resting on the surface.
                if t_enter > 1e-4 or t_exit < 1.0 - 1e-4:
                    return True
        return False

    @staticmethod
    def _axis_interval(s: float, ds: float, center: float, half_extent: float) -> Tuple[float, float]:
        if abs(ds) < 1e-9:
            if abs(s - center) <= half_extent:
                return (0.0, 1.0)
            return (float("inf"), float("-inf"))
        inv = 1.0 / ds
        t1 = (center - half_extent - s) * inv
        t2 = (center + half_extent - s) * inv
        return (min(t1, t2), max(t1, t2))

    def _build_nodes_for_block(self, index: int, block: Block) -> List[SynthNode]:
        cx, cy, cz = block.center
        hx, hy, hz = block.half_extents
        base_z = block.base_z

        gap = min(self.corner_gap, max(0.25, min(hx, hy) - 0.25))
        z_pos = base_z

        team_tag = self._team_tag_for_block(block)
        block_tag = f"block:{index}"

        result: List[SynthNode] = []

        def make_node(node_id: str, x: float, y: float, facing: Tuple[float, float, float], side_tag: str, corner_tag: str) -> None:
            pos = (x, y, z_pos)
            if self._collides_with_any_block(pos):
                return
            fx, fy, fz = facing
            length = math.sqrt(fx * fx + fy * fy + fz * fz)
            if length < 1e-6:
                facing_vec = (0.0, 1.0, 0.0)
            else:
                inv = 1.0 / length
                facing_vec = (fx * inv, fy * inv, fz * inv)

            tags = ["cover", "peek", block_tag, side_tag, corner_tag, team_tag]
            node = SynthNode(
                node_id=node_id,
                pos=pos,
                kind="cover",
                tags=tuple(tags),
                facing=facing_vec,
                radius=max(self.agent_radius + 0.1, 0.5),
            )
            result.append(node)

        # Side builders: east, west, north, south
        side_specs = [
            {
                "name": "east",
                "x": cx + hx + self.cover_offset,
                "y_offsets": self._side_offsets(hy, gap),
                "facing": [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
                "corner_tags": ["corner:NE", "corner:SE"],
            },
            {
                "name": "west",
                "x": cx - hx - self.cover_offset,
                "y_offsets": self._side_offsets(hy, gap),
                "facing": [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
                "corner_tags": ["corner:NW", "corner:SW"],
            },
        ]

        north_offsets = self._side_offsets(hx, gap)
        south_offsets = self._side_offsets(hx, gap)

        side_specs.extend(
            [
                {
                    "name": "north",
                    "y": cy + hy + self.cover_offset,
                    "x_offsets": north_offsets,
                    "facing": [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
                    "corner_tags": ["corner:NE", "corner:NW"],
                },
                {
                    "name": "south",
                    "y": cy - hy - self.cover_offset,
                    "x_offsets": south_offsets,
                    "facing": [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
                    "corner_tags": ["corner:SE", "corner:SW"],
                },
            ]
        )

        counter = 0
        for spec in side_specs:
            side_name = spec["name"]
            side_tag = f"side:{side_name}"
            if side_name in ("east", "west"):
                x = spec["x"]
                offsets = spec["y_offsets"]
                facings = spec["facing"]
                corner_tags = spec["corner_tags"]
                for idx, offset in enumerate(offsets):
                    y = cy + offset
                    facing = facings[min(idx, len(facings) - 1)]
                    corner_tag = corner_tags[min(idx, len(corner_tags) - 1)]
                    node_id = f"b{index}_{side_name}_{counter}"
                    counter += 1
                    make_node(node_id, x, y, facing, side_tag, corner_tag)
            else:
                y = spec["y"]
                offsets = spec["x_offsets"]
                facings = spec["facing"]
                corner_tags = spec["corner_tags"]
                for idx, offset in enumerate(offsets):
                    x = cx + offset
                    facing = facings[min(idx, len(facings) - 1)]
                    corner_tag = corner_tags[min(idx, len(corner_tags) - 1)]
                    node_id = f"b{index}_{side_name}_{counter}"
                    counter += 1
                    make_node(node_id, x, y, facing, side_tag, corner_tag)

        return result

    def _side_offsets(self, half_extent: float, gap: float) -> List[float]:
        if half_extent <= gap:
            return [0.0]
        offset = max(0.0, half_extent - gap)
        if offset <= 1e-6:
            return [0.0]
        return [offset, -offset]

    def _collides_with_any_block(self, pos: Tuple[float, float, float]) -> bool:
        for block in self.blocks:
            if _is_floor(block):
                continue
            if _point_inside_block(pos, block, padding=0.05):
                return True
        return False

    def _team_tag_for_block(self, block: Block) -> str:
        cx = block.center[0]
        threshold = max(5.0, 0.15 * self.bounds_x)
        if cx <= -threshold:
            return "team:red_area"
        if cx >= threshold:
            return "team:blue_area"
        return "team:neutral_area"

    # --- link synthesis -------------------------------------------------

    def _connect_ring_neighbors(self, nodes: Sequence[SynthNode]) -> None:
        if len(nodes) < 2:
            return
        count = len(nodes)
        for i in range(count):
            a = nodes[i]
            b = nodes[(i + 1) % count]
            weight = self._distance(a.pos, b.pos)
            if self._segment_block_intersection(a.pos, b.pos):
                continue
            self.links.append(SynthLink(source=a.node_id, target=b.node_id, weight=weight, bidirectional=True))

    def _connect_proximity_links(self, nodes: Sequence[SynthNode]) -> None:
        link_keys: set[Tuple[str, str]] = set(
            tuple(sorted((link.source, link.target))) for link in self.links
        )
        for i in range(len(nodes)):
            a = nodes[i]
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                key = tuple(sorted((a.node_id, b.node_id)))
                if key in link_keys:
                    continue
                dist = self._distance(a.pos, b.pos)
                if dist > self.link_radius:
                    continue
                if self._segment_block_intersection(a.pos, b.pos):
                    continue
                link_keys.add(key)
                self.links.append(SynthLink(source=a.node_id, target=b.node_id, weight=dist, bidirectional=True))

    @staticmethod
    def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


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


def build_nav_graph(map_path: Path, *, include_existing: bool = False, link_radius: float = 18.0) -> Dict[str, List[Dict[str, object]]]:
    with open(map_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    bounds = data.get("bounds", (0.0, 0.0))
    agent_radius = float(data.get("agent_radius", 0.5) or 0.5)

    builder = NavBuilder(
        agent_radius=agent_radius,
        bounds=bounds,
        link_radius=link_radius,
    )

    for raw in data.get("blocks", []):
        block = Block(center=tuple(raw.get("pos", (0.0, 0.0, 0.0))), size=tuple(raw.get("size", (1.0, 1.0, 1.0))), box_type=int(raw.get("box_type", 0)))
        builder.add_block(block)

    builder.build()

    nodes = [_node_to_dict(node) for node in builder.nodes]
    links = [_link_to_dict(link) for link in builder.links]

    if include_existing:
        nav = data.get("nav", {}) or {}
        nodes.extend(nav.get("nodes", []))
        links.extend(nav.get("links", []))

    return {"nodes": nodes, "links": links}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize nav graph nodes around map obstacles.")
    parser.add_argument("map", type=Path, help="Path to a map JSON file")
    parser.add_argument("--output", type=Path, help="Optional path to write a JSON file with the nav graph")
    parser.add_argument(
        "--update-map",
        action="store_true",
        help="Write a full map JSON with the synthesized nav graph embedded (requires --output)",
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
        help="Maximum distance (m) for auto-generated inter-block links",
    )

    args = parser.parse_args()

    nav_graph = build_nav_graph(
        args.map,
        include_existing=args.include_existing,
        link_radius=args.link_radius,
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
