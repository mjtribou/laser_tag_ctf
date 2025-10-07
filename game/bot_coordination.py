"""Group-level bot coordination: assign roles and synchronized objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from game.constants import TEAM_RED, TEAM_BLUE


@dataclass
class SquadRole:
    name: str
    target_node: Optional[str]
    priority: float
    metadata: Dict[str, float] = field(default_factory=dict)


class SquadCoordinator:
    """Per-map, per-team coordinator that keeps bots in complementary roles."""

    def __init__(self, *, team: int) -> None:
        self.team = team
        self.roles: Dict[int, SquadRole] = {}
        self._revision = 0

    def assign_roles(
        self,
        bots: Iterable[int],
        *,
        nav_index,
        enemy_base: Tuple[float, float, float],
        friendly_base: Tuple[float, float, float],
    ) -> None:
        """Assign anchor/entry/flank roles based on the current nav graph."""

        bot_ids = list(sorted(bots))
        if not bot_ids or nav_index is None:
            return

        anchor_tag = "team:red_area" if self.team == TEAM_RED else "team:blue_area"
        neutral_tag = "team:neutral_area"
        enemy_tag = "team:blue_area" if self.team == TEAM_RED else "team:red_area"

        anchor_nodes = nav_index.nodes_with_tags(["cover"], area=anchor_tag)
        neutral_nodes = nav_index.nodes_with_tags(["cover"], area=neutral_tag)
        enemy_nodes = nav_index.nodes_with_tags(["cover"], area=enemy_tag)

        # Fallbacks
        if not anchor_nodes:
            anchor_nodes = nav_index.nodes_with_tags([], area=anchor_tag)
        if not neutral_nodes:
            neutral_nodes = nav_index.nodes_with_tags([], area=neutral_tag)
        if not enemy_nodes:
            enemy_nodes = nav_index.nodes_with_tags([], area=enemy_tag)

        def _closest_node(nodes, reference: Tuple[float, float]) -> Optional[str]:
            best = None
            best_d = float("inf")
            for node in nodes:
                nx, ny, _ = node.pos
                d = math.hypot(nx - reference[0], ny - reference[1])
                if d < best_d:
                    best_d = d
                    best = node.node_id
            return best

        enemy_ref = (enemy_base[0], enemy_base[1])
        friendly_ref = (friendly_base[0], friendly_base[1])
        mid_ref = ((enemy_base[0] + friendly_base[0]) * 0.5, (enemy_base[1] + friendly_base[1]) * 0.5)

        anchor_node = _closest_node(anchor_nodes, friendly_ref) if anchor_nodes else None
        entry_node = _closest_node(neutral_nodes, mid_ref) if neutral_nodes else None
        flank_node = _closest_node(enemy_nodes, enemy_ref) if enemy_nodes else None

        assignments: List[Tuple[str, Optional[str], float]] = []
        if anchor_node:
            assignments.append(("anchor", anchor_node, 1.0))
        if entry_node:
            assignments.append(("entry", entry_node, 0.8))
        if flank_node:
            assignments.append(("flank", flank_node, 0.7))

        if not assignments:
            return

        for idx, bot_id in enumerate(bot_ids):
            role_name, node_id, priority = assignments[idx % len(assignments)]
            self.roles[bot_id] = SquadRole(role_name, node_id, priority)

        self._revision += 1

    def get_role(self, bot_id: int) -> Optional[SquadRole]:
        return self.roles.get(bot_id)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        for bot_id, role in self.roles.items():
            role_payload: Dict[str, float] = {
                "priority": round(role.priority, 3),
                "revision": float(self._revision),
            }
            if role.target_node:
                role_payload["target_node"] = role.target_node
            payload[str(bot_id)] = role_payload
        return payload
