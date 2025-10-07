from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.bot_coordination import SquadCoordinator
from game.constants import TEAM_RED
from world.map_adapter import TacticalGraph, TacticalLink, TacticalNode


class _FakeIndex:
    def __init__(self, nodes):
        self.nodes = nodes

    def nodes_with_tags(self, tags: Iterable[str], area: Optional[str] = None):
        tags = list(tags or [])
        out = []
        for node in self.nodes.values():
            node_tags = set(getattr(node, "tags", ()))
            if tags:
                if any(tag in node_tags for tag in tags):
                    if not area or area in node_tags:
                        out.append(node)
            elif not area or area in node_tags:
                out.append(node)
        return out


def _make_index():
    nodes = {
        "anchor": TacticalNode("anchor", (-8.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), None, 0.6),
        "neutral": TacticalNode("neutral", (0.0, 1.0, 0.0), "cover", ("cover", "team:neutral_area"), None, 0.6),
        "flank": TacticalNode("flank", (8.0, 5.0, 0.0), "cover", ("cover", "team:blue_area"), None, 0.6),
    }
    links = (
        TacticalLink("anchor", "neutral", 8.0, True),
        TacticalLink("neutral", "flank", 9.0, True),
    )
    graph = TacticalGraph(nodes=nodes, links=links)
    index = _FakeIndex(nodes)
    index.graph = graph
    return index


def test_assign_roles_rotates_anchor_entry_flank():
    index = _make_index()
    coordinator = SquadCoordinator(team=TEAM_RED)
    bots = [1, 2, 3]
    coordinator.assign_roles(bots, nav_index=index, enemy_base=(10.0, 0.0, 0.0), friendly_base=(-10.0, 0.0, 0.0))
    snapshot = coordinator.snapshot()

    assert snapshot[str(1)]["target_node"] == "anchor"
    assert snapshot[str(2)]["target_node"] == "neutral"
    assert snapshot[str(3)]["target_node"] == "flank"


def test_assign_roles_reuses_when_more_bots_than_roles():
    index = _make_index()
    coordinator = SquadCoordinator(team=TEAM_RED)
    bots = [1, 2, 3, 4]
    coordinator.assign_roles(bots, nav_index=index, enemy_base=(10.0, 0.0, 0.0), friendly_base=(-10.0, 0.0, 0.0))
    snapshot = coordinator.snapshot()

    assert snapshot[str(4)]["target_node"] == "anchor"


def test_snapshot_empty_without_roles():
    coordinator = SquadCoordinator(team=TEAM_RED)
    assert coordinator.snapshot() == {}
