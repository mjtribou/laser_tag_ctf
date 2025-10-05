from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.bot_ai import AStarBotBrain
from game.constants import TEAM_RED
from world.map_adapter import TacticalGraph, TacticalNode, TacticalLink


def _make_map(nodes, links):
    return SimpleNamespace(
        nav_nodes=list(nodes.values()),
        nav_links=list(links),
        cube_size=1.0,
        agent_radius=0.5,
        red_base=(0.0, 0.0, 0.0),
        blue_base=(20.0, 0.0, 0.0),
        blocks=[],
    )


def test_graph_path_used_for_planning():
    nodes = {
        "start": TacticalNode("start", (0.0, 0.0, 0.0), "cover", ("cover",), None, 0.6),
        "mid": TacticalNode("mid", (5.0, 0.0, 0.0), "cover", ("cover",), None, 0.6),
        "end": TacticalNode("end", (10.0, 0.0, 0.0), "cover", ("cover",), None, 0.6),
    }
    links = (
        TacticalLink("start", "mid", 5.0, True),
        TacticalLink("mid", "end", 5.0, True),
    )
    graph = TacticalGraph(nodes=nodes, links=links)
    brain = AStarBotBrain(TEAM_RED, base_pos=(0.0, 0.0, 0.0), enemy_base=(10.0, 0.0, 0.0), nav_graph=graph)
    mapdata = _make_map(nodes, links)

    brain._plan(mapdata, (0.1, 0.0), (10.0, 0.0))

    assert brain._path[0] == (0.0, 0.0)
    assert brain._path[-1] == (10.0, 0.0)
    assert (5.0, 0.0) in brain._path


def test_exposed_nodes_penalized():
    nodes = {
        "start": TacticalNode("start", (0.0, 0.0, 0.0), "cover", ("cover",), None, 0.6),
        "unsafe": TacticalNode("unsafe", (4.0, 0.0, 0.0), "generic", (), None, 0.6),
        "end": TacticalNode("end", (8.0, 0.0, 0.0), "cover", ("cover",), None, 0.6),
        "direct": TacticalNode("direct", (8.5, 0.0, 0.0), "cover", ("cover",), None, 0.6),
    }
    links = (
        TacticalLink("start", "unsafe", 4.0, True),
        TacticalLink("unsafe", "end", 4.0, True),
        TacticalLink("start", "direct", 8.5, True),
        TacticalLink("direct", "end", 0.5, True),
    )
    graph = TacticalGraph(nodes=nodes, links=links)
    brain = AStarBotBrain(TEAM_RED, base_pos=(0.0, 0.0, 0.0), enemy_base=(8.0, 0.0, 0.0), nav_graph=graph)
    mapdata = _make_map(nodes, links)

    brain._plan(mapdata, (0.0, 0.0), (8.0, 0.0))

    # The chosen path should go through the direct cover node rather than the exposed "unsafe" node.
    assert (8.5, 0.0) in brain._path
    assert (4.0, 0.0) not in brain._path
