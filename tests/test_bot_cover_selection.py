from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.bot_ai import AStarBotBrain, BotContext
from game.constants import TEAM_RED, TEAM_BLUE
from world.map_adapter import TacticalGraph, TacticalNode


def _make_map(nodes):
    return SimpleNamespace(
        nav_nodes=list(nodes.values()),
        nav_links=[],
        cube_size=1.0,
        agent_radius=0.5,
        red_base=(0.0, 0.0, 0.0),
        blue_base=(30.0, 0.0, 0.0),
        blocks=[],
    )


def _make_context(team, nodes):
    graph = TacticalGraph(nodes=nodes, links=tuple())
    brain = AStarBotBrain(team, base_pos=(0.0, 0.0, 0.0), enemy_base=(30.0, 0.0, 0.0), nav_graph=graph)
    me = SimpleNamespace(x=0.0, y=0.0, z=0.0, pid=1, team=team, alive=True)
    gs = SimpleNamespace(players={1: me}, flags={})
    mapdata = _make_map(nodes)
    ctx = BotContext(brain=brain, me=me, gs=gs, mapdata=mapdata, nav_graph=graph, now=0.0)
    return brain, ctx


def test_cover_selection_prefers_protective_facing():
    nodes = {
        "safe_cover": TacticalNode(
            node_id="safe_cover",
            pos=(-5.0, 0.0, 0.0),
            kind="cover",
            tags=("cover", "peek", "team:red_area"),
            facing=(1.0, 0.0, 0.0),
            radius=0.6,
        ),
        "bad_cover": TacticalNode(
            node_id="bad_cover",
            pos=(5.0, 0.0, 0.0),
            kind="cover",
            tags=("cover", "peek", "team:red_area"),
            facing=(-1.0, 0.0, 0.0),
            radius=0.6,
        ),
    }

    brain, ctx = _make_context(TEAM_RED, nodes)
    threat = SimpleNamespace(x=12.0, y=0.0, z=0.0, alive=True)

    picked = brain._select_cover_for_threat(ctx, threat)

    assert picked is not None
    assert picked.node_id == "safe_cover"


def test_cover_selection_filters_by_area():
    nodes = {
        "left_cover": TacticalNode(
            node_id="left_cover",
            pos=(0.0, -5.0, 0.0),
            kind="cover",
            tags=("cover", "peek", "team:blue_area"),
            facing=(0.0, 1.0, 0.0),
            radius=0.6,
        ),
        "right_cover": TacticalNode(
            node_id="right_cover",
            pos=(0.0, 5.0, 0.0),
            kind="cover",
            tags=("cover", "peek", "team:red_area"),
            facing=(0.0, -1.0, 0.0),
            radius=0.6,
        ),
    }

    brain, ctx = _make_context(TEAM_RED, nodes)
    threat = SimpleNamespace(x=-12.0, y=0.0, z=0.0, alive=True)

    picked = brain._select_cover_for_threat(ctx, threat)

    assert picked is not None
    # Threat is west of base, so the bot should look for blue-area cover (defensive retreat).
    assert picked.node_id == "left_cover"
