from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.bot_ai import AStarBotBrain, BotContext, BotDecision
from game.bot_coordination import SquadRole
from game.constants import TEAM_RED, TEAM_BLUE
from world.map_adapter import TacticalGraph, TacticalLink, TacticalNode


def _build_brain_with_graph(nodes, links, team=TEAM_RED):
    graph = TacticalGraph(nodes=nodes, links=links)
    base_pos = (-10.0, 0.0, 0.0) if team == TEAM_RED else (10.0, 0.0, 0.0)
    enemy_base = (10.0, 0.0, 0.0) if team == TEAM_RED else (-10.0, 0.0, 0.0)
    brain = AStarBotBrain(team, base_pos=base_pos, enemy_base=enemy_base, nav_graph=graph)
    mapdata = SimpleNamespace(
        nav_nodes=list(nodes.values()),
        nav_links=list(links),
        cube_size=1.0,
        agent_radius=0.5,
        blocks=[],
        bounds=(40.0, 40.0),
        red_base=(-10.0, 0.0, 0.0),
        blue_base=(10.0, 0.0, 0.0),
    )
    brain._ensure_nav_index(mapdata, brain.nav_graph)
    return brain, mapdata


def _make_context(brain, mapdata, me, visible_enemies=()):
    if not hasattr(me, "team"):
        setattr(me, "team", brain.team)
    gs = SimpleNamespace(
        players={me.pid: SimpleNamespace(pid=me.pid, team=brain.team, x=me.x, y=me.y, z=me.z, alive=True)},
        flags={}
    )
    ctx = BotContext(
        brain=brain,
        me=me,
        gs=gs,
        mapdata=mapdata,
        nav_graph=brain.nav_graph,
        now=0.0,
        visible_enemies=tuple(visible_enemies),
    )
    return ctx


def test_flank_behavior_prefers_lateral_enemy_cover():
    nodes = {
        "start": TacticalNode("start", (-6.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), None, 0.6),
        "enemy_left": TacticalNode("enemy_left", (9.0, -6.0, 0.0), "cover", ("cover", "team:blue_area"), None, 0.6),
        "enemy_right": TacticalNode("enemy_right", (9.0, 6.0, 0.0), "cover", ("cover", "team:blue_area"), None, 0.6),
    }
    links = (
        TacticalLink("start", "enemy_left", 12.0, True),
        TacticalLink("start", "enemy_right", 12.0, True),
    )
    brain, mapdata = _build_brain_with_graph(nodes, links)

    me = SimpleNamespace(pid=1, x=-6.0, y=0.0, z=0.0, carrying_flag=None)
    enemy = SimpleNamespace(pid=2, team=TEAM_BLUE, x=8.0, y=5.0, z=0.0, alive=True)
    ctx = _make_context(brain, mapdata, me, visible_enemies=(enemy,))

    decision = brain._beh_flank_enemy(ctx)

    assert decision is not None
    assert decision.name == "flank_enemy"
    assert decision.metadata["target_node"] == "enemy_left"
    target = decision.target
    assert target == nodes["enemy_left"].pos


def test_push_behavior_advances_through_neutral_area():
    nodes = {
        "start": TacticalNode("start", (-6.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), None, 0.6),
        "neutral_mid": TacticalNode("neutral_mid", (0.0, 1.0, 0.0), "cover", ("cover", "team:neutral_area"), None, 0.6),
        "neutral_far": TacticalNode("neutral_far", (4.5, -0.5, 0.0), "cover", ("cover", "team:neutral_area"), None, 0.6),
        "enemy_gate": TacticalNode("enemy_gate", (8.0, 0.0, 0.0), "cover", ("cover", "team:blue_area"), None, 0.6),
    }
    links = (
        TacticalLink("start", "neutral_mid", 6.5, True),
        TacticalLink("neutral_mid", "neutral_far", 4.5, True),
        TacticalLink("neutral_far", "enemy_gate", 4.0, True),
    )
    brain, mapdata = _build_brain_with_graph(nodes, links)

    me = SimpleNamespace(pid=3, x=-6.0, y=0.0, z=0.0, carrying_flag=None)
    ctx = _make_context(brain, mapdata, me)

    decision = brain._beh_push_lane(ctx)

    assert decision is not None
    assert decision.name == "push_lane"
    assert decision.metadata["target_node"] == "neutral_far"
    assert decision.metadata["tactic"] == "push"


def test_role_bias_prefers_flank_behavior():
    nodes = {
        "start": TacticalNode("start", (-6.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), None, 0.6),
        "enemy_left": TacticalNode("enemy_left", (9.0, -6.0, 0.0), "cover", ("cover", "team:blue_area"), None, 0.6),
        "enemy_right": TacticalNode("enemy_right", (9.0, 6.0, 0.0), "cover", ("cover", "team:blue_area"), None, 0.6),
        "neutral": TacticalNode("neutral", (0.0, 0.0, 0.0), "cover", ("cover", "team:neutral_area"), None, 0.6),
    }
    links = (
        TacticalLink("start", "neutral", 6.0, True),
        TacticalLink("neutral", "enemy_left", 11.0, True),
        TacticalLink("neutral", "enemy_right", 11.0, True),
    )
    brain, mapdata = _build_brain_with_graph(nodes, links)

    # ensure nav index ready
    brain._ensure_nav_index(mapdata, brain.nav_graph)
    role = SquadRole(name="flank", target_node="enemy_right", priority=0.9)
    brain.set_squad_role(role, brain.nav_index)

    me = SimpleNamespace(pid=5, team=brain.team, x=-6.0, y=0.0, z=0.0, carrying_flag=None)
    enemy = SimpleNamespace(pid=6, team=TEAM_BLUE, x=8.0, y=-5.0, z=0.0, alive=True)
    ctx = _make_context(brain, mapdata, me, visible_enemies=(enemy,))

    decision = brain._evaluate_behaviors(ctx)

    assert decision.metadata["tactic"].startswith("flank")


def test_hold_angle_behavior_for_anchor_role():
    nodes = {
        "anchor": TacticalNode("anchor", (-2.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), (1.0, 0.0, 0.0), 0.6),
    }
    links: Tuple[TacticalLink, ...] = ()
    brain, mapdata = _build_brain_with_graph(nodes, links)
    brain._ensure_nav_index(mapdata, brain.nav_graph)
    role = SquadRole(name="anchor", target_node="anchor", priority=0.75)
    brain.set_squad_role(role, brain.nav_index)

    me = SimpleNamespace(pid=7, team=brain.team, x=-2.5, y=0.1, z=0.0, carrying_flag=None)
    ctx = _make_context(brain, mapdata, me, visible_enemies=())

    decision = brain._beh_hold_angle(ctx)

    assert decision is not None
    assert decision.name == "hold_angle"
    assert decision.metadata["tactic"] == "hold"
    assert decision.crouch is True
    assert decision.walk is False
    assert decision.target == nodes["anchor"].pos
    assert decision.focus is not None


def test_debug_payload_includes_plan_and_metadata():
    nodes = {
        "start": TacticalNode("start", (-2.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), None, 0.6),
    }
    links: Tuple[TacticalLink, ...] = ()
    brain, _mapdata = _build_brain_with_graph(nodes, links)

    brain.last_decision = BotDecision("flank_enemy", 60.0, (5.0, 0.0, 0.0), metadata={"tactic": "flank"})
    brain._path = [(0.0, 0.0), (5.0, 0.0)]
    brain._path_i = 0
    brain._debug_plan = {
        "goal": (5.0, 0.0),
        "path": brain._path.copy(),
        "used_graph": True,
        "decision": "flank_enemy",
    }

    me = SimpleNamespace(pid=4, x=0.0, y=0.0, z=0.0, carrying_flag=None)
    payload = brain.debug_payload(me, now=0.0)

    assert payload["decision_meta"]["tactic"] == "flank"
    assert payload["plan"]["decision"] == "flank_enemy"
    assert payload["plan"]["path"] == [[0.0, 0.0], [5.0, 0.0]]
