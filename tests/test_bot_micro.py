from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from types import SimpleNamespace
from game.bot_ai import AStarBotBrain, BotDecision
from game.bot_coordination import SquadRole
from world.map_adapter import TacticalGraph, TacticalNode


def _make_brain(node):
    nodes = {node.node_id: node}
    graph = TacticalGraph(nodes=nodes, links=tuple())
    brain = AStarBotBrain(team=0, base_pos=(0.0, 0.0, 0.0), enemy_base=(10.0, 0.0, 0.0), nav_graph=graph)
    mapdata = SimpleNamespace(
        nav_nodes=list(nodes.values()),
        nav_links=list(graph.links),
        cube_size=1.0,
        agent_radius=0.5,
        blocks=[],
        bounds=(20.0, 20.0),
        red_base=(0.0, 0.0, 0.0),
        blue_base=(10.0, 0.0, 0.0),
    )
    brain._ensure_nav_index(mapdata, brain.nav_graph)
    return brain, mapdata


def _make_context(brain, mapdata, me, enemies=(), now=5.0):
    players = {
        me.pid: SimpleNamespace(pid=me.pid, team=brain.team, x=me.x, y=me.y, z=me.z, alive=True)
    }
    for enemy in enemies:
        players[enemy.pid] = enemy
    gs = SimpleNamespace(players=players, flags={})
    ctx = SimpleNamespace(
        brain=brain,
        me=me,
        gs=gs,
        mapdata=mapdata,
        nav_graph=brain.nav_graph,
        now=now,
        visible_enemies=tuple(enemies),
    )
    return ctx


def _make_bot(pid=1, x=0.0, y=0.0):
    return SimpleNamespace(pid=pid, team=0, x=x, y=y, z=0.0, alive=True, carrying_flag=None, yaw_rad=0.0, pitch_rad=0.0)


def test_micro_hold_state_for_anchor_role():
    node = TacticalNode("anchor", (0.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), (1.0, 0.0, 0.0), 0.6)
    brain, mapdata = _make_brain(node)
    brain.set_squad_role(SquadRole(name="anchor", target_node="anchor", priority=0.8), brain.nav_index)
    brain.last_decision = BotDecision("hold_angle", 65.0, node.pos, metadata={"tactic": "hold"})
    me = _make_bot()
    ctx = _make_context(brain, mapdata, me)

    brain._update_micro_state(ctx)

    assert brain._micro_state == "hold"
    payload = brain.debug_payload(me, ctx.now).get("micro_state")
    assert payload["state"] == "hold"
    assert payload["target"] == [0.0, 0.0]


def test_micro_retreat_when_engaged():
    node = TacticalNode("flank", (6.0, 0.0, 0.0), "cover", ("cover", "team:blue_area"), (1.0, 0.0, 0.0), 0.6)
    brain, mapdata = _make_brain(node)
    brain.set_squad_role(SquadRole(name="flank", target_node="flank", priority=0.9), brain.nav_index)
    brain.last_decision = BotDecision("flank_enemy", 70.0, node.pos, metadata={"tactic": "flank"})
    me = _make_bot()

    enemy = SimpleNamespace(pid=3, team=1, x=4.0, y=0.0, z=0.0, alive=True)
    ctx = _make_context(brain, mapdata, me, enemies=(enemy,))
    brain.mark_suppressed(ctx.now)

    brain._update_micro_state(ctx)

    assert brain._micro_state == "retreat"
    payload = brain.debug_payload(me, ctx.now).get("micro_state")
    assert payload["state"] == "retreat"
    assert payload["retreat_vec"] is not None


def test_micro_peek_holds_position():
    node = TacticalNode("anchor", (0.0, 0.0, 0.0), "cover", ("cover", "team:red_area"), (1.0, 0.0, 0.0), 0.6)
    brain, mapdata = _make_brain(node)
    brain.set_squad_role(SquadRole(name="anchor", target_node="anchor", priority=0.9), brain.nav_index)
    brain.last_decision = BotDecision("hold_angle", 70.0, node.pos, metadata={"tactic": "hold"})
    me = _make_bot()

    enemy = SimpleNamespace(pid=9, team=1, x=4.0, y=0.0, z=0.0, alive=True)
    ctx = _make_context(brain, mapdata, me, enemies=(enemy,))
    brain._update_micro_state(ctx)

    inputs = brain.decide(me, ctx.gs, mapdata)

    assert brain._micro_state == "peek"
    assert abs(inputs["mx"]) < 1e-6
    assert abs(inputs["mz"]) < 1e-6
