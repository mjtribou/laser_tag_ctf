from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.bot_ai import AStarBotBrain, BotDecision
from game.constants import TEAM_RED, TEAM_BLUE


def _make_map(blocks):
    return SimpleNamespace(
        blocks=list(blocks),
        red_base=(-10.0, 0.0, 0.0),
        blue_base=(10.0, 0.0, 0.0),
        neutral_flag_stand=(0.0, 0.0, 0.0),
        bounds=(30.0, 30.0),
        cube_size=1.0,
        agent_radius=0.5,
        nav_nodes=[],
        nav_links=[],
    )


def _make_entities():
    me = SimpleNamespace(
        pid=1,
        team=TEAM_RED,
        x=-5.0,
        y=0.0,
        z=0.0,
        yaw_rad=0.0,
        pitch_rad=0.0,
        alive=True,
        carrying_flag=None,
    )
    enemy = SimpleNamespace(
        pid=2,
        team=TEAM_BLUE,
        x=5.0,
        y=0.0,
        z=0.0,
        yaw_rad=0.0,
        pitch_rad=0.0,
        alive=True,
    )
    gs = SimpleNamespace(players={1: me, 2: enemy}, flags={})
    return me, enemy, gs


def test_has_line_of_sight_blocks_wall():
    wall = SimpleNamespace(pos=(0.0, 0.0, 1.5), size=(1.0, 6.0, 3.0))
    mapdata = _make_map([wall])

    brain = AStarBotBrain(TEAM_RED, base_pos=(-10.0, 0.0, 0.0), enemy_base=(10.0, 0.0, 0.0))
    me, enemy, _ = _make_entities()

    assert brain._has_line_of_sight(me, enemy, mapdata) is False


def test_decide_fire_requires_clear_sight():
    wall = SimpleNamespace(pos=(0.0, 0.0, 1.5), size=(1.0, 6.0, 3.0))
    map_blocked = _make_map([wall])
    map_clear = _make_map([])

    brain_blocked = AStarBotBrain(TEAM_RED, base_pos=(-10.0, 0.0, 0.0), enemy_base=(10.0, 0.0, 0.0))
    me_blocked, enemy_blocked, gs_blocked = _make_entities()

    brain_blocked._evaluate_behaviors = lambda ctx: BotDecision("hold", 0.0, (ctx.me.x, ctx.me.y, ctx.me.z))

    inputs_blocked = brain_blocked.decide(me_blocked, gs_blocked, map_blocked)
    assert inputs_blocked["fire"] is False

    brain_clear = AStarBotBrain(TEAM_RED, base_pos=(-10.0, 0.0, 0.0), enemy_base=(10.0, 0.0, 0.0))
    me_clear, enemy_clear, gs_clear = _make_entities()

    brain_clear._evaluate_behaviors = lambda ctx: BotDecision("hold", 0.0, (ctx.me.x, ctx.me.y, ctx.me.z))

    inputs_clear = brain_clear.decide(me_clear, gs_clear, map_clear)
    assert inputs_clear["fire"] is True
