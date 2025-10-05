# game/bot_ai.py
import math, random, time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, NamedTuple, Iterable

from .constants import TEAM_RED, TEAM_BLUE
from . import nav_grid as ng
from world.map_adapter import TacticalGraph, TacticalNode

def _wrap_pi(a: float) -> float:
    return ((a + math.pi) % (2 * math.pi)) - math.pi

def _turn_toward(curr: float, target: float, rate_rad_per_s: float, dt: float) -> float:
    diff = ((target - curr + math.pi) % (2 * math.pi)) - math.pi
    maxstep = rate_rad_per_s * dt
    if diff > maxstep:  diff = maxstep
    if diff < -maxstep: diff = -maxstep
    return curr + diff


class SimpleBotBrain:
    """
    Extremely simple bot AI (kept for fallback/regression checks).
    """
    def __init__(self, team: int, base_pos: Tuple[float,float,float], enemy_base: Tuple[float,float,float]):
        self.team = team
        self.base_pos = base_pos
        self.enemy_base = enemy_base
        self.state = "patrol"
        self.target = None
        self.last_repath = 0.0

    def decide(self, me, gs, mapdata):
        now = time.time()
        inputs = {"mx":0.0,"mz":0.0,"jump":False,"crouch":False,"walk":False,"fire":False,"interact":False,
                  "yaw": math.degrees(getattr(me, "yaw_rad", 0.0)),
                  "pitch": math.degrees(getattr(me, "pitch_rad", 0.0))}
        # priorities
        if me.carrying_flag is not None:
            goal = self.base_pos
        else:
            if self.state == "patrol" or self.target is None or (now - self.last_repath) > 5.0:
                self.target = random.choice([
                    self.enemy_base,
                    (0.0, 0.0, 0.0),
                    (self.enemy_base[0]*0.6, self.enemy_base[1]*0.6, 0.0)
                ])
                self.last_repath = now
            goal = self.target

        dx = goal[0]-me.x; dy = goal[1]-me.y
        if abs(dx)+abs(dy) > 0.5:
            desired_yaw = math.atan2(-dx, dy)
            curr = getattr(me, "yaw_rad", 0.0)
            me.yaw_rad = _turn_toward(curr, desired_yaw, rate_rad_per_s=math.radians(120), dt=0.016)
            inputs["yaw"] = math.degrees(me.yaw_rad)

        inputs["mx"] = 0.0
        inputs["mz"] = 1.0
        inputs["fire"] = random.random() < 0.05
        inputs["walk"] = False
        inputs["crouch"] = random.random() < 0.02
        return inputs


# --- Shared nav-cache across all brains (one grid per map/settings) ---
_FIELD_CACHE = {}  # key: (nav_key, red_xy, blue_xy) -> {"red": (dist,parent), "blue": (dist,parent)}
_NAV_CACHE: Dict[Tuple, ng.NavGrid] = {}


class BotIntent(NamedTuple):
    kind: str
    origin_pid: int
    position: Tuple[float, float, float]
    expires_at: float


class BotRadio:
    def __init__(self):
        self._by_team: Dict[int, List[BotIntent]] = {TEAM_RED: [], TEAM_BLUE: []}

    def broadcast(self, team: int, intent: BotIntent) -> None:
        bucket = self._by_team.setdefault(team, [])
        bucket.append(intent)

    def listen(self, team: int, now: float) -> Tuple[BotIntent, ...]:
        bucket = self._by_team.setdefault(team, [])
        active = [intent for intent in bucket if intent.expires_at > now]
        self._by_team[team] = active
        return tuple(active)


RADIO = BotRadio()


@dataclass
class BotContext:
    brain: "AStarBotBrain"
    me: any
    gs: any
    mapdata: any
    nav_graph: Optional[TacticalGraph]
    now: float

    @property
    def team(self) -> int:
        return self.brain.team

    @property
    def base_pos(self) -> Tuple[float, float, float]:
        return self.brain.base_pos

    @property
    def enemy_base(self) -> Tuple[float, float, float]:
        return self.brain.enemy_base

    def nav_nodes(self) -> Iterable[TacticalNode]:
        if self.nav_graph is None:
            nodes = getattr(self.mapdata, "nav_nodes", [])
            return nodes
        return self.nav_graph.nodes.values()

    def nodes_with_tags(self, required: Iterable[str]) -> List[TacticalNode]:
        tags = set(required)
        if not tags:
            return list(self.nav_nodes())
        out = []
        for node in self.nav_nodes():
            node_tags = set(getattr(node, "tags", ()))
            if tags.issubset(node_tags):
                out.append(node)
        return out

    def teammates(self) -> Iterable[any]:
        for pid, player in self.gs.players.items():
            if player.team == self.team and pid != self.me.pid:
                yield player

    def enemies(self) -> Iterable[any]:
        for player in self.gs.players.values():
            if player.team != self.team:
                yield player

    def flag_for_team(self, team: int):
        return self.gs.flags.get(team)


@dataclass
class BotDecision:
    name: str
    score: float
    target: Tuple[float, float, float]
    crouch: bool = False
    walk: bool = False
    broadcast: Optional[BotIntent] = None
    focus: Optional[Tuple[float, float, float]] = None


def _nav_key(mapdata, cell, agent_radius):
    return (
        id(mapdata),
        len(getattr(mapdata, "blocks", [])),
        getattr(mapdata, "bounds", None),
        round(float(cell), 3),
        round(float(agent_radius), 3),
    )

class AStarBotBrain:
    """Utility-driven tactical bot with grid navigation."""

    def __init__(
        self,
        team: int,
        base_pos: Tuple[float, float, float],
        enemy_base: Tuple[float, float, float],
        *,
        target_players: bool = True,
        nav_graph: Optional[TacticalGraph] = None,
        radio: BotRadio = RADIO,
    ) -> None:
        self.team = team
        self.base_pos = base_pos
        self.enemy_base = enemy_base
        self.target_players = bool(target_players)
        self.nav_graph = nav_graph
        self.radio = radio

        # Navigation state
        self._nav: Optional[ng.NavGrid] = None
        self._fields = None
        self._path: List[Tuple[float, float]] = []
        self._path_i: int = 0
        self._last_goal_xy: Optional[Tuple[float, float]] = None
        self._next_plan_t: float = 0.0
        self._last_progress_t: float = 0.0
        self._last_progress_dist: float = float("inf")

    # --- Nav helpers -----------------------------------------------------
    def _ensure_nav(self, mapdata):
        if self._nav is None:
            cell = getattr(mapdata, "cube_size", 1.0)
            radius = getattr(mapdata, "agent_radius", 0.5)
            key = _nav_key(mapdata, cell, radius)
            nav = _NAV_CACHE.get(key)
            if nav is None:
                nav = ng.build_navgrid(mapdata, cell=cell, agent_radius=radius)
                _NAV_CACHE[key] = nav
            self._nav = nav

    def _ensure_fields(self, mapdata):
        self._ensure_nav(mapdata)
        cell = getattr(mapdata, "cube_size", 1.0)
        radius = getattr(mapdata, "agent_radius", 0.5)
        key_nav = _nav_key(mapdata, cell, radius)
        red_xy = (round(mapdata.red_base[0], 3), round(mapdata.red_base[1], 3))
        blue_xy = (round(mapdata.blue_base[0], 3), round(mapdata.blue_base[1], 3))
        fkey = (key_nav, red_xy, blue_xy)
        fields = _FIELD_CACHE.get(fkey)
        if fields is None:
            red = ng.dijkstra_field(self._nav, [red_xy])
            blue = ng.dijkstra_field(self._nav, [blue_xy])
            fields = {"red": red, "blue": blue}
            _FIELD_CACHE[fkey] = fields
        self._fields = fields

    def _need_replan(self, goal_xy: Tuple[float, float]) -> bool:
        if not self._path or self._last_goal_xy is None:
            return True
        gx, gy = goal_xy
        lgx, lgy = self._last_goal_xy
        if (abs(gx - lgx) + abs(gy - lgy)) > 1.0:
            return True
        return time.time() >= self._next_plan_t

    def _plan(self, me_xy: Tuple[float, float], goal_xy: Tuple[float, float]):
        now_t = time.time()
        self._last_goal_xy = goal_xy
        sxy = ng.nearest_passable_xy(self._nav, me_xy[0], me_xy[1], max_radius=8)
        gxy = ng.nearest_passable_xy(self._nav, goal_xy[0], goal_xy[1], max_radius=12)
        self._path = ng.astar_bounded(self._nav, sxy, gxy, w=1.25, pad=12, max_iter=20000)
        self._path_i = 0
        self._last_progress_t = now_t
        self._last_progress_dist = float("inf")
        self._next_plan_t = now_t + random.uniform(1.0, 2.0)

    def _advance_waypoint_if_close(self, me_xy: Tuple[float, float], threshold: float = 0.6):
        if not self._path or self._path_i >= len(self._path):
            return
        tx, ty = self._path[self._path_i]
        if math.hypot(me_xy[0] - tx, me_xy[1] - ty) <= threshold:
            self._path_i += 1
            self._last_progress_t = time.time()

    def _current_target(self) -> Optional[Tuple[float, float]]:
        if not self._path or self._path_i >= len(self._path):
            return None
        j = min(self._path_i + 1, len(self._path) - 1)
        return self._path[j]

    def _stalled(self, me_xy: Tuple[float, float]) -> bool:
        tgt = self._current_target()
        if tgt is None:
            return False
        dist = math.hypot(me_xy[0] - tgt[0], me_xy[1] - tgt[1])
        improved = dist < (self._last_progress_dist - 0.2)
        now_t = time.time()
        if improved:
            self._last_progress_dist = dist
            self._last_progress_t = now_t
            return False
        return (now_t - self._last_progress_t) > 1.7 and dist > 0.9

    def _flag_interact_needed(self, me, gs) -> bool:
        for flag in gs.flags.values():
            if math.hypot(me.x - flag.x, me.y - flag.y) <= 1.6:
                return True
        return False

    def _nearest_enemy(self, me, gs, max_range: float = 45.0, require_line: bool = False):
        best = None
        best_d2 = max_range * max_range
        for enemy in gs.players.values():
            if enemy.team == me.team or not enemy.alive:
                continue
            try:
                if not self.target_players and not getattr(enemy, "is_bot", False):
                    continue
            except Exception:
                pass
            dx = enemy.x - me.x
            dy = enemy.y - me.y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = enemy
        return best

    # --- Behavior evaluation --------------------------------------------

    def _evaluate_behaviors(self, ctx: BotContext) -> BotDecision:
        candidates: List[BotDecision] = []
        for behavior in (
            self._beh_return_flag,
            self._beh_escort_call,
            self._beh_retrieve_flag,
            self._beh_defend_base,
            self._beh_hunt_enemy,
            self._beh_attack_enemy_base,
        ):
            decision = behavior(ctx)
            if decision is not None:
                candidates.append(decision)

        if not candidates:
            return BotDecision("idle_patrol", 0.0, self.enemy_base)

        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates[0]

    def _beh_return_flag(self, ctx: BotContext) -> Optional[BotDecision]:
        if ctx.me.carrying_flag is None:
            return None
        intent = BotIntent("escort_me", ctx.me.pid, (ctx.me.x, ctx.me.y, ctx.me.z), ctx.now + 2.5)
        return BotDecision("return_flag", 100.0, ctx.base_pos, broadcast=intent)

    def _beh_escort_call(self, ctx: BotContext) -> Optional[BotDecision]:
        intents = self.radio.listen(ctx.team, ctx.now)
        best = None
        best_dist = float("inf")
        for intent in intents:
            if intent.origin_pid == ctx.me.pid:
                continue
            d = math.hypot(ctx.me.x - intent.position[0], ctx.me.y - intent.position[1])
            if d < best_dist:
                best = intent
                best_dist = d
        if best is None:
            return None
        score = max(60.0 - best_dist, 15.0)
        return BotDecision("escort_request", score, best.position, walk=False)

    def _beh_retrieve_flag(self, ctx: BotContext) -> Optional[BotDecision]:
        enemy_flag_team = TEAM_RED if ctx.team == TEAM_BLUE else TEAM_BLUE
        enemy_flag = ctx.flag_for_team(enemy_flag_team)
        if enemy_flag is None:
            return None
        if enemy_flag.carried_by is not None and enemy_flag.carried_by == ctx.me.pid:
            return None
        if enemy_flag.at_base and enemy_flag.team != TEAM_NEUTRAL:
            return None
        score = 55.0
        target = (enemy_flag.x, enemy_flag.y, enemy_flag.z)
        return BotDecision("retrieve_flag", score, target)

    def _beh_defend_base(self, ctx: BotContext) -> Optional[BotDecision]:
        threat = None
        threat_dist = float("inf")
        bx, by, _ = ctx.base_pos
        for enemy in ctx.enemies():
            if not enemy.alive:
                continue
            dist = math.hypot(enemy.x - bx, enemy.y - by)
            if dist < 18.0 and dist < threat_dist:
                threat = enemy
                threat_dist = dist
        if threat is None:
            return None
        node = self._closest_node(ctx, (bx, by), required_tags=("defend",))
        target = node.pos if node else ctx.base_pos
        score = 70.0 - min(threat_dist, 25.0)
        return BotDecision("defend_base", score, target, crouch=True, focus=(threat.x, threat.y, threat.z))

    def _beh_hunt_enemy(self, ctx: BotContext) -> Optional[BotDecision]:
        enemy = self._nearest_enemy(ctx.me, ctx.gs, max_range=60.0)
        if enemy is None:
            return None
        score = 35.0
        target = (enemy.x, enemy.y, enemy.z)
        return BotDecision("hunt_enemy", score, target, focus=(enemy.x, enemy.y, enemy.z))

    def _beh_attack_enemy_base(self, ctx: BotContext) -> BotDecision:
        node = self._closest_node(ctx, (ctx.enemy_base[0], ctx.enemy_base[1]), required_tags=("attack",))
        target = node.pos if node else ctx.enemy_base
        return BotDecision("attack_enemy_base", 20.0, target)

    def _closest_node(self, ctx: BotContext, ref_xy: Tuple[float, float], required_tags: Tuple[str, ...]) -> Optional[TacticalNode]:
        nodes = ctx.nodes_with_tags(required_tags)
        if not nodes:
            return None
        best = None
        best_dist = float("inf")
        for node in nodes:
            dx = node.pos[0] - ref_xy[0]
            dy = node.pos[1] - ref_xy[1]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best = node
                best_dist = dist
        return best

    # --- Main decision ---------------------------------------------------

    def decide(self, me, gs, mapdata) -> Dict:
        self._ensure_nav(mapdata)
        self._ensure_fields(mapdata)

        ctx = BotContext(
            brain=self,
            me=me,
            gs=gs,
            mapdata=mapdata,
            nav_graph=self.nav_graph,
            now=time.time(),
        )

        decision = self._evaluate_behaviors(ctx)
        if decision.broadcast is not None:
            self.radio.broadcast(self.team, decision.broadcast)

        inputs = {
            "mx": 0.0,
            "mz": 0.0,
            "jump": False,
            "crouch": decision.crouch,
            "walk": decision.walk,
            "fire": False,
            "interact": False,
            "yaw": math.degrees(getattr(me, "yaw_rad", 0.0)),
            "pitch": math.degrees(getattr(me, "pitch_rad", 0.0)),
        }

        goal_xy = (decision.target[0], decision.target[1])
        me_xy = (me.x, me.y)
        if self._need_replan(goal_xy):
            self._plan(me_xy, goal_xy)
        if self._stalled(me_xy):
            self._plan(me_xy, goal_xy)
        self._advance_waypoint_if_close(me_xy)

        seek = self._current_target()
        if seek is None:
            seek = goal_xy

        dx = seek[0] - me.x
        dy = seek[1] - me.y
        desired_yaw = math.atan2(-dx, dy)
        curr = getattr(me, "yaw_rad", 0.0)
        me.yaw_rad = _turn_toward(curr, desired_yaw, rate_rad_per_s=math.radians(150), dt=0.016)
        inputs["yaw"] = math.degrees(me.yaw_rad)
        inputs["mx"] = 0.0
        inputs["mz"] = 1.0

        enemy = self._nearest_enemy(me, gs)
        if enemy is not None:
            dist = math.hypot(enemy.x - me.x, enemy.y - me.y)
            inputs["fire"] = dist <= 40.0
            if inputs["fire"]:
                desired_yaw = math.atan2(-(enemy.x - me.x), enemy.y - me.y)
                me.yaw_rad = _turn_toward(me.yaw_rad, desired_yaw, rate_rad_per_s=math.radians(260), dt=0.016)
                inputs["yaw"] = math.degrees(me.yaw_rad)

        if self._flag_interact_needed(me, gs):
            inputs["interact"] = True

        return inputs
