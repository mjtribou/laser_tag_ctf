# game/bot_ai.py
import heapq
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional, NamedTuple, Iterable, Sequence

from .constants import TEAM_RED, TEAM_BLUE, TEAM_NEUTRAL, PLAYER_HEIGHT
from .bot_coordination import SquadRole
from . import nav_grid as ng
from world.map_adapter import TacticalGraph, TacticalNode, TacticalLink

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
            me.yaw_rad = _turn_toward(curr, desired_yaw, rate_rad_per_s=math.radians(220), dt=0.016)
            inputs["yaw"] = math.degrees(me.yaw_rad)

        inputs["mx"] = 0.0
        inputs["mz"] = 1.0
        inputs["fire"] = random.random() < 0.05
        inputs["walk"] = False
        inputs["crouch"] = random.random() < 0.02
        return inputs

    def debug_payload(self, me, now: float) -> Dict[str, Any]:
        return {
            "pid": getattr(me, "pid", None),
            "team": self.team,
            "time": now,
            "behavior": self.state,
            "target": self.target,
            "position": (round(me.x, 2), round(me.y, 2), round(me.z, 2)),
        }


# --- Shared nav-cache across all brains (one grid per map/settings) ---
_FIELD_CACHE = {}  # key: (nav_key, red_xy, blue_xy) -> {"red": (dist,parent), "blue": (dist,parent)}
_NAV_CACHE: Dict[Tuple, ng.NavGrid] = {}
_ENEMY_MEMORY: Dict[int, Dict[int, Tuple[Tuple[float, float, float], float]]] = defaultdict(dict)
_ENEMY_MEMORY_TTL = 12.0


@dataclass
class NavGraphIndex:
    """Convenience index for querying navigation nodes by tag/area and traversing links."""

    graph: TacticalGraph
    nodes: Dict[str, TacticalNode]
    neighbors: Dict[str, List[Tuple[str, float]]]
    by_tag: Dict[str, List[str]]
    by_area: Dict[str, List[str]]

    @classmethod
    def from_graph(cls, graph: TacticalGraph) -> "NavGraphIndex":
        nodes: Dict[str, TacticalNode] = {}
        by_tag: Dict[str, List[str]] = defaultdict(list)
        by_area: Dict[str, List[str]] = defaultdict(list)
        for node_id, node in graph.nodes.items():
            nodes[node_id] = node
            tags = tuple(getattr(node, "tags", ()) or ())
            for tag in tags:
                by_tag[tag].append(node_id)
                if tag.startswith("team:"):
                    by_area[tag].append(node_id)

        neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for link in graph.links:
            weight = float(getattr(link, "weight", 1.0) or 1.0)
            if weight <= 0.0:
                weight = 1.0
            src = getattr(link, "source", getattr(link, "from", None))
            dst = getattr(link, "target", getattr(link, "to", None))
            if not src or not dst:
                continue
            if src in nodes and dst in nodes:
                neighbors[src].append((dst, weight))
                if getattr(link, "bidirectional", True):
                    neighbors[dst].append((src, weight))

        return cls(graph=graph, nodes=nodes, neighbors=neighbors, by_tag=by_tag, by_area=by_area)

    # --- Queries -----------------------------------------------------

    def node_ids_with_tags(self, required: Sequence[str], area: Optional[str] = None) -> List[str]:
        required_clean = [tag for tag in (required or []) if tag]
        pools: List[Sequence[str]] = []
        if area:
            pools.append(self.by_area.get(area, ()))
        for tag in required_clean:
            pools.append(self.by_tag.get(tag, ()))
        if not pools:
            return list(self.nodes.keys())
        result: Optional[set] = None
        for pool in pools:
            if not pool:
                return []
            ids = set(pool)
            if result is None:
                result = ids
            else:
                result &= ids
            if not result:
                return []
        return sorted(result) if result else []

    def nodes_with_tags(self, required: Sequence[str], area: Optional[str] = None) -> List[TacticalNode]:
        ids = self.node_ids_with_tags(required, area)
        return [self.nodes[i] for i in ids]

    def nodes_with_any_tags(self, candidates: Sequence[str], area: Optional[str] = None) -> List[TacticalNode]:
        tags = [tag for tag in (candidates or []) if tag]
        if not tags:
            return self.nodes_in_area(area) if area else list(self.nodes.values())
        result_ids: set[str] = set()
        for tag in tags:
            for node_id in self.by_tag.get(tag, ()):  # union of tag pools
                result_ids.add(node_id)
        if area:
            area_ids = set(self.by_area.get(area, ()))
            if result_ids:
                result_ids &= area_ids
            else:
                result_ids = area_ids
        return [self.nodes[i] for i in result_ids]

    def closest(self, position: Tuple[float, float, float], *, required: Sequence[str] = (), area: Optional[str] = None) -> Optional[TacticalNode]:
        ids = self.node_ids_with_tags(required, area)
        if not ids:
            return None
        px, py, pz = position
        best_id = None
        best_dist = float("inf")
        for node_id in ids:
            node = self.nodes[node_id]
            nx, ny, nz = node.pos
            dist = math.sqrt((px - nx) ** 2 + (py - ny) ** 2 + (pz - nz) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_id = node_id
        return self.nodes.get(best_id) if best_id else None

    def nodes_in_area(self, area_tag: str) -> List[TacticalNode]:
        if not area_tag:
            return list(self.nodes.values())
        ids = self.by_area.get(area_tag, [])
        return [self.nodes[i] for i in ids]

    def neighbors_of(self, node_id: str) -> List[Tuple[TacticalNode, float]]:
        out = []
        for neighbor_id, weight in self.neighbors.get(node_id, () ):
            node = self.nodes.get(neighbor_id)
            if node is not None:
                out.append((node, weight))
        return out


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
    visible_enemies: Tuple[any, ...] = ()

    @property
    def team(self) -> int:
        return self.brain.team

    @property
    def base_pos(self) -> Tuple[float, float, float]:
        return self.brain.base_pos

    @property
    def enemy_base(self) -> Tuple[float, float, float]:
        return self.brain.enemy_base

    @property
    def nav_index(self) -> Optional[NavGraphIndex]:
        self.brain._ensure_nav_index(self.mapdata, self.nav_graph)
        return self.brain.nav_index

    def nav_nodes(self) -> Iterable[TacticalNode]:
        index = self.nav_index
        if index is not None:
            return index.nodes.values()
        nodes = getattr(self.mapdata, "nav_nodes", None)
        return nodes if nodes is not None else []

    def nodes_with_tags(self, required: Iterable[str], area: Optional[str] = None) -> List[TacticalNode]:
        index = self.nav_index
        if index is not None:
            return index.nodes_with_tags(required, area)
        tags = set(required)
        if not tags:
            nodes = list(self.nav_nodes())
            if area:
                return [node for node in nodes if area in getattr(node, "tags", ())]
            return nodes
        out: List[TacticalNode] = []
        for node in self.nav_nodes():
            node_tags = set(getattr(node, "tags", ()))
            if tags.issubset(node_tags) and (not area or area in node_tags):
                out.append(node)
        return out

    def nodes_in_area(self, area_tag: str) -> List[TacticalNode]:
        index = self.nav_index
        if index is not None:
            return index.nodes_in_area(area_tag)
        if not area_tag:
            return list(self.nav_nodes())
        out: List[TacticalNode] = []
        for node in self.nav_nodes():
            if area_tag in getattr(node, "tags", ()):  # fall back to manual filter
                out.append(node)
        return out

    def nearest_node(self, position: Tuple[float, float, float], *, required: Iterable[str] = (), area: Optional[str] = None) -> Optional[TacticalNode]:
        index = self.nav_index
        if index is not None:
            return index.closest(position, required=list(required), area=area)
        candidates = self.nodes_with_tags(required)
        if area:
            candidates = [node for node in candidates if area in getattr(node, "tags", ())]
        if not candidates:
            return None
        px, py, pz = position
        best = None
        best_dist = float("inf")
        for node in candidates:
            nx, ny, nz = node.pos
            dist = math.sqrt((px - nx) ** 2 + (py - ny) ** 2 + (pz - nz) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = node
        return best

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
    metadata: Optional[Dict[str, Any]] = None


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
        idle_turn_rate_deg: float = 240.0,
        engaged_turn_rate_deg: float = 420.0,
        engagement_range_m: float = 40.0,
        target_acquire_range_m: float = 45.0,
    ) -> None:
        self.team = team
        self.base_pos = base_pos
        self.enemy_base = enemy_base
        self.target_players = bool(target_players)
        self.nav_graph = nav_graph
        self.radio = radio
        self.nav_index: Optional[NavGraphIndex] = None
        if nav_graph and getattr(nav_graph, "nodes", None):
            self.nav_index = NavGraphIndex.from_graph(nav_graph)

        # Navigation state
        self._nav: Optional[ng.NavGrid] = None
        self._fields = None
        self._path: List[Tuple[float, float]] = []
        self._path_i: int = 0
        self._last_goal_xy: Optional[Tuple[float, float]] = None
        self._next_plan_t: float = 0.0
        self._last_progress_t: float = 0.0
        self._last_progress_dist: float = float("inf")
        self.last_decision: Optional[BotDecision] = None
        self._voxel_lookup = None
        self._debug_plan: Dict[str, Any] = {}
        self._squad_role: Optional[SquadRole] = None
        self._squad_target_pos: Optional[Tuple[float, float, float]] = None
        idle_deg = max(30.0, float(idle_turn_rate_deg))
        engaged_deg = max(idle_deg, float(engaged_turn_rate_deg))
        self._turn_rate_idle = math.radians(idle_deg)
        self._turn_rate_engaged = math.radians(engaged_deg)
        self._engagement_range = max(0.0, float(engagement_range_m))
        target_range = max(self._engagement_range, float(target_acquire_range_m))
        self._target_acquire_range = target_range
        self._micro_state: str = ""
        self._micro_target: Optional[Tuple[float, float]] = None
        self._micro_retreat_vec: Optional[Tuple[float, float]] = None
        self._suppression_timer: float = 0.0

        # Cached geometry for cheap line-of-sight checks
        self._los_cache_key: Optional[Tuple[int, int, float]] = None
        self._los_blocks: List[Tuple[float, float, float, float, float, float]] = []

    # --- Nav graph helpers -------------------------------------------

    def _ensure_nav_index(self, mapdata, nav_graph: Optional[TacticalGraph] = None) -> None:
        if self.nav_index is not None:
            return
        graph = nav_graph or self.nav_graph
        if graph is None or not getattr(graph, "nodes", None):
            nodes = list(getattr(mapdata, "nav_nodes", []) or [])
            links = list(getattr(mapdata, "nav_links", []) or [])
            if nodes and links:
                graph = TacticalGraph(nodes={node.node_id: node for node in nodes}, links=tuple(links))
        if graph and getattr(graph, "nodes", None):
            self.nav_graph = graph
            self.nav_index = NavGraphIndex.from_graph(graph)

    def set_squad_role(self, role: Optional[SquadRole], nav_index: Optional[NavGraphIndex]) -> None:
        self._squad_role = role
        self._squad_target_pos = None
        if role is None or nav_index is None:
            return
        node = nav_index.nodes.get(role.target_node) if role.target_node else None
        if node is not None:
            self._squad_target_pos = node.pos

    def _team_area_tag(self, team: int) -> str:
        if team == TEAM_RED:
            return "team:red_area"
        if team == TEAM_BLUE:
            return "team:blue_area"
        return "team:neutral_area"

    def _enemy_team(self) -> int:
        return TEAM_BLUE if self.team == TEAM_RED else TEAM_RED

    def _forward_sign(self) -> float:
        return 1.0 if self.enemy_base[0] >= self.base_pos[0] else -1.0

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

    def _ensure_los_blocks(self, mapdata) -> None:
        voxel_lookup = getattr(mapdata, "voxel_lookup", None)
        if voxel_lookup is not None:
            self._voxel_lookup = voxel_lookup
            self._los_blocks = []
            self._los_cache_key = ("voxel_lookup", id(voxel_lookup))
            return

        self._voxel_lookup = None
        if mapdata is None:
            self._los_blocks = []
            self._los_cache_key = None
            return

        blocks = getattr(mapdata, "blocks", ()) or ()
        key = (id(mapdata), len(blocks), float(getattr(mapdata, "cube_size", 1.0) or 1.0))
        if self._los_cache_key == key:
            return

        los_blocks: List[Tuple[float, float, float, float, float, float]] = []
        EPS = 1e-4
        for block in blocks:
            try:
                cx, cy, cz = block.pos
                sx, sy, sz = block.size
            except Exception:
                continue

            hx = max(0.0, 0.5 * float(sx)) + EPS
            hy = max(0.0, 0.5 * float(sy)) + EPS
            hz = max(0.0, 0.5 * float(sz)) + EPS
            los_blocks.append(
                (
                    float(cx) - hx,
                    float(cx) + hx,
                    float(cy) - hy,
                    float(cy) + hy,
                    float(cz) - hz,
                    float(cz) + hz,
                )
            )

        self._los_blocks = los_blocks
        self._los_cache_key = key

    @staticmethod
    def _segment_hits_box(
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        bounds: Tuple[float, float, float, float, float, float],
        *,
        eps: float = 1e-4,
    ) -> Optional[float]:
        sx, sy, sz = start
        ex, ey, ez = end
        dx, dy, dz = ex - sx, ey - sy, ez - sz

        tmin, tmax = 0.0, 1.0
        for S, D, mn, mx in ((sx, dx, bounds[0], bounds[1]), (sy, dy, bounds[2], bounds[3]), (sz, dz, bounds[4], bounds[5])):
            if abs(D) < 1e-8:
                if S < mn - eps or S > mx + eps:
                    return None
                continue
            invD = 1.0 / D
            t1 = (mn - S) * invD
            t2 = (mx - S) * invD
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmax < tmin:
                return None

        if tmax < 0.0 or tmin > 1.0:
            return None

        hit_t = tmin if tmin >= 0.0 else tmax
        if hit_t is None:
            return None
        if hit_t <= eps or hit_t >= 1.0 - eps:
            return None
        return hit_t

    def _has_line_of_sight(self, me, target, mapdata) -> bool:
        self._ensure_los_blocks(mapdata)
        if self._voxel_lookup is not None:
            eye_offset = 0.30 * PLAYER_HEIGHT
            start = (
                float(getattr(me, "x", 0.0)),
                float(getattr(me, "y", 0.0)),
                float(getattr(me, "z", 0.0)) + eye_offset,
            )
            end = (
                float(getattr(target, "x", 0.0)),
                float(getattr(target, "y", 0.0)),
                float(getattr(target, "z", 0.0)) + eye_offset,
            )
            return not self._voxel_lookup.ray_hits_solid(start, end, ignore_start=True, ignore_end=True)

        if not self._los_blocks:
            return True

        eye_offset = 0.30 * PLAYER_HEIGHT
        sx = float(getattr(me, "x", 0.0))
        sy = float(getattr(me, "y", 0.0))
        sz = float(getattr(me, "z", 0.0)) + eye_offset

        ex = float(getattr(target, "x", 0.0))
        ey = float(getattr(target, "y", 0.0))
        ez = float(getattr(target, "z", 0.0)) + eye_offset

        start = (sx, sy, sz)
        end = (ex, ey, ez)
        EPS = 1e-4

        for bounds in self._los_blocks:
            minx, maxx, miny, maxy, minz, maxz = bounds
            # Skip the block if shooter or target is inside it; these represent floors/ramps.
            if (
                (minx - EPS <= sx <= maxx + EPS)
                and (miny - EPS <= sy <= maxy + EPS)
                and (minz - EPS <= sz <= maxz + EPS)
            ):
                continue
            if (
                (minx - EPS <= ex <= maxx + EPS)
                and (miny - EPS <= ey <= maxy + EPS)
                and (minz - EPS <= ez <= maxz + EPS)
            ):
                continue

            if self._segment_hits_box(start, end, bounds, eps=EPS) is not None:
                return False
        return True

    def _need_replan(self, goal_xy: Tuple[float, float]) -> bool:
        if not self._path or self._last_goal_xy is None:
            return True
        gx, gy = goal_xy
        lgx, lgy = self._last_goal_xy
        if (abs(gx - lgx) + abs(gy - lgy)) > 1.0:
            return True
        return time.time() >= self._next_plan_t

    def _plan(self, mapdata, me_xy: Tuple[float, float], goal_xy: Tuple[float, float]):
        now_t = time.time()
        self._last_goal_xy = goal_xy

        self._ensure_nav_index(mapdata, self.nav_graph)
        graph_path = None
        if self.nav_index is not None:
            graph_path = self._plan_nav_graph_path(mapdata, me_xy, goal_xy)

        if graph_path:
            self._path = graph_path
        else:
            self._ensure_nav(mapdata)
            sxy = ng.nearest_passable_xy(self._nav, me_xy[0], me_xy[1], max_radius=8)
            gxy = ng.nearest_passable_xy(self._nav, goal_xy[0], goal_xy[1], max_radius=12)
            self._path = ng.astar_bounded(self._nav, sxy, gxy, w=1.25, pad=12, max_iter=20000)

        if not self._path:
            self._path = [goal_xy]

        self._path_i = 0
        self._last_progress_t = now_t
        self._last_progress_dist = float("inf")
        self._next_plan_t = now_t + random.uniform(1.0, 2.0)
        self._debug_plan = {
            "goal": (round(goal_xy[0], 3), round(goal_xy[1], 3)),
            "path": [(round(px, 3), round(py, 3)) for px, py in self._path],
            "used_graph": bool(graph_path),
            "decision": getattr(self.last_decision, "name", None),
        }

    def _plan_nav_graph_path(self, mapdata, me_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        index = self.nav_index
        if index is None or not index.nodes:
            return []

        start_node = index.closest((me_xy[0], me_xy[1], 0.0))
        goal_node = index.closest((goal_xy[0], goal_xy[1], 0.0))
        if start_node is None or goal_node is None:
            return []

        if start_node.node_id == goal_node.node_id:
            return [(goal_xy[0], goal_xy[1])]

        def exposure_penalty(node: TacticalNode) -> float:
            tags = getattr(node, "tags", ()) or ()
            return 0.0 if ("cover" in tags or "peek" in tags) else 0.5

        open_heap: List[Tuple[float, str]] = []
        g_score: Dict[str, float] = {start_node.node_id: 0.0}
        f_score_start = self._graph_heuristic(start_node, goal_node)
        heapq.heappush(open_heap, (f_score_start, start_node.node_id))
        came_from: Dict[str, str] = {}

        while open_heap:
            _, current_id = heapq.heappop(open_heap)
            if current_id == goal_node.node_id:
                return self._reconstruct_nav_path(came_from, current_id, start_node.node_id, goal_xy)

            current_node = index.nodes[current_id]
            current_g = g_score[current_id]
            for neighbor_node, weight in index.neighbors_of(current_id):
                penalty = exposure_penalty(neighbor_node)
                tentative_g = current_g + weight * (1.0 + penalty)
                neighbor_id = neighbor_node.node_id
                if tentative_g >= g_score.get(neighbor_id, float("inf")):
                    continue
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g
                f_score = tentative_g + self._graph_heuristic(neighbor_node, goal_node) + penalty
                heapq.heappush(open_heap, (f_score, neighbor_id))

        return []

    @staticmethod
    def _graph_heuristic(node: TacticalNode, goal: TacticalNode) -> float:
        nx, ny, nz = node.pos
        gx, gy, gz = goal.pos
        return math.sqrt((nx - gx) ** 2 + (ny - gy) ** 2 + (nz - gz) ** 2)

    def _reconstruct_nav_path(
        self,
        came_from: Dict[str, str],
        current_id: str,
        start_id: str,
        goal_xy: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        index = self.nav_index
        if index is None:
            return [goal_xy]
        path_nodes = [current_id]
        while current_id != start_id:
            current_id = came_from[current_id]
            path_nodes.append(current_id)
        path_nodes.reverse()
        coords = [(index.nodes[node_id].pos[0], index.nodes[node_id].pos[1]) for node_id in path_nodes]
        if math.hypot(coords[-1][0] - goal_xy[0], coords[-1][1] - goal_xy[1]) > 0.5:
            coords.append(goal_xy)
        return coords

    def _advance_waypoint_if_close(self, me_xy: Tuple[float, float], threshold: float = 0.6):
        if not self._path or self._path_i >= len(self._path):
            return
        tx, ty = self._path[self._path_i]

        # When navigating around sharp corners we want to make sure the bot fully
        # reaches the corner node before advancing to the next waypoint. Otherwise
        # it can start steering towards the next point too early and scrape
        # against the corner, which is exactly what happens on ai_test.json's
        # push lane. Detect large course changes and tighten the threshold.
        if self._path_i + 1 < len(self._path):
            nxt = self._path[self._path_i + 1]
            prv = self._path[self._path_i - 1] if self._path_i - 1 >= 0 else None
            if prv is not None:
                vx0 = tx - prv[0]
                vy0 = ty - prv[1]
                vx1 = nxt[0] - tx
                vy1 = nxt[1] - ty
                len0 = math.hypot(vx0, vy0)
                len1 = math.hypot(vx1, vy1)
                if len0 > 1e-3 and len1 > 1e-3:
                    dot = (vx0 * vx1 + vy0 * vy1) / (len0 * len1)
                    if dot < 0.5:
                        threshold = min(threshold, 0.35)
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

    def _enemy_memory_bucket(self) -> Dict[int, Tuple[Tuple[float, float, float], float]]:
        return _ENEMY_MEMORY.setdefault(self.team, {})

    def _remember_enemy(self, enemy, now: Optional[float] = None) -> None:
        pid = getattr(enemy, "pid", None)
        if pid is None:
            return
        if now is None:
            now = time.time()
        bucket = self._enemy_memory_bucket()
        bucket[pid] = ((float(enemy.x), float(enemy.y), float(enemy.z)), float(now))

    def _forget_stale_enemies(self, now: float) -> None:
        bucket = self._enemy_memory_bucket()
        stale: List[int] = []
        for pid, (_pos, seen_at) in bucket.items():
            if (now - seen_at) > _ENEMY_MEMORY_TTL:
                stale.append(pid)
        for pid in stale:
            bucket.pop(pid, None)

    def _last_known_enemy(self, now: float) -> Optional[Tuple[Tuple[float, float, float], float, int]]:
        bucket = self._enemy_memory_bucket()
        best: Optional[Tuple[Tuple[float, float, float], float, int]] = None
        for pid, value in bucket.items():
            pos, seen_at = value
            age = now - seen_at
            if age > _ENEMY_MEMORY_TTL:
                continue
            if best is None or seen_at > best[1]:
                best = (pos, seen_at, pid)
        if best is None:
            return None
        pos, seen_at, pid = best
        return pos, now - seen_at, pid

    def _update_visible_enemies(self, me, gs, mapdata, now: float) -> Tuple[any, ...]:
        visible: List[any] = []
        for enemy in gs.players.values():
            if enemy.team == me.team or not getattr(enemy, "alive", True):
                continue
            if self._has_line_of_sight(me, enemy, mapdata):
                self._remember_enemy(enemy, now)
                visible.append(enemy)
        return tuple(visible)

    def _nearest_enemy(
        self,
        me,
        gs,
        max_range: Optional[float] = None,
        require_line: bool = False,
        mapdata=None,
        now: Optional[float] = None,
        candidates: Optional[Iterable] = None,
        prechecked_line: bool = False,
    ):
        if now is None:
            now = time.time()
        if max_range is None:
            max_range = self._target_acquire_range
        best = None
        best_d2 = max_range * max_range
        pool: Iterable
        if candidates is not None:
            pool = candidates
        else:
            pool = gs.players.values()
        for enemy in pool:
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
            if require_line and not prechecked_line and not self._has_line_of_sight(me, enemy, mapdata):
                continue
            if d2 < best_d2:
                best_d2 = d2
                best = enemy
        if best is not None:
            self._remember_enemy(best, now)
        return best

    # --- Behavior evaluation --------------------------------------------

    def _evaluate_behaviors(self, ctx: BotContext) -> BotDecision:
        candidates: List[BotDecision] = []
        for behavior in (
            self._beh_return_flag,
            self._beh_escort_call,
            self._beh_retrieve_flag,
            self._beh_defend_base,
            self._beh_hold_angle,
            self._beh_flank_enemy,
            self._beh_push_lane,
            self._beh_hunt_enemy,
            self._beh_attack_enemy_base,
        ):
            decision = behavior(ctx)
            if decision is not None:
                candidates.append(decision)

        if not candidates and self._squad_role and self._squad_target_pos is not None:
            meta = {
                "tactic": "hold_role",
                "role": self._squad_role.name,
                "target_node": getattr(self._squad_role, "target_node", None),
            }
            score = 45.0 + float(getattr(self._squad_role, "priority", 0.0) * 20.0)
            return BotDecision("role_objective", score, self._squad_target_pos, walk=True, metadata=meta)

        if not candidates:
            return BotDecision("idle_patrol", 0.0, self.enemy_base)

        if self._squad_role is not None:
            preferences = {
                "anchor": {"defend": 12.0, "return_flag": 8.0, "hold_role": 6.0},
                "entry": {"push": 10.0, "attack": 6.0},
                "flank": {"flank": 12.0, "hunt": 5.0, "hunt_memory": 4.0},
            }
            role_name = self._squad_role.name.lower()
            pref = preferences.get(role_name, {})
            for decision in candidates:
                tactic = decision.metadata.get("tactic") if decision.metadata else None
                if tactic in pref:
                    decision.score += pref[tactic]
                else:
                    decision.score -= min(6.0, self._squad_role.priority * 4.0) if decision.metadata else 0.0

        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates[0]

    def _beh_return_flag(self, ctx: BotContext) -> Optional[BotDecision]:
        if ctx.me.carrying_flag is None:
            return None
        intent = BotIntent("escort_me", ctx.me.pid, (ctx.me.x, ctx.me.y, ctx.me.z), ctx.now + 2.5)
        metadata = {"tactic": "return_flag"}
        return BotDecision("return_flag", 100.0, ctx.base_pos, broadcast=intent, metadata=metadata)

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
        metadata = {"tactic": "escort", "origin_pid": best.origin_pid}
        return BotDecision("escort_request", score, best.position, walk=False, metadata=metadata)

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
        metadata = {"tactic": "retrieve_flag", "flag_team": enemy_flag.team}
        return BotDecision("retrieve_flag", score, target, metadata=metadata)

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
        cover = self._select_cover_for_threat(ctx, threat)
        target = cover.pos if cover else ctx.base_pos
        score = 70.0 - min(threat_dist, 25.0)
        metadata = {
            "tactic": "defend",
            "threat_pid": getattr(threat, "pid", None),
            "cover_node": getattr(cover, "node_id", None) if cover else None,
        }
        return BotDecision("defend_base", score, target, crouch=True, focus=(threat.x, threat.y, threat.z), metadata=metadata)

    def _beh_hold_angle(self, ctx: BotContext) -> Optional[BotDecision]:
        if self._squad_role is None or self._squad_target_pos is None:
            return None
        if self._squad_role.name.lower() != "anchor":
            return None

        index = ctx.nav_index
        node = None
        if index is not None and getattr(self._squad_role, "target_node", None):
            node = index.nodes.get(self._squad_role.target_node)
        if node is None and index is not None:
            node = self._closest_node(ctx, (self._squad_target_pos[0], self._squad_target_pos[1]), required_tags=("cover",))
        hold_pos = self._squad_target_pos
        if node is not None:
            hold_pos = node.pos
        dist = math.hypot(hold_pos[0] - ctx.me.x, hold_pos[1] - ctx.me.y)

        focus = None
        if node is not None and getattr(node, "facing", None):
            fx, fy, fz = node.facing
            focus = (
                hold_pos[0] + fx * 10.0,
                hold_pos[1] + fy * 10.0,
                hold_pos[2] + fz * 10.0,
            )
        metadata = {
            "tactic": "hold",
            "role": self._squad_role.name,
            "target_node": getattr(node, "node_id", None),
        }
        score = 60.0 + min(12.0, self._squad_role.priority * 20.0) - min(10.0, dist * 2.0)
        return BotDecision("hold_angle", score, hold_pos, crouch=True, walk=False, focus=focus, metadata=metadata)

    def _beh_flank_enemy(self, ctx: BotContext) -> Optional[BotDecision]:
        index = ctx.nav_index
        if index is None:
            return None

        enemy_area = self._team_area_tag(self._enemy_team())
        flank_nodes = index.nodes_with_tags(["cover"], area=enemy_area)
        if not flank_nodes:
            return None

        enemy_samples: List[Tuple[float, float]] = []
        for enemy in ctx.visible_enemies:
            if getattr(enemy, "alive", True):
                enemy_samples.append((float(enemy.x), float(enemy.y)))
        memory = self._last_known_enemy(ctx.now)
        if not enemy_samples and memory is not None:
            pos, _age, _pid = memory
            enemy_samples.append((float(pos[0]), float(pos[1])))

        if enemy_samples:
            mean_enemy_y = sum(p[1] for p in enemy_samples) / len(enemy_samples)
        else:
            mean_enemy_y = self.enemy_base[1]

        enemy_offset = mean_enemy_y - self.enemy_base[1]
        forward_sign = self._forward_sign()

        best = None
        best_value = -float("inf")
        for node in flank_nodes:
            nx, ny, nz = node.pos
            lateral = ny - self.enemy_base[1]
            forward_progress = forward_sign * (nx - ctx.me.x)
            side_bonus = -lateral * enemy_offset if enemy_offset else abs(lateral)
            value = abs(lateral) * 0.9 + side_bonus * 0.6 + max(0.0, forward_progress) * 0.25
            if value > best_value:
                best_value = value
                best = node

        if best is None:
            return None

        target = best.pos
        dist = math.hypot(target[0] - ctx.me.x, target[1] - ctx.me.y)
        base_score = 62.0 - min(dist * 0.35, 20.0) + min(8.0, best_value)
        metadata = {
            "tactic": "flank",
            "target_node": best.node_id,
            "area": enemy_area,
            "lateral_offset": round(best.pos[1] - self.enemy_base[1], 3),
            "enemy_mean_y": round(mean_enemy_y, 3),
        }
        return BotDecision("flank_enemy", base_score, target, walk=True, metadata=metadata)

    def _beh_push_lane(self, ctx: BotContext) -> Optional[BotDecision]:
        index = ctx.nav_index
        if index is None:
            return None

        neutral_tag = "team:neutral_area"
        forward_sign = self._forward_sign()
        neutral_nodes = index.nodes_with_tags(["cover"], area=neutral_tag)
        if not neutral_nodes:
            neutral_nodes = index.nodes_with_tags(["nav"], area=neutral_tag)
        if not neutral_nodes:
            return None

        best = None
        best_progress = -float("inf")
        for node in neutral_nodes:
            nx, ny, nz = node.pos
            progress = forward_sign * (nx - ctx.me.x)
            if progress < -1.0:
                continue
            lateral = abs(ny - ctx.me.y)
            value = progress - lateral * 0.15
            if value > best_progress:
                best_progress = value
                best = node

        if best is None:
            return None

        target = best.pos
        dist = math.hypot(target[0] - ctx.me.x, target[1] - ctx.me.y)
        score = 58.0 + min(12.0, best_progress * 0.6) - min(12.0, dist * 0.3)
        metadata = {
            "tactic": "push",
            "target_node": best.node_id,
            "area": neutral_tag,
            "progress": round(best_progress, 3),
        }
        return BotDecision("push_lane", score, target, walk=False, metadata=metadata)

    def _beh_hunt_enemy(self, ctx: BotContext) -> Optional[BotDecision]:
        if ctx.visible_enemies:
            enemy = self._nearest_enemy(
                ctx.me,
                ctx.gs,
                max_range=60.0,
                mapdata=ctx.mapdata,
                now=ctx.now,
                candidates=ctx.visible_enemies,
                prechecked_line=True,
            )
        else:
            enemy = None
        if enemy is not None:
            score = 35.0
            target = (enemy.x, enemy.y, enemy.z)
            metadata = {"tactic": "hunt", "target_pid": getattr(enemy, "pid", None)}
            return BotDecision("hunt_enemy", score, target, focus=(enemy.x, enemy.y, enemy.z), metadata=metadata)

        memory = self._last_known_enemy(ctx.now)
        if memory is None:
            return None
        pos, age, _pid = memory
        freshness = max(0.0, 1.0 - (age / _ENEMY_MEMORY_TTL))
        score = 20.0 + 10.0 * freshness
        target = (pos[0], pos[1], pos[2])
        metadata = {"tactic": "hunt_memory", "age": round(age, 2)}
        return BotDecision("hunt_enemy", score, target, focus=(pos[0], pos[1], pos[2]), metadata=metadata)

    def _update_micro_state(self, ctx: BotContext) -> None:
        enemy = None
        if ctx.visible_enemies:
            enemy = min(ctx.visible_enemies, key=lambda e: math.hypot(e.x - ctx.me.x, e.y - ctx.me.y))

        suppressed = ctx.now < self._suppression_timer
        if suppressed and enemy is not None:
            dx = ctx.me.x - enemy.x
            dy = ctx.me.y - enemy.y
            length = math.hypot(dx, dy) or 1.0
            self._micro_state = "retreat"
            self._micro_retreat_vec = (dx / length, dy / length)
            self._micro_target = None
            return

        self._micro_retreat_vec = None

        if enemy is not None:
            tactic = None
            if self.last_decision and self.last_decision.metadata:
                tactic = self.last_decision.metadata.get("tactic")
            if tactic in {"defend", "hold", "hold_role"}:
                self._micro_state = "peek"
                self._micro_target = (float(enemy.x), float(enemy.y))
            else:
                self._micro_state = "fight"
                self._micro_target = None
            return

        if self.last_decision and self.last_decision.metadata and self.last_decision.metadata.get("tactic") == "hold":
            self._micro_state = "hold"
            if self._squad_target_pos is not None:
                self._micro_target = (self._squad_target_pos[0], self._squad_target_pos[1])
            else:
                self._micro_target = None
            return

        self._micro_state = ""
        self._micro_target = None

    def mark_suppressed(self, now_ts: float, duration: float = 0.8) -> None:
        self._suppression_timer = max(self._suppression_timer, now_ts + max(0.1, duration))

    def _beh_attack_enemy_base(self, ctx: BotContext) -> BotDecision:
        node = self._closest_node(ctx, (ctx.enemy_base[0], ctx.enemy_base[1]), required_tags=("attack",))
        target = node.pos if node else ctx.enemy_base
        metadata = {
            "tactic": "attack",
            "target_node": getattr(node, "node_id", None) if node else None,
            "area": self._team_area_tag(self._enemy_team()),
        }
        return BotDecision("attack_enemy_base", 20.0, target, metadata=metadata)

    def _closest_node(self, ctx: BotContext, ref_xy: Tuple[float, float], required_tags: Tuple[str, ...]) -> Optional[TacticalNode]:
        ref_pos = (ref_xy[0], ref_xy[1], 0.0)
        candidate = ctx.nearest_node(ref_pos, required=required_tags)
        return candidate

    def _select_cover_for_threat(self, ctx: BotContext, threat) -> Optional[TacticalNode]:
        index = ctx.nav_index
        if index is None:
            candidates = ctx.nodes_with_tags(("cover", "peek"))
        else:
            area_tag = None
            if threat.x <= ctx.base_pos[0] - 5.0:
                area_tag = "team:blue_area" if ctx.team == TEAM_RED else "team:red_area"
            elif threat.x >= ctx.base_pos[0] + 5.0:
                area_tag = "team:red_area" if ctx.team == TEAM_RED else "team:blue_area"
            candidates = index.nodes_with_any_tags(["cover", "peek"], area=area_tag)
        if not candidates:
            return None

        px, py, pz = ctx.me.x, ctx.me.y, ctx.me.z
        tx, ty, tz = threat.x, threat.y, threat.z
        threat_vec = (tx - px, ty - py, tz - pz)
        threat_dist = math.sqrt(threat_vec[0] ** 2 + threat_vec[1] ** 2 + threat_vec[2] ** 2) or 1.0
        threat_dir = (threat_vec[0] / threat_dist, threat_vec[1] / threat_dist, threat_vec[2] / threat_dist)

        best = None
        best_score = -float("inf")
        for node in candidates:
            nx, ny, nz = node.pos
            fx, fy, fz = getattr(node, "facing", (0.0, 1.0, 0.0))
            facing_len = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
            facing_dir = (fx / facing_len, fy / facing_len, fz / facing_len)

            to_threat = (tx - nx, ty - ny, tz - nz)
            to_threat_len = math.sqrt(to_threat[0] ** 2 + to_threat[1] ** 2 + to_threat[2] ** 2) or 1.0
            to_threat_dir = (to_threat[0] / to_threat_len, to_threat[1] / to_threat_len, to_threat[2] / to_threat_len)

            cover_score = -(nx - tx) * threat_dir[0] - (ny - ty) * threat_dir[1]
            facing_score = facing_dir[0] * to_threat_dir[0] + facing_dir[1] * to_threat_dir[1] + facing_dir[2] * to_threat_dir[2]
            distance_penalty = to_threat_len
            total = 2.5 * facing_score - 0.4 * distance_penalty + 0.6 * cover_score
            if total > best_score:
                best_score = total
                best = node
        return best

    # --- Main decision ---------------------------------------------------

    def decide(self, me, gs, mapdata) -> Dict:
        self._ensure_nav(mapdata)
        self._ensure_fields(mapdata)

        now = time.time()
        self._forget_stale_enemies(now)
        visible = self._update_visible_enemies(me, gs, mapdata, now)

        ctx = BotContext(
            brain=self,
            me=me,
            gs=gs,
            mapdata=mapdata,
            nav_graph=self.nav_graph,
            now=now,
            visible_enemies=visible,
        )
        self._update_micro_state(ctx)

        decision = self._evaluate_behaviors(ctx)
        self.last_decision = decision
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
            self._plan(mapdata, me_xy, goal_xy)
        if self._stalled(me_xy):
            self._plan(mapdata, me_xy, goal_xy)
        self._advance_waypoint_if_close(me_xy)

        seek = self._current_target()
        if seek is None:
            seek = goal_xy

        move_vec = (seek[0] - me.x, seek[1] - me.y)
        move_dist = math.hypot(move_vec[0], move_vec[1])
        move_dir = (0.0, 0.0)
        if move_dist > 1e-4:
            move_dir = (move_vec[0] / move_dist, move_vec[1] / move_dist)

        enemy = None
        if ctx.visible_enemies:
            enemy = self._nearest_enemy(
                me,
                gs,
                mapdata=mapdata,
                now=ctx.now,
                candidates=ctx.visible_enemies,
                prechecked_line=True,
            )
        aim_point = None
        turn_rate = self._turn_rate_idle
        if enemy is not None:
            dist = math.hypot(enemy.x - me.x, enemy.y - me.y)
            inputs["fire"] = dist <= self._engagement_range
            aim_point = (enemy.x, enemy.y)
            turn_rate = self._turn_rate_engaged
        elif self._micro_state == "peek" and self._micro_target:
            inputs["crouch"] = False
            aim_point = (self._micro_target[0], self._micro_target[1])
            turn_rate = self._turn_rate_engaged
        if aim_point is None and decision.focus is not None:
            fx, fy, _ = decision.focus
            aim_point = (fx, fy)
        if aim_point is None and move_dist > 1e-4:
            aim_point = seek

        curr_yaw = getattr(me, "yaw_rad", 0.0)
        if aim_point is not None:
            desired_yaw = math.atan2(-(aim_point[0] - me.x), aim_point[1] - me.y)
            me.yaw_rad = _turn_toward(curr_yaw, desired_yaw, rate_rad_per_s=turn_rate, dt=0.016)
        inputs["yaw"] = math.degrees(getattr(me, "yaw_rad", 0.0))

        yaw = getattr(me, "yaw_rad", 0.0)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)
        forward = (-sin_yaw, cos_yaw)
        right = (cos_yaw, sin_yaw)

        if move_dist > 1e-4:
            mz = forward[0] * move_dir[0] + forward[1] * move_dir[1]
            mx = right[0] * move_dir[0] + right[1] * move_dir[1]
            inputs["mx"] = max(-1.0, min(1.0, mx))
            inputs["mz"] = max(-1.0, min(1.0, mz))
        else:
            inputs["mx"] = 0.0
            inputs["mz"] = 0.0

        if self._flag_interact_needed(me, gs):
            inputs["interact"] = True

        if self._micro_state == "peek":
            inputs["mx"] = 0.0
            inputs["mz"] = 0.0
        elif self._micro_state == "retreat" and self._micro_retreat_vec is not None:
            rx, ry = self._micro_retreat_vec
            inputs["mx"] = max(-1.0, min(1.0, rx))
            inputs["mz"] = max(-1.0, min(1.0, ry))
            inputs["crouch"] = True
            inputs["fire"] = False

        return inputs

    def debug_payload(self, me, now: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pid": me.pid,
            "team": self.team,
            "time": now,
            "position": (round(me.x, 2), round(me.y, 2), round(me.z, 2)),
            "carrying_flag": getattr(me, "carrying_flag", None),
            "path_remaining": max(0, len(self._path) - self._path_i) if self._path else 0,
        }
        if self.last_decision is not None:
            payload.update(
                {
                    "behavior": self.last_decision.name,
                    "score": round(self.last_decision.score, 2),
                    "target": tuple(round(v, 2) for v in self.last_decision.target),
                }
            )
            if self.last_decision.metadata:
                payload["decision_meta"] = self.last_decision.metadata
        remaining_path: List[Tuple[float, float]] = []
        if self._path and self._path_i < len(self._path):
            tail = self._path[self._path_i : self._path_i + 6]
            remaining_path = [(round(px, 2), round(py, 2)) for px, py in tail]
        if remaining_path:
            payload["path_nodes"] = remaining_path
        current_target = self._current_target()
        if current_target is not None:
            payload["current_target"] = (round(current_target[0], 2), round(current_target[1], 2))
        if self._debug_plan:
            payload["plan"] = {
                "goal": list(self._debug_plan.get("goal", ())),
                "path": [list(p) for p in self._debug_plan.get("path", [])],
                "used_graph": bool(self._debug_plan.get("used_graph", False)),
                "decision": self._debug_plan.get("decision"),
            }
        if self._squad_role is not None:
            role_payload = {
                "role": self._squad_role.name,
                "priority": round(float(self._squad_role.priority), 2),
                "target_node": getattr(self._squad_role, "target_node", None),
            }
            if self._squad_target_pos is not None:
                role_payload["target_pos"] = tuple(round(v, 2) for v in self._squad_target_pos)
            payload["squad_role"] = role_payload
        if self._micro_state:
            payload["micro_state"] = {
                "state": self._micro_state,
                "target": list(self._micro_target) if self._micro_target else None,
                "retreat_vec": list(self._micro_retreat_vec) if self._micro_retreat_vec else None,
                "suppressed_until": round(self._suppression_timer, 2),
            }
        return payload
    def nodes_with_any_tags(self, tags: Iterable[str], area: Optional[str] = None) -> List[TacticalNode]:
        index = self.nav_index
        if index is not None:
            return index.nodes_with_any_tags(tags, area)
        tags = [tag for tag in tags if tag]
        nodes = list(self.nav_nodes())
        out: List[TacticalNode] = []
        for node in nodes:
            node_tags = set(getattr(node, "tags", ()))
            if tags:
                if any(tag in node_tags for tag in tags):
                    if not area or area in node_tags:
                        out.append(node)
            elif not area or area in node_tags:
                out.append(node)
        return out
