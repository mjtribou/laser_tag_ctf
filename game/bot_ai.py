# game/bot_ai.py
import math, random, time
from typing import Dict, Tuple, List, Optional

from .constants import TEAM_RED, TEAM_BLUE
from . import nav_grid as ng

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

def _nav_key(mapdata, cell, agent_radius):
    return (
        id(mapdata),
        len(getattr(mapdata, "blocks", [])),
        getattr(mapdata, "bounds", None),
        round(float(cell), 3),
        round(float(agent_radius), 3),
    )

class AStarBotBrain:
    """
    Grid A* navigation:
    - Builds a nav grid from mapdata.blocks (inflated by player radius).
    - Plans toward enemy base (or home base if carrying flag).
    - Presses 'interact' near flags to pick up/return.
    """
    def __init__(self, team: int, base_pos: Tuple[float,float,float], enemy_base: Tuple[float,float,float]):
        self.team = team
        self.base_pos = base_pos
        self.enemy_base = enemy_base
        self._nav: Optional[ng.NavGrid] = None
        self._path: List[Tuple[float,float]] = []
        self._path_i: int = 0

        # planning cadence
        self._last_plan_t: float = 0.0
        self._next_plan_t: float = 0.0
        self._plan_interval_min = 1.3
        self._plan_interval_max = 2.2

        # goal tracking
        self._last_goal_xy: Optional[Tuple[float,float]] = None

        # progress tracking vs current waypoint
        self._last_progress_t: float = 0.0
        self._last_progress_dist: float = float("inf")

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
        """Compute/cached flow fields to each base."""
        self._ensure_nav(mapdata)
        cell = getattr(mapdata, "cube_size", 1.0)
        radius = getattr(mapdata, "agent_radius", 0.5)
        key_nav = _nav_key(mapdata, cell, radius)
        red_xy  = (round(mapdata.red_base[0], 3),  round(mapdata.red_base[1], 3))
        blue_xy = (round(mapdata.blue_base[0], 3), round(mapdata.blue_base[1], 3))
        fkey = (key_nav, red_xy, blue_xy)

        fields = _FIELD_CACHE.get(fkey)
        if fields is None:
            red  = ng.dijkstra_field(self._nav, [red_xy])
            blue = ng.dijkstra_field(self._nav, [blue_xy])
            fields = {"red": red, "blue": blue}
            _FIELD_CACHE[fkey] = fields
        self._fields = fields

    def _need_replan(self, goal_xy: Tuple[float,float]) -> bool:
        if not self._path or self._last_goal_xy is None:
            return True
        gx, gy = goal_xy
        lgx, lgy = self._last_goal_xy
        # immediate replan if goal shifted materially
        if (abs(gx - lgx) + abs(gy - lgy)) > 1.0:
            return True
        # otherwise: plan on a cadence
        return time.time() >= self._next_plan_t

    def _plan(self, me_xy: Tuple[float, float], goal_xy: Tuple[float, float]):
        now_t = time.time()
        self._last_plan_t = now_t
        self._last_goal_xy = goal_xy

        # Snap onto passable cells
        sxy = ng.nearest_passable_xy(self._nav, me_xy[0], me_xy[1], max_radius=8)
        gxy = ng.nearest_passable_xy(self._nav, goal_xy[0], goal_xy[1], max_radius=12)

        # Cheap bounded, weighted A* (arrayified)
        self._path = ng.astar_bounded(self._nav, sxy, gxy, w=1.3, pad=10, max_iter=15000)
        self._path_i = 0

        # reset progress trackers
        self._last_progress_t = now_t
        self._last_progress_dist = float("inf")

        # schedule next plan with jitter
        interval = random.uniform(self._plan_interval_min, self._plan_interval_max)
        self._next_plan_t = now_t + interval

    def _advance_waypoint_if_close(self, me_xy: Tuple[float,float], threshold: float = 0.6):
        if not self._path or self._path_i >= len(self._path):
            return
        tx, ty = self._path[self._path_i]
        if math.hypot(me_xy[0] - tx, me_xy[1] - ty) <= threshold:
            self._path_i += 1
            self._last_progress_xy = me_xy
            self._last_progress_t = time.time()

    def _current_target(self) -> Optional[Tuple[float,float]]:
        if not self._path or self._path_i >= len(self._path):
            return None
        # look-ahead one step to smooth corners
        j = min(self._path_i + 1, len(self._path) - 1)
        return self._path[j]

    def _flag_interact_needed(self, me, gs, mapdata) -> bool:
        # Press 'interact' if we're near any flag (pickup/return handled server-side).
        for team_id, fl in gs.flags.items():
            d = math.hypot(me.x - fl.x, me.y - fl.y)
            if d <= 1.6:  # slightly generous vs server constants
                return True
        return False

    def _stalled(self, me_xy: Tuple[float,float]) -> bool:
        # If we have a waypoint, check distance improvement; tolerate small oscillations.
        tgt = self._current_target()
        if tgt is None:
            return False
        dist = math.hypot(me_xy[0] - tgt[0], me_xy[1] - tgt[1])
        improved = dist < (self._last_progress_dist - 0.15)
        now_t = time.time()
        if improved:
            self._last_progress_dist = dist
            self._last_progress_t = now_t
            return False
        # If we haven't gotten closer for a while and we're not basically there, call it a stall.
        return (now_t - self._last_progress_t) > 1.5 and dist > 0.8

    def _nearest_enemy(self, me, gs, max_range: float = 45.0):
        """Return nearest alive enemy Player within max_range (meters), else None."""
        best = None
        best_d2 = max_range * max_range
        for pid, p in gs.players.items():
            if p.team == me.team or not p.alive:
                continue
            dx = p.x - me.x
            dy = p.y - me.y
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best = p
        return best

    def decide(self, me, gs, mapdata) -> Dict:
        # Ensure nav grid + cached flow fields are ready
        self._ensure_nav(mapdata)
        self._ensure_fields(mapdata)

        # Choose strategic goal
        going_home = (me.carrying_flag is not None)
        if going_home:
            goal = self.base_pos
            # Flow field toward our own base
            field_dist, field_parent = self._fields["red" if self.team == TEAM_RED else "blue"]
        else:
            goal = self.enemy_base
            # Flow field toward enemy base
            field_dist, field_parent = self._fields["blue" if self.team == TEAM_RED else "red"]

        goal_xy = (goal[0], goal[1])
        me_xy = (me.x, me.y)

        # Opportunistic plan refresh; also replan on stall (with a small guard so we don't spam)
        if self._need_replan(goal_xy):
            self._plan(me_xy, goal_xy)
        if self._stalled(me_xy) and (time.time() + 0.2) >= self._next_plan_t:
            self._plan(me_xy, goal_xy)

        # Advance along the current waypoint path if we’re close
        self._advance_waypoint_if_close(me_xy, threshold=0.6)

        # Steering target: prefer the flow field parent from our current cell; fallback to path/goal
        target_xy = None
        c = self._nav.world_to_cell(me.x, me.y)
        if self._nav.in_bounds(c):
            px, py = field_parent[c.iy][c.ix]
            if px >= 0 and py >= 0:
                # Use the next cell on the flow field toward the goal
                target_xy = self._nav.cell_center(ng.Cell(px, py))

        if target_xy is None:
            # Fallback: use A* path lookahead or the raw goal
            target_xy = self._current_target() or goal_xy

        # Build input packet (server expects degrees for yaw/pitch)
        inputs = {
            "mx": 0.0, "mz": 0.0, "jump": False, "crouch": False, "walk": False,
            "fire": False, "interact": False,
            "yaw": math.degrees(getattr(me, "yaw_rad", 0.0)),
            "pitch": math.degrees(getattr(me, "pitch_rad", 0.0)),
        }

        # Turn toward target smoothly
        dx = target_xy[0] - me.x
        dy = target_xy[1] - me.y
        if abs(dx) + abs(dy) > 1e-3:
            desired_yaw = math.atan2(-dx, dy)  # Panda: H=0 faces +Y; +H rotates CCW
            curr = getattr(me, "yaw_rad", 0.0)
            me.yaw_rad = _turn_toward(curr, desired_yaw, rate_rad_per_s=math.radians(180), dt=0.016)
            inputs["yaw"] = math.degrees(me.yaw_rad)

        # Move: forward bias; slight lateral nudge only when recently stalled
        inputs["mz"] = 1.0
        if self._stalled(me_xy):
            # small strafe + occasional hop to break wall-hug contact
            inputs["mx"] = random.choice((-0.6, 0.6))
            if getattr(me, "on_ground", True):
                inputs["jump"] = (random.random() < 0.25)

        # Interact near flags (pickup/return handled server-side)
        inputs["interact"] = self._flag_interact_needed(me, gs, mapdata)

        # --- Opportunistic shooting: face nearest enemy and fire when aligned ---
        tgt = self._nearest_enemy(me, gs, max_range=45.0)
        if tgt is not None:
            dx = tgt.x - me.x
            dy = tgt.y - me.y
            desired_yaw = math.atan2(-dx, dy)  # Panda H=0 along +Y
            # turn faster when engaging
            err = abs(_wrap_pi(desired_yaw - me.yaw_rad))
            if err < math.radians(50):
                me.yaw_rad = _turn_toward(me.yaw_rad, desired_yaw,
                                          rate_rad_per_s=math.radians(360), dt=0.016)
                inputs["yaw"] = math.degrees(me.yaw_rad)
            # pull the trigger when reasonably aligned
            if err < math.radians(10):
                inputs["fire"] = True

        return inputs
