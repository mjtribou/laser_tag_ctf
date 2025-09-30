# server.py — Authoritative server with Bullet physics (players, walls, floor, obstacles),
# Bullet ray tests for lasers, safe-spawn, auto-unstick, and shared solid collide-mask fix.
import asyncio, json, math, time, random, argparse, signal
from typing import Dict, Any, List, Tuple, Optional
from bisect import bisect_right

from common.net import read_json, send_json, lan_discovery_server
from game.constants import (
    TEAM_RED, TEAM_BLUE, TEAM_NEUTRAL, MAX_PLAYERS,
    PLAYER_HEIGHT, PLAYER_RADIUS,
    FLAG_PICKUP_RADIUS, FLAG_RETURN_RADIUS, BASE_CAPTURE_RADIUS,
)
from game.map_gen import generate
from game.server_state import GameState, Player, Flag
from game.transform import wrap_pi, deg_to_rad, rad_to_deg, forward_vector, local_move_delta, heading_forward_xy
from game.bot_ai import SimpleBotBrain
from game.ecs import (
    World as ECSWorld,
    CharacterBody,
    CollisionBody,
    FlagState,
    CombatSystem,
    CollisionSystem,
    GrenadeRequest,
    Health,
    HitEvent,
    MovementState,
    MovementSystem,
    Physics,
    PlayerInfo,
    PlayerInput as ECSPlayerInput,
    Position,
    Projectile,
    Weapon,
)

# --- Panda3D / Bullet (headless) ---
from panda3d.core import Vec3, Point3, NodePath, BitMask32, LPoint3
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletBoxShape, BulletCharacterControllerNode, BulletSphereShape
# ---------- Config ----------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

# ---------- Utility ----------
def now() -> float:
    return time.time()

# Shared collide mask for all SOLID things (players + static).
# In Panda3D+Bullet, ONLY the "into" mask is used, and bodies must share a bit to collide.
# See: Panda3D Manual — Bullet Collision Filtering.
MASK_SOLID = BitMask32.bit(0)  # everything physically solid uses this

def aabb_contains(x: float, y: float, z: float, center: Tuple[float,float,float], size: Tuple[float,float,float], margin=0.0) -> bool:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5 - margin, sy * 0.5 - margin, sz * 0.5 - margin
    return (abs(x - cx) <= hx) and (abs(y - cy) <= hy) and (abs(z - cz) <= hz)

# ---------- Server ----------
class LaserTagServer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.gs = GameState()
        self.next_pid = 1

        size_x, size_z = self.cfg["gameplay"]["arena_size_m"]
        cube_cfg = self.cfg.get("cubes", {})
        self.mapdata = generate(seed=42, size_x=size_x, size_z=size_z, cubes=cube_cfg)

        # Single center-flag mode: spawn one neutral flag at arena center
        cx, cy, cz = getattr(self.mapdata, "neutral_flag_stand", (0.0, 0.0, 0.0))
        neutral_flag = Flag(team=TEAM_NEUTRAL, at_base=True, carried_by=None,
                            x=cx, y=cy, z=cz)
        self.gs.flags = {TEAM_NEUTRAL: neutral_flag}

        # networking / gameplay
        self.clients: Dict[int, asyncio.StreamWriter] = {}
        self.inputs: Dict[int, Dict[str, Any]] = {}
        self.bot_brains: Dict[int, SimpleBotBrain] = {}
        self.recent_beams: List[Dict[str, Any]] = []
        self._state_history: List[Tuple[float, Dict[int, Tuple[float,float,float]]]] = []

        # ECS world and entity mappings
        self.ecs = ECSWorld()
        self.pid_to_entity: Dict[int, int] = {}
        self.entity_to_pid: Dict[int, int] = {}
        self.map_entities: List[int] = []
        self.flag_entities: Dict[int, int] = {}

        # ---- Bullet world ----
        self._root = NodePath("world")
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -float(cfg["gameplay"]["gravity"])))

        self._static_nodes: List[NodePath] = []  # floor, obstacles, walls
        self._char_np: Dict[int, NodePath] = {}  # pid -> NodePath (character)
        self._char_node: Dict[int, BulletCharacterControllerNode] = {}
        self._node_id_to_pid: Dict[int, int] = {}  # id(BulletNode) -> pid (for ray hits)
        self._corpse_np: Dict[int, NodePath] = {}
        self._corpse_node: Dict[int, "BulletRigidBodyNode"] = {}
        self._last_hitinfo: Dict[int, Tuple[Tuple[float,float,float], Tuple[float,float,float]]] = {}

        # Grenade state
        self._grenades: Dict[int, Dict[str, Any]] = {}  # gid -> {owner, np, node, explode_at}
        self._next_gid: int = 1
        self._recent_explosions: List[Tuple[float,float,float]] = []

        # Unstick state
        self._last_pos: Dict[int, Tuple[float,float,float]] = {}
        self._stuck_since: Dict[int, float] = {}
        self._last_safe_pos: Dict[int, Tuple[float,float,float]] = {}

        # ECS systems – movement/combat/collision
        self.movement_system = MovementSystem(self.ecs, self.cfg["gameplay"])
        self.combat_system = CombatSystem(self.ecs, self.cfg["gameplay"], self.cfg["server"], self.world, now_fn=now)
        self.collision_system = CollisionSystem(self.ecs, self.cfg["gameplay"])
        self._pre_physics_systems = [self.movement_system, self.combat_system]
        self._post_physics_systems = [self.movement_system, self.collision_system]

        self._build_static_world()
        self._init_flag_entities()

        # rolling list of killfeed events sent to clients
        # each item: {"t","attacker","attacker_name","victim","victim_name","cause"}
        self.killfeed = []
        # rolling list of game message events (e.g., flag pickup/drop/capture)
        # each item: {"t","event","actor","actor_name"}
        self.messagefeed = []

    # ---------- Physics world ----------
    def _attach_static_box(self, center: Tuple[float, float, float], size: Tuple[float, float, float], tag: str):
        hx, hy, hz = size[0]*0.5, size[1]*0.5, size[2]*0.5
        shape = BulletBoxShape(Vec3(hx, hy, hz))
        node = BulletRigidBodyNode(f"static_{tag}")
        node.addShape(shape)
        node.setMass(0.0)
        node.setIntoCollideMask(MASK_SOLID)  # <-- share mask with characters
        np = self._root.attachNewNode(node)
        np.setPos(center[0], center[1], center[2])
        self.world.attachRigidBody(node)
        np.setPythonTag("kind", "static")
        np.setPythonTag("tag", tag)
        self._static_nodes.append(np)
        entity = self.ecs.create_entity()
        self.map_entities.append(entity)
        self.ecs.add_component(entity, Position(x=center[0], y=center[1], z=center[2]))
        self.ecs.add_component(entity, CollisionBody(node=node, nodepath=np, tag=tag))
        return np

    def _build_static_world(self):
        size_x, size_z = self.cfg["gameplay"]["arena_size_m"]

        # Floor slab (top at z=0). Make it thick to be safe.
        self._attach_static_box((0.0, 0.0, -2.5), (size_x, size_z, 5.0), "floor")

        # Procedural blocks → merged static boxes
        count_blocks = 0
        try:
            from game.collider_merge import build_merged_colliders
            merge_mode = str(self.cfg.get("server", {}).get("collider_merge", "stack"))
        except Exception:
            build_merged_colliders = None  # type: ignore
            merge_mode = "none"

        colliders = None
        if build_merged_colliders is not None and merge_mode.lower() != "none":
            try:
                strategy = "level" if merge_mode.lower().startswith("level") else "stack"
                colliders = build_merged_colliders(self.mapdata, strategy=strategy)
            except Exception as e:
                print(f"[server] collider merge failed ({e}); falling back to per-cube")
                colliders = None

        if colliders is None:
            # Fallback: one collider per cube
            for i, b in enumerate(self.mapdata.blocks):
                cx, cy, cz = b.pos
                sx, sy, sz = b.size
                self._attach_static_box((cx, cy, cz), (sx, sy, sz), f"block_{i}")
                count_blocks += 1
            try:
                print(f"[server] static world: {count_blocks} block colliders attached (unmerged)")
            except Exception:
                pass
        else:
            for i, b in enumerate(colliders):
                cx, cy, cz = b.pos
                sx, sy, sz = b.size
                self._attach_static_box((cx, cy, cz), (sx, sy, sz), f"blkM_{i}")
                count_blocks += 1
            try:
                print(f"[server] static world: {count_blocks} merged colliders attached (mode={merge_mode})")
            except Exception:
                pass

        # Arena walls (thin boxes)
        wall_t = 0.5
        wall_h = 5.0
        hx, hz = size_x * 0.5, size_z * 0.5
        self._attach_static_box((-hx - wall_t*0.5, 0.0, wall_h*0.5), (wall_t, size_z, wall_h), "wall_w")
        self._attach_static_box((+hx + wall_t*0.5, 0.0, wall_h*0.5), (wall_t, size_z, wall_h), "wall_e")
        self._attach_static_box((0.0, -hz - wall_t*0.5, wall_h*0.5), (size_x, wall_t, wall_h), "wall_s")
        self._attach_static_box((0.0, +hz + wall_t*0.5, wall_h*0.5), (size_x, wall_t, wall_h), "wall_n")

    def _create_character(self, entity: int, pid: int, pos: Tuple[float, float, float], yaw_rad: float):
        """Create a kinematic character using an **axis-aligned box** (AABB) shape.

        We keep the Bullet *character controller* but swap the shape to a
        BulletBoxShape and ensure we never rotate the Bullet node so the collider
        stays axis-aligned in world space (visuals can still rotate).
        """
        height = float(self.cfg["gameplay"]["player_height"])     # full height
        radius = float(self.cfg["gameplay"]["player_radius"])     # half-width in X/Y

        # Box half-extents: X/Y from radius, Z from half height
        half = Vec3(radius, radius, height * 0.5)
        shape = BulletBoxShape(half)
        # Step height helps climbing small ledges; make configurable
        step_height = float(self.cfg.get("gameplay", {}).get("step_height", 0.4))

        ch = BulletCharacterControllerNode(shape, step_height, f"char_{pid}")
        ch.setGravity(float(self.cfg["gameplay"]["gravity"]))
        ch.setJumpSpeed(float(self.cfg["gameplay"].get("jump_speed", 3.0)))
        ch.setIntoCollideMask(MASK_SOLID)  # collide with static world

        np = self._root.attachNewNode(ch)
        np.setPos(pos[0], pos[1], pos[2])
        # IMPORTANT: do **not** set H on the Bullet node; keep collider axis-aligned
        np.setPythonTag("kind", "player")
        np.setPythonTag("pid", pid)

        self.world.attachCharacter(ch)

        self._char_np[pid] = np
        self._char_node[pid] = ch
        self._node_id_to_pid[id(ch)] = pid

        # Attach/refresh ECS CharacterBody component
        self.ecs.add_component(entity, CharacterBody(controller=ch, nodepath=np))

        self._last_pos[pid] = (pos[0], pos[1], pos[2])
        self._stuck_since[pid] = 0.0
        self._last_safe_pos[pid] = (pos[0], pos[1], pos[2])

    def _remove_character(self, pid: int):
        entity = self.pid_to_entity.get(pid)
        ch = self._char_node.get(pid)
        np = self._char_np.get(pid)
        if ch is not None:
            try:
                self.world.removeCharacter(ch)
            except Exception:
                pass
            self._node_id_to_pid.pop(id(ch), None)
            del self._char_node[pid]
        if np is not None:
            np.removeNode()
            del self._char_np[pid]
        self._last_pos.pop(pid, None)
        self._stuck_since.pop(pid, None)
        self._last_safe_pos.pop(pid, None)
        if entity is not None:
            self.ecs.remove_component(entity, CharacterBody)

    # ---------- Spawn helpers ----------
    def _point_inside_any_block(self, x: float, y: float, z: float, margin: float = 0.02) -> bool:
        for b in self.mapdata.blocks:
            if aabb_contains(x, y, z, b.pos, b.size, margin):
                return True
        return False

    def _find_safe_spawn_near(self, base_xy: Tuple[float,float], tries: int = 30) -> Tuple[float,float,float]:
        bx, by = base_xy
        for r in [0.0, 2.5, 4.0, 5.5, 7.0]:
            for _ in range(max(1, tries // 5)):
                ang = random.uniform(0, 2*math.pi)
                x = bx + math.cos(ang) * r + random.uniform(-0.75, 0.75)
                y = by + math.sin(ang) * r + random.uniform(-0.75, 0.75)
                z = 1.2  # slightly above floor; gravity settles
                if not self._point_inside_any_block(x, y, z, margin=0.05):
                    return (x, y, z)
        return (bx, by, 1.5)

    def _init_flag_entities(self) -> None:
        self.flag_entities.clear()
        for team, flag in self.gs.flags.items():
            entity = self.ecs.create_entity()
            self.flag_entities[team] = entity
            home = self._flag_home_pos(flag)
            self.ecs.add_component(entity, Position(x=flag.x, y=flag.y, z=flag.z))
            self.ecs.add_component(
                entity,
                FlagState(
                    team=team,
                    at_base=flag.at_base,
                    carried_by=flag.carried_by,
                    home_position=home,
                ),
            )

    # ---------- Players / bots ----------
    def assign_spawn(self, team: int) -> Tuple[float, float, float, float]:
        base = self.mapdata.red_base if team == TEAM_RED else self.mapdata.blue_base
        x, y, z = self._find_safe_spawn_near((base[0], base[1]))
        yaw_rad = 0.0 if team == TEAM_RED else math.pi
        return x, y, z, yaw_rad

    def add_player(self, name: str, is_bot: bool = False) -> int:
        red_ct = sum(1 for p in self.gs.players.values() if p.team == TEAM_RED)
        blue_ct = sum(1 for p in self.gs.players.values() if p.team == TEAM_BLUE)
        team = TEAM_RED if red_ct <= blue_ct else TEAM_BLUE

        pid = self.next_pid
        self.next_pid += 1

        x, y, z, yaw_rad = self.assign_spawn(team)
        shots_per_mag = int(self.cfg["gameplay"].get("shots_per_mag", 20))

        entity = self.ecs.create_entity()
        self.pid_to_entity[pid] = entity
        self.entity_to_pid[entity] = pid

        self.ecs.add_component(entity, Position(x=x, y=y, z=z, yaw=yaw_rad, pitch=0.0))
        self.ecs.add_component(entity, Physics(vx=0.0, vy=0.0, vz=0.0, on_ground=True))
        self.ecs.add_component(entity, MovementState())
        self.ecs.add_component(entity, ECSPlayerInput(yaw=math.degrees(yaw_rad), pitch=0.0))
        self.ecs.add_component(entity, PlayerInfo(pid=pid, name=name, team=team, is_bot=is_bot))
        self.ecs.add_component(
            entity,
            Weapon(
                shots_remaining=shots_per_mag,
                shots_per_mag=shots_per_mag,
                reload_end=0.0,
                last_fire_time=0.0,
                cooldown=0.0,
                spread_deg=float(self.cfg["gameplay"].get("base_spread_deg", 1.0)),
            ),
        )
        self.ecs.add_component(entity, Health(hp=1, max_hp=1, alive=True, respawn_at=0.0))

        p = Player(pid=pid, name=name, team=team, is_bot=is_bot, x=x, y=y, z=z, yaw_rad=yaw_rad, pitch_rad=0.0, on_ground=True, shots_remaining=shots_per_mag, reload_end=0.0)
        self.gs.players[pid] = p

        self._create_character(entity, pid, (x, y, z), yaw_rad)
        self._last_safe_pos[pid] = (x, y, z)

        if is_bot:
            base_pos = self.mapdata.red_base if team == TEAM_RED else self.mapdata.blue_base
            enemy_base = self.mapdata.blue_base if team == TEAM_RED else self.mapdata.red_base
            # Use A* navigation brain for bots
            from game.bot_ai import AStarBotBrain
            target_players = bool(self.cfg.get("server", {}).get("bots_target_players", True))
            brain = AStarBotBrain(team, base_pos, enemy_base, target_players=target_players)
            self.bot_brains[pid] = brain

        print(f"[join] pid={pid} name={name} team={'RED' if team==TEAM_RED else 'BLUE'}")
        return pid

    def remove_player(self, pid: int):
        if pid in self.clients:
            del self.clients[pid]
        if pid in self.inputs:
            del self.inputs[pid]
        if pid in self.bot_brains:
            del self.bot_brains[pid]
        self._remove_character(pid)
        entity = self.pid_to_entity.pop(pid, None)
        if entity is not None:
            self.entity_to_pid.pop(entity, None)
            self.ecs.remove_entity(entity)
        if pid in self.gs.players:
            print(f"[leave] pid={pid} name={self.gs.players[pid].name}")
            del self.gs.players[pid]

    def respawn_player(self, pid: int):
        p = self.gs.players.get(pid)
        if not p:
            return

        # Remove any corpse body
        self._remove_corpse(pid)

        # Choose a safe spawn and reset state
        x, y, z, yaw_rad = self.assign_spawn(p.team)
        p.x, p.y, p.z = x, y, z
        p.yaw_rad = yaw_rad
        p.pitch_rad = 0.0
        p.alive = True
        p.respawn_at = 0.0
        p.on_ground = True
        p.crouching = False
        p.walking = False
        p.shots_remaining = int(self.cfg["gameplay"].get("shots_per_mag", 20))
        p.reload_end = 0.0
        p.recoil_accum = 0.0

        entity = self.pid_to_entity.get(pid)
        if entity is not None:
            pos_comp = self.ecs.get_component(entity, Position)
            if pos_comp:
                pos_comp.x = x
                pos_comp.y = y
                pos_comp.z = z
                pos_comp.yaw = yaw_rad
                pos_comp.pitch = 0.0
            phys_comp = self.ecs.get_component(entity, Physics)
            if phys_comp:
                phys_comp.vx = phys_comp.vy = phys_comp.vz = 0.0
                phys_comp.on_ground = True
            move_state = self.ecs.get_component(entity, MovementState)
            if move_state:
                move_state.walking = False
                move_state.crouching = False
            input_comp = self.ecs.get_component(entity, ECSPlayerInput)
            if input_comp:
                input_comp.yaw = math.degrees(yaw_rad)
                input_comp.pitch = 0.0
                input_comp.mx = 0.0
                input_comp.mz = 0.0
                input_comp.fire = False
                input_comp.jump = False
                input_comp.walk = False
                input_comp.crouch = False
            weapon_comp = self.ecs.get_component(entity, Weapon)
            if weapon_comp:
                weapon_comp.shots_remaining = weapon_comp.shots_per_mag
                weapon_comp.reload_end = 0.0
                weapon_comp.recoil_accum = 0.0
            health_comp = self.ecs.get_component(entity, Health)
            if health_comp:
                health_comp.hp = max(1, health_comp.max_hp or 1)
                health_comp.alive = True
                health_comp.respawn_at = 0.0

        self._last_safe_pos[pid] = (x, y, z)

        # Recreate character controller if missing, else just move it
        body = self.ecs.get_component(entity, CharacterBody) if entity is not None else None
        if body is None or body.nodepath is None:
            self._create_character(entity or self.ecs.create_entity(), pid, (x, y, z), yaw_rad)
        else:
            body.nodepath.setPos(x, y, z)
            try:
                body.controller.setLinearMovement(Vec3(0, 0, 0), False)
            except Exception:
                pass

    # ---------- Game mechanics ----------
    def _flag_home_pos(self, fl: Flag) -> Tuple[float, float, float]:
        """Return the home stand position for a flag (team or neutral)."""
        if fl.team == TEAM_RED:
            return self.mapdata.red_flag_stand
        if fl.team == TEAM_BLUE:
            return self.mapdata.blue_flag_stand
        # Neutral / single-flag
        return getattr(self.mapdata, "neutral_flag_stand", (0.0, 0.0, 0.0))

    def _drop_flag(self, team_flag: int, x: float, y: float, z: float):
        fl = self.gs.flags[team_flag]
        fl.carried_by = None
        fl.at_base = False
        fl.x, fl.y, fl.z = x, y, max(0.0, z)
        fl.dropped_at_time = now()

    def _pickup_try(self, p: Player):
        for team_flag, fl in self.gs.flags.items():
            # Returning own flag (legacy two-flag CTF) — not applicable for neutral flag
            if team_flag in (TEAM_RED, TEAM_BLUE) and team_flag == p.team:
                d = math.hypot(p.x - fl.x, p.y - fl.y)
                if (not fl.at_base) and fl.carried_by is None and d <= FLAG_RETURN_RADIUS:
                    fl.at_base = True
                    fx, fy, fz = self._flag_home_pos(fl)
                    fl.x, fl.y, fl.z = fx, fy, fz
                    fl.dropped_at_time = 0.0
                    p.defences += 1
                continue

            if fl.carried_by is not None:
                continue
            if fl.at_base:
                fx, fy, fz = self._flag_home_pos(fl)
            else:
                fx, fy, fz = fl.x, fl.y, fl.z
            d = math.hypot(p.x - fx, p.y - fy)
            if d <= FLAG_PICKUP_RADIUS:
                fl.carried_by = p.pid
                fl.at_base = False
                fl.x, fl.y, fl.z = p.x, p.y, p.z + 0.8
                # Track on player so bots/clients can react
                try:
                    p.carrying_flag = team_flag
                except Exception:
                    pass
                # Log message feed: picked up flag
                try:
                    self.messagefeed.append({
                        "t": now(),
                        "event": "pickup",
                        "actor": p.pid,
                        "actor_name": p.name,
                    })
                except Exception:
                    pass

    def _carry_flags_update(self):
        for team_flag, fl in self.gs.flags.items():
            if fl.carried_by and fl.carried_by in self.gs.players:
                carrier = self.gs.players[fl.carried_by]
                fl.x, fl.y, fl.z = carrier.x, carrier.y, carrier.z + 0.8
                # Keep player's carrying_flag in sync (helps bot behavior)
                if getattr(carrier, "carrying_flag", None) != team_flag:
                    carrier.carrying_flag = team_flag

    def _dropped_flags_auto_return(self):
        ttl = float(self.cfg["server"]["flag_return_seconds"])
        t = now()
        for fl in self.gs.flags.values():
            if (not fl.at_base) and (fl.carried_by is None) and fl.dropped_at_time > 0.0:
                if (t - fl.dropped_at_time) >= ttl:
                    fl.at_base = True
                    fl.dropped_at_time = 0.0
                    fx, fy, fz = self._flag_home_pos(fl)
                    fl.x, fl.y, fl.z = fx, fy, fz

    def _check_captures(self):
        to_win = int(self.cfg["server"]["captures_to_win"])
        for team_id in (TEAM_RED, TEAM_BLUE):
            if self.gs.teams[team_id].captures >= to_win:
                self.gs.match_over = True
                self.gs.winner = team_id

        for pid, p in list(self.gs.players.items()):
            if not p.alive:
                continue
            # Single-flag mode (neutral flag)
            if len(self.gs.flags) == 1 and (TEAM_NEUTRAL in self.gs.flags):
                fl = next(iter(self.gs.flags.values()))
                if fl.carried_by != pid:
                    continue
                bx, by, bz = (self.mapdata.red_base if p.team == TEAM_RED else self.mapdata.blue_base)
                if math.hypot(p.x - bx, p.y - by) <= BASE_CAPTURE_RADIUS:
                    self.gs.teams[p.team].captures += 1
                    fl.carried_by = None
                    p.captures += 1
                    # Clear carrier flag
                    p.carrying_flag = None
                    # Return neutral flag to center
                    fl.at_base = True
                    fx, fy, fz = self._flag_home_pos(fl)
                    fl.x, fl.y, fl.z = fx, fy, fz
                    print(f"[score] Team {'RED' if p.team==TEAM_RED else 'BLUE'} captured! -> {self.gs.teams[p.team].captures}")
                    # Log message feed: capture
                    try:
                        self.messagefeed.append({
                            "t": now(),
                            "event": "capture",
                            "actor": p.pid,
                            "actor_name": p.name,
                        })
                    except Exception:
                        pass
                continue

            # Legacy two-flag mode fallback
            enemy_flag_team = TEAM_BLUE if p.team == TEAM_RED else TEAM_RED
            fl = self.gs.flags.get(enemy_flag_team)
            if not fl or fl.carried_by != pid:
                continue
            bx, by, bz = (self.mapdata.red_base if p.team == TEAM_RED else self.mapdata.blue_base)
            if math.hypot(p.x - bx, p.y - by) <= BASE_CAPTURE_RADIUS:
                self.gs.teams[p.team].captures += 1
                fl.carried_by = None
                p.captures += 1
                p.carrying_flag = None
                fl.at_base = True
                fx, fy, fz = self._flag_home_pos(fl)
                fl.x, fl.y, fl.z = fx, fy, fz
                print(f"[score] Team {'RED' if p.team==TEAM_RED else 'BLUE'} captured! -> {self.gs.teams[p.team].captures}")
                # Log message feed: capture
                try:
                    self.messagefeed.append({
                        "t": now(),
                        "event": "capture",
                        "actor": p.pid,
                        "actor_name": p.name,
                    })
                except Exception:
                    pass

    # ---------- Lag-comp history ----------
    def _record_history(self):
        snap = {pid: (p.x, p.y, p.z) for pid, p in self.gs.players.items()}
        t = now()
        self._state_history.append((t, snap))
        # keep ~1s of history
        while self._state_history and t - self._state_history[0][0] > 1.0:
            self._state_history.pop(0)

    def _positions_at(self, t: float) -> Dict[int, Tuple[float, float, float]]:
        if not self._state_history:
            return {pid: (p.x, p.y, p.z) for pid, p in self.gs.players.items()}
        times = [ts for ts, _ in self._state_history]
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return dict(self._state_history[0][1])
        if idx >= len(self._state_history) - 1:
            return dict(self._state_history[-1][1])
        t0, s0 = self._state_history[idx]
        t1, s1 = self._state_history[idx + 1]
        if t1 <= t0:
            return dict(s1)
        a = (t - t0) / (t1 - t0)
        pos = {}
        keys = set(s0.keys()) | set(s1.keys())
        for pid in keys:
            p0 = s0.get(pid)
            p1 = s1.get(pid)
            if p0 and p1:
                pos[pid] = (
                    (1 - a) * p0[0] + a * p1[0],
                    (1 - a) * p0[1] + a * p1[1],
                    (1 - a) * p0[2] + a * p1[2],
                )
            else:
                pos[pid] = p1 or p0
        return pos

    # ---------- Firing / hitscan ----------
    def _add_beam(self, team: int, start: Tuple[float,float,float], end: Tuple[float,float,float], shooter_pid: int = None):
        sx, sy, sz = start
        ex, ey, ez = end
        dx, dy, dz = (ex - sx), (ey - sy), (ez - sz)
        length = math.sqrt(dx*dx + dy*dy + dz*dz)

        self.recent_beams.append({
            "team": team,
            "sx": sx, "sy": sy, "sz": sz,
            "ex": ex, "ey": ey, "ez": ez,
            "len": length,          # <= for client projectile animation
            "t": now(),             # shot start time
            "owner": shooter_pid,
        })

    def _apply_hitscan(self, shooter: "Player", spread_rad: float, fire_time: Optional[float] = None):
        """
        Fire one hitscan ray with cone spread.
        Returns (victim_pid, hit_point_xyz, dir_xyz) or (None, None, None).
        Also ALWAYS enqueues a beam for the client to draw.
        """
        import math, random

        positions = self._positions_at(fire_time) if fire_time is not None else {pid: (p.x, p.y, p.z) for pid, p in self.gs.players.items()}

        # Shooter eye position (align with client camera)
        ph = float(self.cfg["gameplay"]["player_height"])
        sx, sy, sz = positions.get(shooter.pid, (shooter.x, shooter.y, shooter.z))
        sx, sy, sz = sx, sy, sz + 0.30 * ph  # was +0.50

        fx, fy, fz = forward_vector(shooter.yaw_rad, shooter.pitch_rad)

        # Random spread within a small cone (no numpy needed)
        if spread_rad and spread_rad > 1e-6:
            # make a simple ONB around f
            rx, ry, rz = (-fy, fx, 0.0)  # right ≈ perpendicular to f in XY
            rlen = math.sqrt(rx*rx + ry*ry + rz*rz) or 1.0
            rx, ry, rz = rx/rlen, ry/rlen, 0.0
            # up = right × f
            ux, uy, uz = (ry*fz - rz*fy, rz*fx - rx*fz, rx*fy - ry*fx)
            ulen = math.sqrt(ux*ux + uy*uy + uz*uz) or 1.0
            ux, uy, uz = ux/ulen, uy/ulen, uz/ulen

            a = random.uniform(0.0, 2.0 * math.pi)
            r = (random.uniform(0.0, 1.0) ** 0.5) * spread_rad
            ca, sa, cr, sr = math.cos(a), math.sin(a), math.cos(r), math.sin(r)
            fx, fy, fz = (
                fx*cr + sr*(rx*ca + ux*sa),
                fy*cr + sr*(ry*ca + uy*sa),
                fz*cr + sr*(rz*ca + uz*sa),
            )

        # Ray end
        max_range = float(self.cfg["gameplay"].get("laser_range_m", 80.0))
        ex, ey, ez = sx + fx*max_range, sy + fy*max_range, sz + fz*max_range

        Dx, Dy, Dz = (ex - sx), (ey - sy), (ez - sz)
        Dlen2 = Dx*Dx + Dy*Dy + Dz*Dz
        if Dlen2 <= 1e-12:
            self._add_beam(shooter.team, (sx, sy, sz - 0.05 * ph), (ex, ey, ez), shooter_pid=shooter.pid)
            return None, None, (fx, fy, fz)

        # ---- Bullet wall-occlusion: first static block along the ray ----
        # We filter hits to NodePaths tagged kind="static" by _attach_static_box()
        t_wall = None
        wall_point = None
        EPS_HIT = 1e-4
        try:
            res = self.world.rayTestAll(LPoint3(sx, sy, sz), LPoint3(ex, ey, ez))
            # Find the closest STATIC hit strictly after the start (avoid self-grazes)
            for hit in res.getHits():
                np_hit = NodePath(hit.getNode())
                if np_hit.is_empty():
                    continue
                if np_hit.getPythonTag("kind") != "static":
                    continue  # ignore players, dynamic bodies, etc.
                frac = float(hit.getHitFraction())
                if frac <= EPS_HIT:
                    continue
                if (t_wall is None) or (frac < t_wall):
                    t_wall = frac
                    wall_point = (sx + Dx*frac, sy + Dy*frac, sz + Dz*frac)
        except Exception:
            # If Bullet isn't available for some reason, we just treat as no wall
            pass

        # ---- Player AABB hit test (kept from your version) ----
        friendly_fire = bool(self.cfg["gameplay"].get("friendly_fire", False))
        r = float(self.cfg["gameplay"]["player_radius"])
        hz = 0.5 * ph  # half-height
        eps = 0.0
        hx, hy, hz = max(0.0, r - eps), max(0.0, r - eps), max(0.0, hz - eps)

        def hit_t_for_aabb(cx, cy, cz, hx, hy, hz):
            # Slab method: find intersection interval t∈[0,1] across X,Y,Z slabs
            tmin, tmax = 0.0, 1.0
            for S, D, C, H in ((sx, Dx, cx, hx), (sy, Dy, cy, hy), (sz, Dz, cz, hz)):
                if abs(D) < 1e-8:
                    if abs(S - C) > H:
                        return None  # ray parallel & outside slab
                    continue
                invD = 1.0 / D
                t1 = ((C - H) - S) * invD
                t2 = ((C + H) - S) * invD
                if t1 > t2: t1, t2 = t2, t1
                if t1 > tmin: tmin = t1
                if t2 < tmax: tmax = t2
                if tmax < tmin: return None
            if tmax < 0.0 or tmin > 1.0:
                return None
            return max(tmin, 0.0)

        victim_pid, t_player = None, None
        for pid, v in self.gs.players.items():
            if pid == shooter.pid:               continue
            if not v.alive:                      continue
            if (not friendly_fire) and (v.team == shooter.team):  continue
            vx, vy, vz = positions.get(pid, (v.x, v.y, v.z))
            t = hit_t_for_aabb(vx, vy, vz, hx, hy, hz)
            if t is not None and (t_player is None or t < t_player):
                t_player, victim_pid = t, pid

        # ---- Decide outcome with occlusion ----
        if (t_player is not None) and ((t_wall is None) or (t_player <= t_wall - EPS_HIT)):
            # Player is closer than any wall -> register hit
            hit_point = (sx + Dx*t_player, sy + Dy*t_player, sz + Dz*t_player)
            beam_end  = hit_point
        else:
            # Blocked by wall or no player -> stop beam on the wall or go full range
            hit_point = None
            beam_end  = wall_point if wall_point is not None else (ex, ey, ez)
            victim_pid = None

        # Always enqueue a beam so the client can draw it
        self._add_beam(shooter.team, (sx, sy, sz - 0.05 * ph), beam_end, shooter_pid=shooter.pid)

        # Return normalized direction (already nearly unit, but normalize anyway)
        dlen = math.sqrt(fx*fx + fy*fy + fz*fz) or 1.0
        return victim_pid, hit_point, (fx/dlen, fy/dlen, fz/dlen)

    
    def _spawn_grenade(self, owner: Player, power: float):
        gid = self._next_gid
        self._next_gid += 1
        radius = 0.2
        shape = BulletSphereShape(radius)
        node = BulletRigidBodyNode(f"grenade-{gid}")
        node.setMass(1.0)
        node.addShape(shape)
        node.setFriction(0.5)
        node.setRestitution(0.6)
        node.setIntoCollideMask(MASK_SOLID)
        np = self._root.attachNewNode(node)
        # Spawn at eye height and slightly forward to avoid intersecting the player collider.
        ph = float(self.cfg["gameplay"]["player_height"])  # full height
        pr = float(self.cfg["gameplay"].get("player_radius", 0.5))
        eye_z = owner.z + 0.30 * ph
        # Offset forward in the XY plane (independent of pitch) past player + grenade radius.
        off_xy = (pr + radius + 0.05)
        hx, hy = heading_forward_xy(owner.yaw_rad)
        spawn_x = owner.x + hx * off_xy
        spawn_y = owner.y + hy * off_xy
        spawn_z = eye_z
        np.setPos(spawn_x, spawn_y, spawn_z)
        self.world.attachRigidBody(node)
        fx, fy, fz = forward_vector(owner.yaw_rad, owner.pitch_rad)
        base = float(self.cfg["gameplay"].get("grenade_throw_speed", 15.0))
        max_charge = float(self.cfg["gameplay"].get("grenade_max_charge", 1.5))
        speed = base * min(1.0, power / max_charge)
        owner_vel = Vec3(owner.vx, owner.vy, owner.vz)
        node.setLinearVelocity(Vec3(fx*speed, fy*speed, fz*speed + float(self.cfg["gameplay"].get("grenade_lob_speed", 5.0))) + owner_vel)
        fuse = float(self.cfg["gameplay"].get("grenade_fuse", 3.0))
        entity = self.ecs.create_entity()
        self.ecs.add_component(entity, Position(x=spawn_x, y=spawn_y, z=spawn_z))
        self.ecs.add_component(
            entity,
            Projectile(
                owner_pid=owner.pid,
                speed=speed,
                damage=float(self.cfg["gameplay"].get("grenade_damage", 0.0)),
                ttl=fuse,
                spawn_time=now(),
            ),
        )
        self._grenades[gid] = {"owner": owner.pid, "np": np, "node": node, "explode_at": now() + fuse, "entity": entity}
        print(f'[grenade] spawn gid={gid} owner={owner.pid} power={power:.2f}')

    def _update_grenades(self):
        radius = float(self.cfg["gameplay"].get("grenade_radius", 5.0))
        tnow = now()
        remove = []
        for gid, g in list(self._grenades.items()):
            if tnow >= g["explode_at"]:
                pos = g["np"].getPos()
                for pid, pl in self.gs.players.items():
                    if not pl.alive or pid == g["owner"]:
                        continue
                    thrower = self.gs.players.get(g["owner"])
                    friendly_fire = bool(self.cfg["server"].get("friendly_fire", False))
                    if thrower and pl.team == thrower.team and not friendly_fire:
                        continue
                    dx, dy, dz = pl.x - pos.x, pl.y - pos.y, pl.z - pos.z
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    if dist <= radius:
                        k = float(self.cfg["gameplay"].get("grenade_knockback", 3.0))
                        self._handle_player_death(pid, attacker=g["owner"], hit_point=(pl.x, pl.y, pl.z), shot_dir=(dx, dy, dz), cause="grenade", impulse_mult=k)
                        print(f'[grenade] kill gid={gid} victim={pid} by={g["owner"]}')
                self._recent_explosions.append((float(pos.x), float(pos.y), float(pos.z)))
                self.world.removeRigidBody(g["node"])
                g["np"].removeNode()
                remove.append(gid)
        for gid in remove:
            entity = self._grenades[gid].get("entity")
            if entity is not None:
                self.ecs.remove_entity(entity)
            del self._grenades[gid]
# ---------- Input & per-tick ----------
    def _sync_inputs_to_ecs(self):
        now_t = now()
        for pid, entity in list(self.pid_to_entity.items()):
            comp = self.ecs.get_component(entity, ECSPlayerInput)
            if comp is None:
                comp = ECSPlayerInput()
                self.ecs.add_component(entity, comp)

            data = self.inputs.get(pid, {})
            comp.mx = float(data.get("mx", 0.0))
            comp.mz = float(data.get("mz", 0.0))
            comp.yaw = float(data.get("yaw", comp.yaw))
            comp.pitch = float(data.get("pitch", comp.pitch))
            comp.fire = bool(data.get("fire", False))
            comp.fire_timestamp = float(data.get("fire_t", now_t))
            comp.jump = bool(data.get("jump", False))
            comp.crouch = bool(data.get("crouch", False))
            comp.walk = bool(data.get("walk", False))
            comp.grenade_charge = float(data.get("grenade", 0.0))

    def _sync_ecs_to_gamestate(self):
        for pid, entity in self.pid_to_entity.items():
            player = self.gs.players.get(pid)
            if player is None:
                continue

            pos = self.ecs.get_component(entity, Position)
            phys = self.ecs.get_component(entity, Physics)
            move_state = self.ecs.get_component(entity, MovementState)
            health = self.ecs.get_component(entity, Health)
            weapon = self.ecs.get_component(entity, Weapon)

            if pos:
                player.x = pos.x
                player.y = pos.y
                player.z = pos.z
                player.yaw_rad = pos.yaw
                player.pitch_rad = pos.pitch
            if phys:
                player.vx = phys.vx
                player.vy = phys.vy
                player.vz = phys.vz
                player.on_ground = phys.on_ground
            if move_state:
                player.walking = move_state.walking
                player.crouching = move_state.crouching
            if health:
                player.alive = health.alive
                player.respawn_at = health.respawn_at
            if weapon:
                player.shots_remaining = weapon.shots_remaining
                player.reload_end = weapon.reload_end
                player.recoil_accum = weapon.recoil_accum

    def _sync_flags_to_components(self) -> None:
        for team, flag in self.gs.flags.items():
            entity = self.flag_entities.get(team)
            if entity is None:
                continue
            pos_comp = self.ecs.get_component(entity, Position)
            if pos_comp:
                pos_comp.x = flag.x
                pos_comp.y = flag.y
                pos_comp.z = flag.z
            flag_comp = self.ecs.get_component(entity, FlagState)
            if flag_comp:
                flag_comp.at_base = flag.at_base
                flag_comp.carried_by = flag.carried_by

    def _process_system_events(self) -> None:
        if self.combat_system.beam_events:
            self.recent_beams.extend(self.combat_system.beam_events)

        for req in self.combat_system.grenade_requests:
            owner = self.gs.players.get(req.owner_pid)
            if owner is None:
                continue
            self._spawn_grenade(owner, req.power)
            if req.owner_pid in self.inputs:
                self.inputs[req.owner_pid]["grenade"] = 0.0

        for event in self.combat_system.hit_events:
            victim_pid = self.entity_to_pid.get(event.victim_entity)
            if victim_pid is None:
                continue
            attacker_pid = self.entity_to_pid.get(event.attacker_entity) if event.attacker_entity is not None else None
            self._handle_player_death(
                victim_pid,
                attacker=attacker_pid,
                hit_point=event.hit_point,
                shot_dir=event.shot_dir,
                cause=event.cause,
            )

    def _handle_player_death(self, victim_pid: int, attacker: Optional[int] = None,
                             hit_point: Optional[Tuple[float,float,float]] = None,
                             shot_dir: Optional[Tuple[float,float,float]] = None,
                             cause: str = "beam",
                             impulse_mult: float = 1.0):
        v = self.gs.players.get(victim_pid)
        if not v:
            return

        entity = self.pid_to_entity.get(victim_pid)
        pos_comp = self.ecs.get_component(entity, Position) if entity is not None else None
        health_comp = self.ecs.get_component(entity, Health) if entity is not None else None
        if health_comp:
            health_comp.alive = False
            health_comp.hp = 0
            health_comp.respawn_at = now() + float(self.cfg["server"]["respawn_seconds"])

        px = pos_comp.x if pos_comp else v.x
        py = pos_comp.y if pos_comp else v.y
        pz = pos_comp.z if pos_comp else v.z
        v.x, v.y, v.z = px, py, pz

        # Stat updates
        v.outs += 1
        if attacker is not None and attacker in self.gs.players:
            self.gs.players[attacker].tags += 1

        # Mark dead + schedule respawn (unchanged)
        v.alive = False
        v.respawn_at = now() + float(self.cfg["server"]["respawn_seconds"])

        # --- Killfeed (unchanged) ---
        try:
            a = self.gs.players.get(attacker) if attacker is not None else None
            evt = {
                "t": now(),
                "attacker": attacker,
                "attacker_name": (a.name if a else "World"),
                "victim": victim_pid,
                "victim_name": v.name,
                "cause": cause,
            }
            self.killfeed.append(evt)
            max_keep = int(self.cfg.get("hud", {}).get("killfeed_max", 6)) * 3
            max_keep = max(12, max_keep)
            if len(self.killfeed) > max_keep:
                self.killfeed = self.killfeed[-max_keep:]
        except Exception as e:
            print(e)

        # Stop any character movement immediately
        ch = self._char_node.get(victim_pid)
        if ch is not None:
            try:
                ch.setLinearMovement(Vec3(0.0, 0.0, 0.0), True)
            except Exception:
                pass

        # Drop any carried flag where they are and clear player carry state
        for team_flag, fl in self.gs.flags.items():
            if fl.carried_by == victim_pid:
                self._drop_flag(team_flag, px, py, pz)
                try:
                    v.carrying_flag = None
                except Exception:
                    pass
                # Log message feed: dropped flag
                try:
                    self.messagefeed.append({
                        "t": now(),
                        "event": "drop",
                        "actor": victim_pid,
                        "actor_name": v.name,
                    })
                except Exception:
                    pass

        # === NEW: swap KCC -> dynamic rigid body "corpse" ===
        # Remove character controller from world to avoid double-collision
        self._remove_character(victim_pid)

        pr = float(self.cfg["gameplay"]["player_radius"])
        ph = float(self.cfg["gameplay"]["player_height"])
        rag = self.cfg.get("ragdoll", {})
        mass = float(rag.get("mass_kg", 75.0))
        fric = float(rag.get("friction", 0.9))
        rest = float(rag.get("restitution", 0.05))
        ldmp = float(rag.get("linear_damping", 0.04))
        admp = float(rag.get("angular_damping", 0.05))
        Jmag = float(rag.get("knockback_impulse", 140.0)) * impulse_mult  # N·s

        # Box roughly matching player hull (axis-aligned for simplicity)
        shape = BulletBoxShape(Vec3(pr, pr, 0.5 * ph))
        rb = BulletRigidBodyNode(f"corpse-{victim_pid}")
        rb.setMass(mass)
        rb.addShape(shape)
        rb.setFriction(fric)
        rb.setRestitution(rest)
        rb.setLinearDamping(ldmp)
        rb.setAngularDamping(admp)
        rb.setIntoCollideMask(MASK_SOLID)

        np = self._root.attachNewNode(rb)
        np.setPos(px, py, pz)  # v.z is the player center in this codebase
        np.setPythonTag("kind", "corpse")
        np.setPythonTag("pid", victim_pid)

        self.world.attachRigidBody(rb)
        self._corpse_np[victim_pid] = np
        self._corpse_node[victim_pid] = rb

        # Apply impulse at impact point to blast backward
        if hit_point is not None and shot_dir is not None:
            # Blast AWAY from the shooter (along shot_dir), not toward them
            dx, dy, dz = shot_dir
            L = max(1e-6, math.sqrt(dx*dx + dy*dy + dz*dz))
            jx, jy, jz = (dx / L * Jmag, dy / L * Jmag, dz / L * Jmag)

            # Lever arm from COM to the actual hit point (world space)
            rel = Point3(hit_point[0], hit_point[1], hit_point[2]) - np.getPos()

            # Apply linear impulse at contact offset; Bullet induces spin via r × J
            rb.applyImpulse(Vec3(jx, jy, jz), rel)

            # Base torque from off-center hit
            tau = Vec3(rel).cross(Vec3(jx, jy, jz))

            # Add a small twist perpendicular to the beam to guarantee visible rotation
            axis = Vec3(dx, dy, dz).cross(Vec3(0, 0, 1))
            if axis.length_squared() < 1e-6:
                axis = Vec3(1, 0, 0)
            axis.normalize()

            rb.applyTorqueImpulse(tau)
        else:
            # (keep your existing fallback block)
            ax, ay, az = (self.gs.players[attacker].x, self.gs.players[attacker].y, self.gs.players[attacker].z) if attacker in self.gs.players else (v.x, v.y - 1.0, v.z)
            dx, dy, dz = (v.x - ax, v.y - ay, max(0.2, v.z - az))
            L = max(1e-6, math.sqrt(dx*dx + dy*dy + dz*dz))
            rb.applyImpulse(Vec3(dx / L * Jmag, dy / L * Jmag, dz / L * Jmag), Point3(0, 0, 0))
            spin = 0.25 * Jmag
            rb.applyTorqueImpulse(Vec3(
                random.uniform(-spin, spin),
                random.uniform(-spin, spin),
                random.uniform(-spin, spin)
            ))

    def _auto_unstick(self, dt: float):
        for pid, p in self.gs.players.items():
            if not p.alive:
                self._stuck_since[pid] = 0.0
                continue

            np = self._char_np.get(pid)
            if np is None:
                continue

            inp = self.inputs.get(pid, {})
            intent_mag = min(1.0, math.hypot(float(inp.get("mx", 0.0)), float(inp.get("mz", 0.0))))

            prev = self._last_pos.get(pid, (p.x, p.y, p.z))
            dx, dy, dz = p.x - prev[0], p.y - prev[1], p.z - prev[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            self._last_pos[pid] = (p.x, p.y, p.z)

            if not self._point_inside_any_block(p.x, p.y, p.z, margin=0.05):
                self._last_safe_pos[pid] = (p.x, p.y, p.z)
                self._stuck_since[pid] = 0.0
                continue

            t = now()
            if intent_mag > 0.1 and dist < 0.005:
                if self._stuck_since.get(pid, 0.0) == 0.0:
                    self._stuck_since[pid] = t
                elif (t - self._stuck_since[pid]) > 0.6:
                    np.setPos(p.x + random.uniform(-0.15, 0.15),
                              p.y + random.uniform(-0.15, 0.15),
                              p.z + 0.6)
                    self._stuck_since[pid] = t + 1e-3
            elif self._point_inside_any_block(p.x, p.y, p.z, margin=0.05):
                safe = self._last_safe_pos.get(pid)
                if safe:
                    np.setPos(safe[0], safe[1], max(0.9, safe[2]))
                else:
                    x, y, z, _ = self.assign_spawn(p.team)
                    np.setPos(x, y, z)
                self._stuck_since[pid] = 0.0

    def _after_physics_sync(self):
        tick_dt = 1.0 / float(self.cfg["server"]["tick_hz"])

        # Alive players: sync from character controller nodes and record velocity
        for pid, p in self.gs.players.items():
            np = self._char_np.get(pid)
            if np is not None:
                pos = np.getPos()
                p.vx = (float(pos.x) - p.x) / tick_dt
                p.vy = (float(pos.y) - p.y) / tick_dt
                p.vz = (float(pos.z) - p.z) / tick_dt
                p.x, p.y, p.z = float(pos.x), float(pos.y), float(pos.z)
                try:
                    p.on_ground = bool(self._char_node[pid].isOnGround())
                except Exception:
                    p.on_ground = p.z <= 0.01

        # Dead players: sync from their ragdoll rigid body if present
        for pid, p in self.gs.players.items():
            if p.alive:
                continue
            np = self._corpse_np.get(pid)
            if np is None:
                continue
            pos = np.getPos()
            p.vx = (float(pos.x) - p.x) / tick_dt
            p.vy = (float(pos.y) - p.y) / tick_dt
            p.vz = (float(pos.z) - p.z) / tick_dt
            p.x, p.y, p.z = float(pos.x), float(pos.y), float(pos.z)

        # Lightweight player-player separation for KCCs
        alive_pids = [pid for pid, pp in self.gs.players.items() if pp.alive]
        r = float(self.cfg["gameplay"]["player_radius"])
        min_d = 2.0 * r * 0.98
        for i in range(len(alive_pids)):
            for j in range(i + 1, len(alive_pids)):
                pa = self.gs.players[alive_pids[i]]
                pb = self.gs.players[alive_pids[j]]
                dx = pb.x - pa.x
                dy = pb.y - pa.y
                d2 = dx * dx + dy * dy
                if d2 < (min_d * min_d) and d2 > 1e-6:
                    d = math.sqrt(d2)
                    push = 0.5 * (min_d - d)
                    ux, uy = dx / d, dy / d
                    npa, npb = self._char_np[pa.pid], self._char_np[pb.pid]
                    npa.setPos(npa.getX() - ux * push, npa.getY() - uy * push, npa.getZ())
                    npb.setPos(npb.getX() + ux * push, npb.getY() + uy * push, npb.getZ())

    def _remove_corpse(self, pid: int):
        rb = self._corpse_node.pop(pid, None)
        np = self._corpse_np.pop(pid, None)
        if rb is not None:
            try:
                self.world.removeRigidBody(rb)
            except Exception:
                pass
        if np is not None:
            np.removeNode()

    # ---------- Snapshots ----------
    def _trim_old_beams(self):
        vis_cfg = self.cfg.get("laser_visual", {})
        speed = float(vis_cfg.get("projectile_speed_mps", 180.0))
        fade  = float(vis_cfg.get("fadeout_s", 0.05))

        now_t = now()
        alive = []
        for b in self.recent_beams:
            # compute length if older servers haven't added "len" yet
            L = b.get("len")
            if L is None:
                dx = b["ex"] - b["sx"]; dy = b["ey"] - b["sy"]; dz = b["ez"] - b["sz"]
                L = math.sqrt(dx*dx + dy*dy + dz*dz)
            ttl = (L / max(1e-6, speed)) + fade
            if (now_t - b["t"]) <= ttl:
                alive.append(b)
        self.recent_beams = alive

    def build_snapshot(self) -> Dict[str, Any]:
        self._trim_old_beams()
        now_t = now()
        players = []
        for p in self.gs.players.values():
            d = {
                "pid": p.pid, "name": p.name, "team": p.team,
                "x": p.x, "y": p.y, "z": p.z,
                "alive": p.alive,
                "shots": p.shots_remaining,
                "reload": max(0.0, p.reload_end - now_t),
                "tags": p.tags, "outs": p.outs, "captures": p.captures, "defences": p.defences, "ping": int(p.ping_ms)
            }
            if p.alive:
                d["yaw"]   = rad_to_deg(p.yaw_rad)
                d["pitch"] = rad_to_deg(p.pitch_rad)
                d["roll"]  = 0.0
            else:
                np = self._corpse_np.get(p.pid)
                if np is not None:
                    h, pu, r = np.getHpr()  # degrees
                    d["yaw"], d["pitch"], d["roll"] = float(h), float(pu), float(r)
                else:
                    # fallback to last commanded angles if corpse NP is missing
                    d["yaw"] = rad_to_deg(p.yaw_rad)
                    d["pitch"] = rad_to_deg(p.pitch_rad)
                    d["roll"] = 0.0
            players.append(d)

        flags = []
        for fl in self.gs.flags.values():
            flags.append({
                "team": fl.team, "at_base": fl.at_base, "carried_by": fl.carried_by,
                "x": fl.x, "y": fl.y, "z": fl.z
            })
        
        grenades = []
        for gid, g in self._grenades.items():
            pos = g['np'].getPos()
            player = self.gs.players.get(g["owner"])
            team = player.team if player else TEAM_RED
            grenades.append({"id": gid, "team": team, "x": float(pos.x), "y": float(pos.y), "z": float(pos.z)})

        explosions = [{"x": x, "y": y, "z": z} for (x, y, z) in self._recent_explosions]
        self._recent_explosions.clear()
        now_t = now()
        hud_cfg = self.cfg.get("hud", {})
        ttl = float(hud_cfg.get("killfeed_ttl", 4.0))
        kmax = int(hud_cfg.get("killfeed_max", 6))
        feed = [e for e in self.killfeed if (now_t - float(e.get("t", 0.0))) <= ttl]
        feed = feed[-kmax:]  # last N within TTL
        # message feed (flag events etc.)
        messages = [m for m in self.messagefeed if (now_t - float(m.get("t", 0.0))) <= ttl]
        messages = messages[-kmax:]

        return {
            "type": "state",
            "time": now(),
            "players": players,
            "flags": flags,
            "grenades": grenades,
            "explosions": explosions,
            "teams": {TEAM_RED: {"captures": self.gs.teams[TEAM_RED].captures},
                      TEAM_BLUE: {"captures": self.gs.teams[TEAM_BLUE].captures}},
            "match_over": self.gs.match_over,
            "winner": self.gs.winner,
            "beams": self.recent_beams,
            "killfeed": feed,
            "messages": messages,
        }

    # ---------- Networking ----------
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        hello = await read_json(reader)
        if hello.get("type") != "hello":
            writer.close()
            await writer.wait_closed()
            return

        name = hello.get("name", "Player")
        pid = self.add_player(name, is_bot=False)
        self.clients[pid] = writer

        await send_json(writer, {"type": "welcome", "pid": pid, "team": self.gs.players[pid].team})

        try:
            while True:
                msg = await read_json(reader)
                if not msg:
                    break
                if msg.get("type") == "input":
                    data = msg.get("data", {})
                    current = self.inputs.get(pid, {})
                    current.update(data)
                    self.inputs[pid] = current
                    p = self.gs.players.get(pid)
                    t = msg.get("time")
                    if p is not None and t is not None:
                        p.ping_ms = max(0.0, (now() - float(t)) * 1000.0)
        except Exception as e:
            print(f"[client] {addr} error: {e}")
        finally:
            self.remove_player(pid)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _broadcast_loop(self):
        snap_dt = 1.0 / float(self.cfg["server"]["snapshot_hz"])
        while True:
            s = self.build_snapshot()
            dead = []
            for pid, w in self.clients.items():
                try:
                    await send_json(w, s)
                except Exception:
                    dead.append(pid)
            for pid in dead:
                self.remove_player(pid)
            await asyncio.sleep(snap_dt)

    # ---------- Main game loop ----------
    def _ensure_bot_fill(self):
        if not bool(self.cfg["server"].get("bot_fill", True)):
            return
        per_team = int(self.cfg["server"].get("bot_per_team_target", 5))
        rc = sum(1 for p in self.gs.players.values() if p.team == TEAM_RED and p.pid not in self.clients)
        bc = sum(1 for p in self.gs.players.values() if p.team == TEAM_BLUE and p.pid not in self.clients)
        while rc < per_team:
            self.add_player(name=f"BotR{rc+1}", is_bot=True)
            rc += 1
        while bc < per_team:
            self.add_player(name=f"BotB{bc+1}", is_bot=True)
            bc += 1

    def _update_bots(self):
        THINK_HZ = 10.0
        THINK_DT = 1.0 / THINK_HZ
        tnow = now()

        for idx, (pid, brain) in enumerate(list(self.bot_brains.items())):
            # Initialize per-brain cadence with deterministic jitter by pid
            nxt = getattr(brain, "_next_think_t", None)
            if nxt is None:
                jitter = ((pid % 10) / 10.0) * THINK_DT  # spread inside a slice
                brain._next_think_t = tnow + jitter
                # seed a benign last input so bots stay idle until first think
                brain._last_inputs = {
                    "mx": 0.0, "mz": 0.0, "jump": False, "crouch": False, "walk": False,
                    "fire": False, "interact": False,
                    "yaw": math.degrees(getattr(self.gs.players[pid], "yaw_rad", 0.0)),
                    "pitch": math.degrees(getattr(self.gs.players[pid], "pitch_rad", 0.0)),
                }
                self.inputs[pid] = brain._last_inputs
                continue

            if tnow >= brain._next_think_t:
                try:
                    inputs = brain.decide(self.gs.players[pid], self.gs, self.mapdata)
                    brain._last_inputs = inputs
                    self.inputs[pid] = inputs
                except Exception as e:
                    print(f"[bot_ai] pid={pid} decide() error: {e}")
                    if hasattr(brain, "_last_inputs"):
                        self.inputs[pid] = brain._last_inputs

                # schedule next think
                brain._next_think_t = tnow + THINK_DT
            else:
                # Between thinks, reuse the last inputs (holds steering & fire decisions steady)
                if hasattr(brain, "_last_inputs"):
                    self.inputs[pid] = brain._last_inputs


    async def run(self):
        self._ensure_bot_fill()

        tick_dt = 1.0 / float(self.cfg["server"]["tick_hz"])
        # Physics stepping: split into a few substeps for robust contacts
        substeps = 4
        fixed_dt = tick_dt / substeps

        while True:
            t0 = now()

            # Bots
            self._update_bots()

            # Sync raw inputs into ECS components
            self._sync_inputs_to_ecs()

            # Run ECS systems prior to physics integration
            for system in self._pre_physics_systems:
                system.update(tick_dt)

            # Step Bullet world with substeps
            self.world.doPhysics(tick_dt, substeps, fixed_dt)

            # Post-physics systems
            for system in self._post_physics_systems:
                system.late_update(tick_dt)

            # Ensure legacy GameState reflects physics before event processing
            self._sync_ecs_to_gamestate()

            # Resolve movement/combat events (beams, deaths, grenades)
            self._process_system_events()
            self._sync_ecs_to_gamestate()

            # Flags housekeeping
            self._carry_flags_update()
            self._dropped_flags_auto_return()
            self._sync_flags_to_components()

            # Grenades and unsticking use the updated game state
            self._update_grenades()
            self._auto_unstick(tick_dt)
            self._record_history()

            # Interactions
            for pid, p in self.gs.players.items():
                if not p.alive:
                    continue
                # Auto-pickup/return flags when close; no keypress required
                self._pickup_try(p)

            # Respawns
            for pid, p in list(self.gs.players.items()):
                if (not p.alive) and now() >= p.respawn_at:
                    self.respawn_player(pid)

            # Safety kill-plane: if anyone somehow gets below the world, respawn them
            kill_z = -10.0
            for pid, p in list(self.gs.players.items()):
                if p.alive and p.z < kill_z:
                    self.respawn_player(pid)

            # Win condition
            self._check_captures()

            # Tick pacing
            await asyncio.sleep(max(0, tick_dt - (now() - t0)))

# ---------- Entrypoint ----------
async def main_async(args):
    cfg = load_config(args.config)
    server = LaserTagServer(cfg)

    srv = await asyncio.start_server(server.handle_client, "0.0.0.0", cfg["server"]["port"])
    print(f"[tcp] listening on :{cfg['server']['port']}")

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass  # e.g., Windows

    disc_task  = asyncio.create_task(lan_discovery_server(
        cfg["server"]["name"], cfg["server"]["port"], cfg["server"]["lan_discovery_port"]), name="discovery")
    bcast_task = asyncio.create_task(server._broadcast_loop(), name="broadcast")
    run_task   = asyncio.create_task(server.run(), name="game_loop")

    def _report_done(t: asyncio.Task):
        exc = t.exception()
        if exc:
            print(f"[task:{t.get_name()}] crashed: {exc!r}")
            stop.set()
    for t in (disc_task, bcast_task, run_task):
        t.add_done_callback(_report_done)

    async with srv:
        tcp_task = asyncio.create_task(srv.serve_forever(), name="tcp_server")
        try:
            await stop.wait()          # run until a signal or a task fails
        finally:
            tcp_task.cancel()
            for t in (disc_task, bcast_task, run_task):
                t.cancel()
            await asyncio.gather(tcp_task, disc_task, bcast_task, run_task, return_exceptions=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/defaults.json")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
