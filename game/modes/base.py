"""Game mode base classes and helpers."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from panda3d.core import Vec3

from game.constants import TEAM_BLUE, TEAM_RED
from game.ecs import (
    CharacterBody,
    FlagCarrier,
    Health,
    MovementState,
    Physics,
    PlayerInfo,
    PlayerInput as ECSPlayerInput,
    PlayerStats,
    Position,
    Weapon,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from server import LaserTagServer
    from game.ecs import CombatSystem, CollisionSystem, MovementSystem
    from game.ecs.views import PlayerView, FlagView


@dataclass
class ModeSystems:
    """Container bundling the ECS systems a mode contributes."""

    pre_physics: List[Any]
    post_physics: List[Any]
    movement: Optional["MovementSystem"] = None
    combat: Optional["CombatSystem"] = None
    collision: Optional["CollisionSystem"] = None


class GameMode:
    """Abstract game mode facade used by the authoritative server."""

    def __init__(self, server: "LaserTagServer", cfg: Dict[str, Any]) -> None:
        self.server = server
        self.cfg = cfg

    # --- Lifecycle --------------------------------------------------
    def setup(self) -> None:
        """Called once the server has finished constructing core state."""

    def build_systems(self) -> ModeSystems:
        """Create the ECS systems this mode requires."""

        from game.ecs import CombatSystem, CollisionSystem, MovementSystem

        gameplay_cfg = self.server.cfg.get("gameplay", {})
        server_cfg = self.server.cfg.get("server", {})

        movement = MovementSystem(self.server.ecs, gameplay_cfg)
        combat = CombatSystem(
            self.server.ecs,
            gameplay_cfg,
            server_cfg,
            self.server.world,
            voxel_query=self.server._voxel_query,
            now_fn=time.time,
        )
        collision = CollisionSystem(self.server.ecs, gameplay_cfg)
        return ModeSystems(
            pre_physics=[movement, combat],
            post_physics=[movement, collision],
            movement=movement,
            combat=combat,
            collision=collision,
        )

    # --- Team & HUD data -------------------------------------------
    def teams(self) -> Iterable[int]:
        """Return the team identifiers the mode supports."""

        return (TEAM_RED, TEAM_BLUE)

    def team_snapshot(self) -> Dict[int, Dict[str, Any]]:
        """Return scoreboard payload for each team."""

        captures = self.server.team_captures
        return {int(team): {"captures": captures.get(team, 0)} for team in self.teams()}

    def hud_snapshot(self, now_t: float) -> Optional[Dict[str, Any]]:
        """Optional mode-specific HUD payload for clients."""

        return None

    # --- Player lifecycle ------------------------------------------
    def select_team(self) -> int:
        """Choose the team for a newly joining player."""

        red_ct = sum(1 for p in self.server.player_views.values() if p.team == TEAM_RED)
        blue_ct = sum(1 for p in self.server.player_views.values() if p.team == TEAM_BLUE)
        return TEAM_RED if red_ct <= blue_ct else TEAM_BLUE

    def spawn_location(self, team: int) -> Tuple[float, float, float, float]:
        """Return (x, y, z, yaw_rad) for the given team."""

        base = self.server.mapdata.red_base if team == TEAM_RED else self.server.mapdata.blue_base
        x, y, z = self.server._find_safe_spawn_near((base[0], base[1]))
        yaw_rad = 0.0 if team == TEAM_RED else math.pi
        return x, y, z, yaw_rad

    def create_player_info(self, pid: int, name: str, team: int, is_bot: bool) -> PlayerInfo:
        return PlayerInfo(pid=pid, name=name, team=team, is_bot=is_bot)

    def create_weapon(self, team: int, is_bot: bool) -> Weapon:
        gameplay = self.server.cfg.get("gameplay", {})
        shots_per_mag = int(gameplay.get("shots_per_mag", 20))
        return Weapon(
            shots_remaining=shots_per_mag,
            shots_per_mag=shots_per_mag,
            reload_end=0.0,
            last_fire_time=0.0,
            cooldown=0.0,
            spread_deg=float(gameplay.get("base_spread_deg", 1.0)),
            recoil_accum=0.0,
        )

    def create_health(self, team: int, is_bot: bool) -> Health:
        return Health(hp=1, max_hp=1, alive=True, respawn_at=0.0)

    def create_player_stats(self) -> PlayerStats:
        return PlayerStats()

    def create_flag_carrier(self) -> FlagCarrier:
        return FlagCarrier(flag_team=None)

    def add_player(self, name: str, is_bot: bool = False) -> int:
        """Create a new player controlled by a client or bot."""

        team = self.select_team()
        pid = self.server.next_pid
        self.server.next_pid += 1

        x, y, z, yaw_rad = self.spawn_location(team)

        entity = self.server.ecs.create_entity()
        self.server.pid_to_entity[pid] = entity
        self.server.entity_to_pid[entity] = pid

        position = Position(x=x, y=y, z=z, yaw=yaw_rad, pitch=0.0)
        physics = Physics(vx=0.0, vy=0.0, vz=0.0, on_ground=True)
        movement = MovementState()
        inputs = ECSPlayerInput(yaw=math.degrees(yaw_rad), pitch=0.0)
        info = self.create_player_info(pid, name, team, is_bot)
        weapon = self.create_weapon(team, is_bot)
        health = self.create_health(team, is_bot)
        stats = self.create_player_stats()
        carrier = self.create_flag_carrier()

        self.server.ecs.add_component(entity, position)
        self.server.ecs.add_component(entity, physics)
        self.server.ecs.add_component(entity, movement)
        self.server.ecs.add_component(entity, inputs)
        self.server.ecs.add_component(entity, info)
        self.server.ecs.add_component(entity, weapon)
        self.server.ecs.add_component(entity, health)
        self.server.ecs.add_component(entity, stats)
        self.server.ecs.add_component(entity, carrier)

        from game.ecs.views import PlayerView

        player_view = PlayerView(
            pid=pid,
            info=info,
            position=position,
            physics=physics,
            movement=movement,
            weapon=weapon,
            health=health,
            stats=stats,
            flag_carrier=carrier,
        )
        self.server.player_views[pid] = player_view

        self.server._create_character(entity, pid, (x, y, z), yaw_rad)
        self.server._last_safe_pos[pid] = (x, y, z)

        if is_bot:
            brain = self.create_bot_brain(player_view)
            if brain is not None:
                self.server.bot_brains[pid] = brain

        print(f"[join] pid={pid} name={name} team={'RED' if team==TEAM_RED else 'BLUE'}")
        return pid

    def create_bot_brain(self, player: "PlayerView"):
        """Return an optional bot brain for ``player``."""

        if not player.is_bot:
            return None

        from game.bot_ai import AStarBotBrain

        team = player.team
        base_pos = self.server.mapdata.red_base if team == TEAM_RED else self.server.mapdata.blue_base
        enemy_base = self.server.mapdata.blue_base if team == TEAM_RED else self.server.mapdata.red_base

        bot_cfg = self.server.cfg.get("server", {}).get("bot", {})
        target_players = bool(bot_cfg.get("target_players", True))
        turn_cfg = bot_cfg.get("turn_rates", {})
        idle_turn = float(turn_cfg.get("idle", 240.0))
        engaged_turn = float(turn_cfg.get("engaged", 420.0))
        engagement_range = float(bot_cfg.get("engagement_range_m", 40.0))
        target_acquire_range = float(bot_cfg.get("target_acquire_range_m", max(engagement_range, 45.0)))
        if target_acquire_range < engagement_range:
            target_acquire_range = engagement_range

        return AStarBotBrain(
            team,
            base_pos,
            enemy_base,
            target_players=target_players,
            nav_graph=self.server.nav_graph,
            idle_turn_rate_deg=idle_turn,
            engaged_turn_rate_deg=engaged_turn,
            engagement_range_m=engagement_range,
            target_acquire_range_m=target_acquire_range,
        )

    def spawn_weapon_state(self, team: int, is_bot: bool) -> Dict[str, float]:
        """Return weapon parameters that should be applied on spawn."""

        shots_per_mag = int(self.server.cfg.get("gameplay", {}).get("shots_per_mag", 20))
        return {
            "shots_per_mag": shots_per_mag,
            "shots_remaining": shots_per_mag,
            "reload_end": 0.0,
            "recoil_accum": 0.0,
        }

    def respawn_player(self, pid: int) -> None:
        """Reset state and move the player to a spawn location."""

        player = self.server.gs.players.get(pid)
        if not player:
            return

        self.server._remove_corpse(pid)

        x, y, z, yaw_rad = self.spawn_location(player.team)
        player.x, player.y, player.z = x, y, z
        player.yaw_rad = yaw_rad
        player.pitch_rad = 0.0
        player.vx = player.vy = player.vz = 0.0
        player.on_ground = True
        player.crouching = False
        player.walking = False
        player.carrying_flag = None
        player.alive = True
        player.respawn_at = 0.0

        weapon_state = self.spawn_weapon_state(player.team, player.is_bot)
        player.shots_per_mag = int(weapon_state["shots_per_mag"])
        player.shots_remaining = int(weapon_state["shots_remaining"])
        player.reload_end = float(weapon_state["reload_end"])
        player.recoil_accum = float(weapon_state["recoil_accum"])

        self.server._last_safe_pos[pid] = (x, y, z)

        entity = self.server.pid_to_entity.get(pid)
        if entity is not None:
            inputs = self.server.ecs.get_component(entity, ECSPlayerInput)
            if inputs:
                inputs.yaw = math.degrees(yaw_rad)
                inputs.pitch = 0.0
                inputs.mx = 0.0
                inputs.mz = 0.0
                inputs.fire = False
                inputs.jump = False
                inputs.walk = False
                inputs.crouch = False

            body = self.server.ecs.get_component(entity, CharacterBody)
            if body is None or getattr(body, "nodepath", None) is None:
                self.server._create_character(entity, pid, (x, y, z), yaw_rad)
            else:
                try:
                    body.nodepath.setPos(x, y, z)
                    body.controller.setLinearMovement(Vec3(0, 0, 0), False)
                except Exception:
                    pass

            health = self.server.ecs.get_component(entity, Health)
            if health:
                max_hp = getattr(health, "max_hp", 1) or 1
                health.hp = max(1, int(max_hp))
                health.alive = True
                health.respawn_at = 0.0

    # --- Per-frame hooks ---------------------------------------------
    def pre_tick(self, now_t: float) -> None:
        """Called at the start of each simulation tick."""

    def after_flag_interactions(self, dt: float) -> None:
        """Called after flag touch checks each tick."""

    # --- Core behaviour ----------------------------------------------
    def handle_flag_touch(self, player: "PlayerView", team_flag: int, flag: "FlagView") -> bool:
        """Return True if the mode consumed the interaction with ``flag``."""

        return False

    def respawns_locked(self) -> bool:
        """Return True if players should not respawn this frame."""

        return False

    def check_objectives(self) -> None:
        """Evaluate win conditions and objective progress for the mode."""

    def snapshot(self, now_t: float) -> Optional[Dict[str, Any]]:
        """Return additional snapshot payload for clients."""

        return None

    # --- Extension points used by derived modes ----------------------
    def on_neutral_flag_delivered(self, carrier: "PlayerView", flag: "FlagView") -> bool:
        """Hook when a neutral flag carrier reaches their home base."""

        return False
