"""Thin proxy views over ECS components for legacy code paths."""
from __future__ import annotations

from typing import Dict, Optional

from .components import (
    FlagCarrier,
    FlagState,
    Health,
    MovementState,
    Physics,
    PlayerInfo,
    PlayerStats,
    Position,
    Weapon,
)


class PlayerView:
    """Legacy-style player object backed by ECS components."""

    def __init__(
        self,
        *,
        pid: int,
        info: PlayerInfo,
        position: Position,
        physics: Physics,
        movement: MovementState,
        weapon: Weapon,
        health: Health,
        stats: PlayerStats,
        flag_carrier: FlagCarrier,
    ) -> None:
        self.pid = pid
        self._info = info
        self._position = position
        self._physics = physics
        self._movement = movement
        self._weapon = weapon
        self._health = health
        self._stats = stats
        self._flag_carrier = flag_carrier

    # --- Identity / metadata ---
    @property
    def name(self) -> str:
        return self._info.name

    @name.setter
    def name(self, value: str) -> None:
        self._info.name = value

    @property
    def team(self) -> int:
        return self._info.team

    @team.setter
    def team(self, value: int) -> None:
        self._info.team = value

    @property
    def is_bot(self) -> bool:
        return self._info.is_bot

    @is_bot.setter
    def is_bot(self, value: bool) -> None:
        self._info.is_bot = value

    # --- Transform ---
    @property
    def x(self) -> float:
        return self._position.x

    @x.setter
    def x(self, value: float) -> None:
        self._position.x = value

    @property
    def y(self) -> float:
        return self._position.y

    @y.setter
    def y(self, value: float) -> None:
        self._position.y = value

    @property
    def z(self) -> float:
        return self._position.z

    @z.setter
    def z(self, value: float) -> None:
        self._position.z = value

    @property
    def yaw_rad(self) -> float:
        return self._position.yaw

    @yaw_rad.setter
    def yaw_rad(self, value: float) -> None:
        self._position.yaw = value

    @property
    def pitch_rad(self) -> float:
        return self._position.pitch

    @pitch_rad.setter
    def pitch_rad(self, value: float) -> None:
        self._position.pitch = value

    # --- Kinematics ---
    @property
    def vx(self) -> float:
        return self._physics.vx

    @vx.setter
    def vx(self, value: float) -> None:
        self._physics.vx = value

    @property
    def vy(self) -> float:
        return self._physics.vy

    @vy.setter
    def vy(self, value: float) -> None:
        self._physics.vy = value

    @property
    def vz(self) -> float:
        return self._physics.vz

    @vz.setter
    def vz(self, value: float) -> None:
        self._physics.vz = value

    @property
    def on_ground(self) -> bool:
        return self._physics.on_ground

    @on_ground.setter
    def on_ground(self, value: bool) -> None:
        self._physics.on_ground = value

    # --- Movement state ---
    @property
    def crouching(self) -> bool:
        return self._movement.crouching

    @crouching.setter
    def crouching(self, value: bool) -> None:
        self._movement.crouching = value

    @property
    def walking(self) -> bool:
        return self._movement.walking

    @walking.setter
    def walking(self, value: bool) -> None:
        self._movement.walking = value

    # --- Health ---
    @property
    def alive(self) -> bool:
        return self._health.alive

    @alive.setter
    def alive(self, value: bool) -> None:
        self._health.alive = value

    @property
    def respawn_at(self) -> float:
        return self._health.respawn_at

    @respawn_at.setter
    def respawn_at(self, value: float) -> None:
        self._health.respawn_at = value

    # --- Weapon ---
    @property
    def shots_remaining(self) -> int:
        return self._weapon.shots_remaining

    @shots_remaining.setter
    def shots_remaining(self, value: int) -> None:
        self._weapon.shots_remaining = value

    @property
    def shots_per_mag(self) -> int:
        return self._weapon.shots_per_mag

    @shots_per_mag.setter
    def shots_per_mag(self, value: int) -> None:
        self._weapon.shots_per_mag = value

    @property
    def reload_end(self) -> float:
        return self._weapon.reload_end

    @reload_end.setter
    def reload_end(self, value: float) -> None:
        self._weapon.reload_end = value

    @property
    def last_fire_time(self) -> float:
        return self._weapon.last_fire_time

    @last_fire_time.setter
    def last_fire_time(self, value: float) -> None:
        self._weapon.last_fire_time = value

    @property
    def recoil_accum(self) -> float:
        return self._weapon.recoil_accum

    @recoil_accum.setter
    def recoil_accum(self, value: float) -> None:
        self._weapon.recoil_accum = value

    # --- Flag ---
    @property
    def carrying_flag(self) -> Optional[int]:
        return self._flag_carrier.flag_team

    @carrying_flag.setter
    def carrying_flag(self, value: Optional[int]) -> None:
        self._flag_carrier.flag_team = value

    # --- Stats ---
    @property
    def tags(self) -> int:
        return self._stats.tags

    @tags.setter
    def tags(self, value: int) -> None:
        self._stats.tags = value

    @property
    def outs(self) -> int:
        return self._stats.outs

    @outs.setter
    def outs(self, value: int) -> None:
        self._stats.outs = value

    @property
    def captures(self) -> int:
        return self._stats.captures

    @captures.setter
    def captures(self, value: int) -> None:
        self._stats.captures = value

    @property
    def defences(self) -> int:
        return self._stats.defences

    @defences.setter
    def defences(self, value: int) -> None:
        self._stats.defences = value

    @property
    def ping_ms(self) -> float:
        return self._stats.ping_ms

    @ping_ms.setter
    def ping_ms(self, value: float) -> None:
        self._stats.ping_ms = value


class FlagView:
    """Legacy-style flag object backed by ECS components."""

    def __init__(self, *, entity: int, state: FlagState, position: Position) -> None:
        self.entity = entity
        self._state = state
        self._position = position

    @property
    def team(self) -> int:
        return self._state.team

    @property
    def at_base(self) -> bool:
        return self._state.at_base

    @at_base.setter
    def at_base(self, value: bool) -> None:
        self._state.at_base = value

    @property
    def carried_by(self) -> Optional[int]:
        return self._state.carried_by

    @carried_by.setter
    def carried_by(self, value: Optional[int]) -> None:
        self._state.carried_by = value

    @property
    def home_position(self):
        return self._state.home_position

    @home_position.setter
    def home_position(self, value) -> None:
        self._state.home_position = value

    @property
    def dropped_at_time(self) -> float:
        return self._state.dropped_at_time

    @dropped_at_time.setter
    def dropped_at_time(self, value: float) -> None:
        self._state.dropped_at_time = value

    @property
    def x(self) -> float:
        return self._position.x

    @x.setter
    def x(self, value: float) -> None:
        self._position.x = value

    @property
    def y(self) -> float:
        return self._position.y

    @y.setter
    def y(self, value: float) -> None:
        self._position.y = value

    @property
    def z(self) -> float:
        return self._position.z

    @z.setter
    def z(self, value: float) -> None:
        self._position.z = value


class TeamView:
    """Mutable view over team captures."""

    def __init__(self, captures: Dict[int, int], team_id: int) -> None:
        self._captures = captures
        self.team_id = team_id

    @property
    def captures(self) -> int:
        return self._captures[self.team_id]

    @captures.setter
    def captures(self, value: int) -> None:
        self._captures[self.team_id] = value


class GameStateView:
    """Aggregate facade exposing legacy GameState-like accessors."""

    def __init__(self, server) -> None:
        self._server = server

    @property
    def players(self) -> Dict[int, PlayerView]:
        return self._server.player_views

    @property
    def flags(self) -> Dict[int, FlagView]:
        return self._server.flag_views

    @property
    def teams(self) -> Dict[int, TeamView]:
        return self._server.team_views

    @property
    def match_over(self) -> bool:
        return self._server.match_over

    @match_over.setter
    def match_over(self, value: bool) -> None:
        self._server.match_over = value

    @property
    def winner(self) -> Optional[int]:
        return self._server.winner

    @winner.setter
    def winner(self, value: Optional[int]) -> None:
        self._server.winner = value

    @property
    def start_time(self) -> float:
        return self._server.start_time
