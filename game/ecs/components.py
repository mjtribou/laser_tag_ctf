"""Core ECS components used by the authoritative server."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass
class Position:
    """World-space transform data."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians
    pitch: float = 0.0  # radians


@dataclass
class Physics:
    """Basic kinematic state derived from the physics simulation."""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    on_ground: bool = True


@dataclass
class MovementState:
    """High-level movement toggles exposed via the net protocol."""
    walking: bool = False
    crouching: bool = False


@dataclass
class Health:
    """Simple health/life state."""
    hp: int = 0
    max_hp: int = 0
    alive: bool = True
    respawn_at: float = 0.0


@dataclass
class Weapon:
    """Weapon state for rapid-fire laser tag blasters."""
    shots_remaining: int = 0
    shots_per_mag: int = 0
    reload_end: float = 0.0
    last_fire_time: float = 0.0
    cooldown: float = 0.0
    spread_deg: float = 0.0
    recoil_accum: float = 0.0


@dataclass
class PlayerInput:
    """Immediate input intent gathered from clients/bots."""
    mx: float = 0.0
    mz: float = 0.0
    yaw: float = 0.0  # degrees, mirrors client protocol
    pitch: float = 0.0  # degrees
    fire: bool = False
    fire_timestamp: float = 0.0
    jump: bool = False
    crouch: bool = False
    walk: bool = False
    grenade_charge: float = 0.0


@dataclass
class PlayerInfo:
    """Out-of-band metadata for scoreboards and team logic."""
    pid: int
    name: str
    team: int
    is_bot: bool = False


@dataclass
class CharacterBody:
    """Wrapper around Bullet character controller nodes."""
    controller: Any
    nodepath: Any


@dataclass
class CollisionBody:
    """Static or dynamic collision body tracked by the ECS."""
    node: Any
    nodepath: Any
    tag: str = ""


@dataclass
class Projectile:
    """Simple projectile state (e.g. grenades) tracked as entities."""
    owner_pid: Optional[int] = None
    speed: float = 0.0
    damage: float = 0.0
    ttl: float = 0.0
    spawn_time: float = 0.0


@dataclass
class FlagState:
    """Component describing a capture-the-flag objective."""
    team: int
    at_base: bool = True
    carried_by: Optional[int] = None
    home_position: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    dropped_at_time: float = 0.0


@dataclass
class PlayerStats:
    """Scoreboard and telemetry for a player."""
    tags: int = 0
    outs: int = 0
    captures: int = 0
    defences: int = 0
    ping_ms: float = 0.0


@dataclass
class FlagCarrier:
    """Tracks which flag a player is currently holding."""
    flag_team: Optional[int] = None
