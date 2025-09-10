# game/server_state.py
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import time

from .constants import TEAM_RED, TEAM_BLUE, PLAYER_HEIGHT, PLAYER_RADIUS, FLAG_PICKUP_RADIUS, FLAG_RETURN_RADIUS, BASE_CAPTURE_RADIUS

@dataclass
class Player:
    pid: int
    name: str
    team: int
    x: float = 0.0
    y: float = 0.0          # Panda Y (forward)
    z: float = 0.9          # Panda Z (up)
    yaw_rad: float = 0.0  # radians, wrapped [-pi,pi]
    pitch_rad: float = 0.0  # radians
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.9          # Panda Z (up)
    on_ground: bool = True
    crouching: bool = False
    walking: bool = False
    alive: bool = True
    respawn_at: float = 0.0
    carrying_flag: Optional[int] = None  # 0 red flag, 1 blue flag (team id)
    last_fire_time: float = 0.0
    recoil_accum: float = 0.0
    shots_remaining: int = 0
    reload_end: float = 0.0
    # --- Stats for scoreboard ---
    tags: int = 0
    outs: int = 0
    captures: int = 0
    defences: int = 0
    ping_ms: float = 0.0

@dataclass
class Flag:
    team: int
    at_base: bool = True
    carried_by: Optional[int] = None  # pid
    x: float = 0.0
    y: float = 0.0
    z: float = 0.9          # Panda Z (up)
    dropped_at_time: float = 0.0

@dataclass
class TeamState:
    captures: int = 0

@dataclass
class GameState:
    players: Dict[int, Player] = field(default_factory=dict)
    flags: Dict[int, Flag] = field(default_factory=dict)
    teams: Dict[int, TeamState] = field(default_factory=lambda:{TEAM_RED: TeamState(), TEAM_BLUE: TeamState()})
    match_over: bool = False
    winner: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    
