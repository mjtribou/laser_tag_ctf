"""Snapshot/replication helpers for ECS-backed game state."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from .world import World


class SnapshotBuilder:
    """Serialises ECS component data into the legacy snapshot payload."""

    def __init__(self, world: World) -> None:
        self.world = world

    def build(
        self,
        *,
        now_t: float,
        beams: List[Dict[str, Any]],
        grenades: List[Dict[str, Any]],
        explosions: List[Dict[str, Any]],
        killfeed: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        team_captures: Dict[int, int],
        match_over: bool,
        winner: Optional[int],
        corpse_angles: Dict[int, Tuple[float, float, float]],
    ) -> Dict[str, Any]:
        players = self._collect_players(now_t, corpse_angles)
        flags = self._collect_flags()
        teams = {team_id: {"captures": captures} for team_id, captures in team_captures.items()}

        return {
            "type": "state",
            "time": now_t,
            "players": players,
            "flags": flags,
            "grenades": grenades,
            "explosions": explosions,
            "teams": teams,
            "match_over": match_over,
            "winner": winner,
            "beams": list(beams),
            "killfeed": list(killfeed),
            "messages": list(messages),
        }

    def _collect_players(
        self,
        now_t: float,
        corpse_angles: Dict[int, Tuple[float, float, float]],
    ) -> List[Dict[str, Any]]:
        players: List[Dict[str, Any]] = []
        for entity, comps in self.world.query(
            PlayerInfo,
            Position,
            Health,
            Weapon,
            PlayerStats,
            MovementState,
            Physics,
            FlagCarrier,
        ):
            info: PlayerInfo
            pos: Position
            health: Health
            weapon: Weapon
            stats: PlayerStats
            movement: MovementState
            physics: Physics
            carrier: FlagCarrier
            info, pos, health, weapon, stats, movement, physics, carrier = comps

            pid = info.pid
            base = {
                "pid": pid,
                "name": info.name,
                "team": info.team,
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
                "alive": health.alive,
                "shots": weapon.shots_remaining,
                "reload": max(0.0, weapon.reload_end - now_t),
                "tags": stats.tags,
                "outs": stats.outs,
                "captures": stats.captures,
                "defences": stats.defences,
                "ping": int(stats.ping_ms),
            }

            if health.alive:
                base["yaw"] = math.degrees(pos.yaw)
                base["pitch"] = math.degrees(pos.pitch)
                base["roll"] = 0.0
            else:
                yaw, pitch, roll = corpse_angles.get(
                    pid,
                    (math.degrees(pos.yaw), math.degrees(pos.pitch), 0.0),
                )
                base["yaw"] = yaw
                base["pitch"] = pitch
                base["roll"] = roll

            base["carrying_flag"] = carrier.flag_team
            players.append(base)

        return players

    def _collect_flags(self) -> List[Dict[str, Any]]:
        flags: List[Dict[str, Any]] = []
        for _, (state, pos) in self.world.query(FlagState, Position):
            flags.append(
                {
                    "team": state.team,
                    "at_base": state.at_base,
                    "carried_by": state.carried_by,
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z,
                }
            )
        return flags
