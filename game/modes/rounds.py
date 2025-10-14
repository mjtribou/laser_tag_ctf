"""Round-based neutral flag mode."""
from __future__ import annotations

import math
import time
from typing import Dict, Any, Optional, Set, TYPE_CHECKING

from game.constants import TEAM_RED, TEAM_BLUE, TEAM_NEUTRAL, FLAG_PICKUP_RADIUS

from .neutral_flag import NeutralFlagGameMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from game.ecs.views import PlayerView, FlagView


class RoundNeutralFlagGameMode(NeutralFlagGameMode):
    """Counter-Strike inspired neutral flag rounds."""

    def __init__(self, server, cfg: Dict[str, Any]) -> None:
        super().__init__(server, cfg)
        rounds_cfg = dict(server.cfg.get("server", {}).get("rounds", {}))
        captures_goal = int(server.cfg.get("server", {}).get("captures_to_win", 3))
        self.to_win = int(rounds_cfg.get("to_win", captures_goal))
        self.hold_seconds = float(rounds_cfg.get("hold_seconds", 35.0))
        self.pickup_seconds = float(rounds_cfg.get("pickup_seconds", 5.0))
        self.setup_seconds = float(rounds_cfg.get("setup_seconds", 5.0))
        self.round_seconds = float(rounds_cfg.get("round_seconds", 120.0))

        self.round_index = 0
        self.round_phase: str = "intermission"
        self.round_defender: Optional[int] = None
        self.round_attacker: Optional[int] = None
        self.round_hold_until: float = 0.0
        self.round_secure_started: float = 0.0
        self.round_pickup_progress: float = 0.0
        self.round_deadline_at: float = 0.0
        self._pickup_contenders: Set[int] = set()
        self._next_start_at: float = 0.0

    # ------------------------------------------------------------------
    def setup(self) -> None:
        # Round wins determine the match target.
        self.server.cfg.setdefault("server", {})["captures_to_win"] = self.to_win
        self._schedule_intermission(self.setup_seconds)

    # ------------------------------------------------------------------
    def pre_tick(self, now_t: float) -> None:
        if self.server.match_over:
            return
        if self.round_phase == "intermission" and now_t >= self._next_start_at:
            self._begin_round()
        self._pickup_contenders.clear()

    def after_flag_interactions(self, dt: float) -> None:
        if self.server.match_over or self.round_phase != "secured":
            return
        if not self._pickup_contenders:
            self.round_pickup_progress = 0.0
            return
        attackers = [
            pid
            for pid in self._pickup_contenders
            if pid in self.server.gs.players and self.server.gs.players[pid].alive
        ]
        if not attackers:
            self.round_pickup_progress = 0.0
            return
        base_time = max(0.1, self.pickup_seconds)
        self.round_pickup_progress += dt * len(attackers) / base_time
        if self.round_pickup_progress >= 1.0:
            self._award_round(self.round_attacker, "recovered")
        else:
            self.round_pickup_progress = min(1.0, self.round_pickup_progress)

    def handle_flag_touch(self, player: "PlayerView", team_flag: int, flag: "FlagView") -> bool:
        if (
            team_flag == TEAM_NEUTRAL
            and self.round_phase == "secured"
            and self.round_attacker == player.team
            and player.alive
        ):
            distance = math.hypot(player.x - flag.x, player.y - flag.y)
            if distance <= FLAG_PICKUP_RADIUS:
                self._pickup_contenders.add(player.pid)
            return True
        return False

    def respawns_locked(self) -> bool:
        return self.round_phase in {"neutral", "secured", "intermission"}

    def check_objectives(self) -> None:
        if self.server.match_over:
            return
        self._check_flag_captures()
        if (
            self.round_phase == "secured"
            and self.round_hold_until > 0.0
            and time.time() >= self.round_hold_until
        ):
            self._award_round(self.round_defender, "hold")
        self._check_elimination()
        self._check_round_timeout()
        self._check_match_end()

    def on_neutral_flag_delivered(self, carrier: "PlayerView", flag: "FlagView") -> bool:
        if self.round_phase != "neutral":
            return False
        self._flag_secured(carrier.team, flag, carrier)
        return True

    def snapshot(self, now_t: float) -> Optional[Dict[str, Any]]:
        hold_remaining = 0.0
        pickup = 0.0
        round_remaining = 0.0
        intermission_remaining = 0.0
        if self.round_phase in {"neutral", "secured"} and self.round_deadline_at > 0.0:
            round_remaining = max(0.0, self.round_deadline_at - now_t)
        if self.round_phase == "intermission" and self._next_start_at > 0.0:
            intermission_remaining = max(0.0, self._next_start_at - now_t)
        if self.round_phase == "secured":
            if self.round_hold_until > 0.0:
                hold_remaining = max(0.0, self.round_hold_until - now_t)
            pickup = max(0.0, min(1.0, self.round_pickup_progress))
        wins = {
            TEAM_RED: int(self.server.team_captures.get(TEAM_RED, 0)),
            TEAM_BLUE: int(self.server.team_captures.get(TEAM_BLUE, 0)),
        }
        return {
            "enabled": True,
            "current": int(self.round_index),
            "phase": self.round_phase,
            "defender": self.round_defender,
            "attacker": self.round_attacker,
            "hold_remaining": hold_remaining,
            "pickup_progress": pickup,
            "round_time_remaining": round_remaining,
            "intermission_remaining": intermission_remaining,
            "deadline": self.round_deadline_at if self.round_deadline_at > 0.0 else None,
            "round_seconds": float(self.round_seconds),
            "to_win": int(self.to_win),
            "wins": wins,
        }

    def hud_snapshot(self, now_t: float) -> Optional[Dict[str, Any]]:
        hud_time = 0.0
        if self.server.match_over and self.round_phase == "complete":
            hud_time = 0.0
        elif self.round_phase in {"neutral", "secured"} and self.round_deadline_at > 0.0:
            hud_time = max(0.0, self.round_deadline_at - now_t)
        intermission = 0.0
        if self.round_phase == "intermission" and self._next_start_at > 0.0:
            intermission = max(0.0, self._next_start_at - now_t)
        hold_remaining = 0.0
        if self.round_phase == "secured" and self.round_hold_until > 0.0:
            hold_remaining = max(0.0, self.round_hold_until - now_t)
        return {
            "rounds": {
                "enabled": True,
                "round": int(self.round_index),
                "phase": self.round_phase,
                "time_remaining": hud_time,
                "intermission_remaining": intermission,
                "hold_remaining": hold_remaining,
                "defender": self.round_defender,
                "attacker": self.round_attacker,
                "to_win": int(self.to_win),
                "wins": {
                    TEAM_RED: int(self.server.team_captures.get(TEAM_RED, 0)),
                    TEAM_BLUE: int(self.server.team_captures.get(TEAM_BLUE, 0)),
                },
            }
        }

    def _schedule_intermission(self, delay: float) -> None:
        self.round_phase = "intermission"
        self.round_defender = None
        self.round_attacker = None
        self.round_hold_until = 0.0
        self.round_secure_started = 0.0
        self.round_pickup_progress = 0.0
        self.round_deadline_at = 0.0
        self._pickup_contenders.clear()
        self._next_start_at = time.time() + max(0.0, delay)
        self._reset_flag_position()

    def _begin_round(self) -> None:
        if self.server.match_over:
            return
        self.round_index += 1
        self.round_phase = "neutral"
        self.round_defender = None
        self.round_attacker = None
        self.round_hold_until = 0.0
        now_t = time.time()
        self.round_secure_started = now_t
        self.round_pickup_progress = 0.0
        self.round_deadline_at = (now_t + max(0.0, self.round_seconds)) if self.round_seconds > 0.0 else 0.0
        self._pickup_contenders.clear()
        self._reset_flag_position()
        for pid in list(self.server.gs.players.keys()):
            self.server.respawn_player(pid)
        self._next_start_at = 0.0
        print(f"[round] starting round {self.round_index}")
        self._log_event("round_start", {"round": self.round_index})

    def _reset_flag_position(self) -> None:
        flag = self.server.gs.flags.get(TEAM_NEUTRAL)
        if flag is None:
            return
        fx, fy, fz = self.server._flag_home_pos(flag)
        flag.carried_by = None
        flag.at_base = True
        flag.x, flag.y, flag.z = fx, fy, fz
        flag.dropped_at_time = 0.0
        for player in self.server.gs.players.values():
            if getattr(player, "carrying_flag", None) == TEAM_NEUTRAL:
                player.carrying_flag = None

    def _flag_secured(self, team: int, flag: "FlagView", carrier: "PlayerView") -> None:
        self.round_defender = team
        self.round_attacker = TEAM_BLUE if team == TEAM_RED else TEAM_RED
        self.round_phase = "secured"
        now_t = time.time()
        self.round_secure_started = now_t
        hold_until = now_t + max(0.0, self.hold_seconds)
        self.round_hold_until = hold_until
        self.round_deadline_at = hold_until
        self.round_pickup_progress = 0.0
        self._pickup_contenders.clear()
        pedestal = (
            self.server.mapdata.red_flag_stand
            if team == TEAM_RED
            else self.server.mapdata.blue_flag_stand
        )
        flag.carried_by = None
        flag.at_base = True
        flag.x, flag.y, flag.z = pedestal
        flag.dropped_at_time = 0.0
        carrier.carrying_flag = None
        print(f"[round] Team {'RED' if team==TEAM_RED else 'BLUE'} secured the flag")
        self._log_event("flag_secured", {"team": team})

    def _check_elimination(self) -> None:
        if self.round_phase not in {"neutral", "secured"}:
            return
        red_alive = any(p.alive for p in self.server.gs.players.values() if p.team == TEAM_RED)
        blue_alive = any(p.alive for p in self.server.gs.players.values() if p.team == TEAM_BLUE)
        if red_alive and blue_alive:
            return
        if red_alive and not blue_alive:
            self._award_round(TEAM_RED, "elimination")
        elif blue_alive and not red_alive:
            self._award_round(TEAM_BLUE, "elimination")

    def _check_round_timeout(self) -> None:
        if self.round_phase not in {"neutral", "secured"}:
            return
        if self.round_deadline_at <= 0.0:
            return
        if time.time() < self.round_deadline_at:
            return
        self._handle_round_draw()

    def _handle_round_draw(self) -> None:
        if self.server.match_over or self.round_phase not in {"neutral", "secured"}:
            return
        print(f"[round] Round {self.round_index} expired -> draw")
        self._log_event("round_draw", {"round": self.round_index})
        self.round_hold_until = 0.0
        self.round_pickup_progress = 0.0
        self.round_deadline_at = 0.0
        self._pickup_contenders.clear()
        self._schedule_intermission(self.setup_seconds)

    def _award_round(self, team: Optional[int], reason: str) -> None:
        if team is None or self.server.match_over or self.round_phase == "complete":
            return
        current = int(self.server.team_captures.get(team, 0))
        self.server.team_captures[team] = current + 1
        self.round_hold_until = 0.0
        self.round_pickup_progress = 0.0
        self.round_deadline_at = 0.0
        self._pickup_contenders.clear()
        self._log_event("round_win", {"team": team, "reason": reason})
        print("[round] Team {} wins round {} ({}) -> {}".format(
            "RED" if team == TEAM_RED else "BLUE",
            self.round_index,
            reason,
            self.server.team_captures[team],
        ))
        if self.server.team_captures[team] >= self.to_win:
            self.server.match_over = True
            self.server.winner = team
            self.round_phase = "complete"
            self._next_start_at = float("inf")
            self._reset_flag_position()
        else:
            self._schedule_intermission(self.setup_seconds)

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        data = {"t": time.time(), "event": event, **payload}
        try:
            self.server.messagefeed.append(data)
        except Exception:
            pass
