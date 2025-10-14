"""Neutral flag capture-the-flag game mode."""
from __future__ import annotations

import math
import time
from typing import Dict, Any, TYPE_CHECKING

from game.constants import TEAM_RED, TEAM_BLUE, TEAM_NEUTRAL, BASE_CAPTURE_RADIUS

from .base import GameMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from game.ecs.views import PlayerView, FlagView


class NeutralFlagGameMode(GameMode):
    """Classic neutral-flag capture mode used by default."""

    def __init__(self, server, cfg: Dict[str, Any]) -> None:
        super().__init__(server, cfg)
        server_cfg = server.cfg.get("server", {})
        self.to_win = int(server_cfg.get("captures_to_win", 3))

    # ------------------------------------------------------------------
    def check_objectives(self) -> None:
        if self.server.match_over:
            return
        self._check_match_end()
        if self.server.match_over:
            return
        self._check_flag_captures()

    # ------------------------------------------------------------------
    def _check_match_end(self) -> None:
        for team in (TEAM_RED, TEAM_BLUE):
            if self.server.gs.teams[team].captures >= self.to_win:
                self.server.match_over = True
                self.server.winner = team
                return

    def _check_flag_captures(self) -> None:
        flags = self.server.gs.flags
        players = list(self.server.gs.players.items())
        for pid, player in players:
            if not player.alive:
                continue
            if TEAM_NEUTRAL in flags and len(flags) == 1:
                flag = next(iter(flags.values()))
                if flag.carried_by != pid:
                    continue
                if self.on_neutral_flag_delivered(player, flag):
                    continue
                base = self.server.mapdata.red_base if player.team == TEAM_RED else self.server.mapdata.blue_base
                if math.hypot(player.x - base[0], player.y - base[1]) <= BASE_CAPTURE_RADIUS:
                    self._award_neutral_capture(player, flag)
                continue

            enemy_team = TEAM_BLUE if player.team == TEAM_RED else TEAM_RED
            flag = flags.get(enemy_team)
            if not flag or flag.carried_by != pid:
                continue
            base = self.server.mapdata.red_base if player.team == TEAM_RED else self.server.mapdata.blue_base
            if math.hypot(player.x - base[0], player.y - base[1]) <= BASE_CAPTURE_RADIUS:
                self._award_team_capture(player, flag)

    def _award_neutral_capture(self, player: "PlayerView", flag: "FlagView") -> None:
        flag.carried_by = None
        flag.at_base = True
        fx, fy, fz = self.server._flag_home_pos(flag)
        flag.x, flag.y, flag.z = fx, fy, fz
        flag.dropped_at_time = 0.0
        player.captures += 1
        player.carrying_flag = None
        self.server.gs.teams[player.team].captures += 1
        score = self.server.gs.teams[player.team].captures
        print(f"[score] Team {'RED' if player.team==TEAM_RED else 'BLUE'} captured! -> {score}")
        self._log_message("capture", actor=player)
        self._check_match_end()

    def _award_team_capture(self, player: "PlayerView", flag: "FlagView") -> None:
        flag.carried_by = None
        flag.at_base = True
        fx, fy, fz = self.server._flag_home_pos(flag)
        flag.x, flag.y, flag.z = fx, fy, fz
        flag.dropped_at_time = 0.0
        player.captures += 1
        player.carrying_flag = None
        self.server.gs.teams[player.team].captures += 1
        score = self.server.gs.teams[player.team].captures
        print(f"[score] Team {'RED' if player.team==TEAM_RED else 'BLUE'} captured! -> {score}")
        self._log_message("capture", actor=player)
        self._check_match_end()

    def _log_message(self, event: str, *, actor: "PlayerView") -> None:
        try:
            self.server.messagefeed.append(
                {
                    "t": time.time(),
                    "event": event,
                    "actor": actor.pid,
                    "actor_name": actor.name,
                }
            )
        except Exception:
            pass
