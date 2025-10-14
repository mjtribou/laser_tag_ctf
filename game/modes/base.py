"""Game mode base classes."""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from server import LaserTagServer
    from game.ecs.views import PlayerView, FlagView


class GameMode:
    """Abstract game mode facade used by the authoritative server."""

    def __init__(self, server: "LaserTagServer", cfg: Dict[str, Any]) -> None:
        self.server = server
        self.cfg = cfg

    # --- Lifecycle -----------------------------------------------------
    def setup(self) -> None:
        """Called once the server has finished constructing core state."""

    # --- Per-frame hooks -----------------------------------------------
    def pre_tick(self, now_t: float) -> None:
        """Called at the start of each simulation tick."""

    def after_flag_interactions(self, dt: float) -> None:
        """Called after flag touch checks each tick."""

    # --- Core behaviour ------------------------------------------------
    def handle_flag_touch(self, player: "PlayerView", team_flag: int, flag: "FlagView") -> bool:
        """Return True if the mode consumed the interaction with ``flag``.

        When True the server will skip its default pickup/return handling for
        this player/flag pair.
        """
        return False

    def respawns_locked(self) -> bool:
        """Return True if players should not respawn this frame."""
        return False

    def check_objectives(self) -> None:
        """Evaluate win conditions and objective progress for the mode."""

    def snapshot(self, now_t: float) -> Optional[Dict[str, Any]]:
        """Return additional snapshot payload for clients."""
        return None

    # --- Extension points used by derived modes -----------------------
    def on_neutral_flag_delivered(self, carrier: "PlayerView", flag: "FlagView") -> bool:
        """Hook when a neutral flag carrier reaches their home base.

        Return True to consume the event and prevent the default neutral flag
        scoring behaviour.
        """
        return False
