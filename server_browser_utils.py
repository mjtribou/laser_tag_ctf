"""Utility helpers for the client server browser."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


KNOWN_SERVERS_FILE = os.path.join("configs", "known_servers.json")


def load_known_servers(path: str = KNOWN_SERVERS_FILE) -> List[Dict[str, object]]:
    """Load saved server entries, filtering out malformed data."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return []

    servers = raw.get("servers") if isinstance(raw, dict) else raw
    if not isinstance(servers, list):
        return []

    cleaned: List[Dict[str, object]] = []
    for item in servers:
        if not isinstance(item, dict):
            continue

        host = str(item.get("host", "")).strip()
        if not host:
            continue

        entry: Dict[str, object] = {"host": host}
        entry["port"] = _safe_int(item.get("port", 0))
        entry["discovery_port"] = _safe_int(item.get("discovery_port", 0))

        label = item.get("label")
        if isinstance(label, str) and label.strip():
            entry["label"] = label.strip()

        if "added_at" in item:
            entry["added_at"] = item.get("added_at")

        cleaned.append(entry)

    return cleaned


def save_known_servers(path: str, servers: List[Dict[str, object]]) -> None:
    """Persist the known servers list to disk."""

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"servers": servers}, handle, indent=2)
    except Exception as exc:
        print(f"[settings] failed to save known servers: {exc}")


def format_uptime(seconds: Optional[float]) -> str:
    """Render a friendly uptime string from seconds."""

    if seconds is None:
        return "?"

    try:
        total = int(seconds)
    except Exception:
        return "?"

    if total < 60:
        return f"{total}s"

    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0
