"""Lightweight loader for engine configuration toggles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "engine.json"
_CONFIG_DATA: Dict[str, Any] = {}


def _load() -> Dict[str, Any]:
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception:
        # TODO: consider surfacing this via logging if config errors need visibility.
        return {}


def _ensure_loaded() -> None:
    global _CONFIG_DATA
    if not _CONFIG_DATA:
        _CONFIG_DATA = _load()


def get(path: str, default: Any = None) -> Any:
    """Return a config value using dotted paths, or default when missing."""
    _ensure_loaded()
    if not path:
        return _CONFIG_DATA

    current: Any = _CONFIG_DATA
    for segment in path.split('.'):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
        else:
            return default
    return current
