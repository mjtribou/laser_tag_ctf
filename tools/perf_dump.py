"""Utilities for printing a one-shot performance snapshot."""
from __future__ import annotations

import os
import platform
from typing import Mapping, Optional

from panda3d.core import PandaSystem, SceneGraphAnalyzer, StringStream


_ENV_KEY = "PERF_DUMP"
_PREFIX = "[perf]"


def _should_dump(env: Optional[Mapping[str, str]] = None) -> bool:
    env_map = env if env is not None else os.environ
    raw = str(env_map.get(_ENV_KEY, "")).strip().lower()
    if not raw:
        return False
    return raw not in {"0", "false", "no"}


def _gpu_string(base) -> Optional[str]:
    try:
        win = getattr(base, "win", None)
        if not win:
            return None
        gsg = win.getGsg()
        if not gsg:
            return None
        vendor = (gsg.getDriverVendor() or "").strip()
        renderer = (gsg.getDriverRenderer() or "").strip()
        version = (gsg.getDriverVersion() or "").strip()
        parts = []
        if vendor and vendor not in renderer:
            parts.append(vendor)
        if renderer:
            parts.append(renderer)
        if version:
            parts.append(version)
        text = " ".join(p for p in parts if p)
        return text or None
    except Exception:
        return None


def _scene_counts(analyzer: SceneGraphAnalyzer) -> tuple[int, int, int]:
    return (
        analyzer.getNumNodes(),
        analyzer.getNumGeoms(),
        analyzer.getNumVertices(),
    )


def _analyze_dump(analyzer: SceneGraphAnalyzer) -> str:
    stream = StringStream()
    analyzer.write(stream)
    data = stream.getData()
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    return data.strip()


def maybe_dump(base, env: Optional[Mapping[str, str]] = None) -> None:
    """If PERF_DUMP is truthy, emit startup diagnostics and analyze the scene."""
    if not _should_dump(env):
        return

    panda_ver = PandaSystem.get_version_string()
    os_str = platform.platform(aliased=True)
    analyzer = SceneGraphAnalyzer()
    analyzer.addNode(base.render.node())
    counts = _scene_counts(analyzer)
    gpu = _gpu_string(base)

    parts = [f"Panda={panda_ver}", f"OS={os_str}"]
    if gpu:
        parts.append(f"GPU={gpu}")
    parts.append(f"NodePaths={counts[0]}")
    parts.append(f"Geoms={counts[1]}")
    parts.append(f"Vertices={counts[2]}")

    print(f"{_PREFIX} {' | '.join(parts)}")

    dump = _analyze_dump(analyzer)
    for line in dump.splitlines():
        print(f"{_PREFIX} {line}")
