"""Runtime chunk activation manager for render and physics."""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Set, Tuple

from panda3d.core import NodePath

try:  # Bullet is optional on the client side.
    from panda3d.bullet import BulletWorld, BulletRigidBodyNode
except Exception:  # pragma: no cover - Bullet may be absent in some builds.
    BulletWorld = None  # type: ignore
    BulletRigidBodyNode = None  # type: ignore

from world.chunks import ChunkKey


class WorldChunkManager:
    """Keeps render/physics chunks near the focus point active."""

    def __init__(
        self,
        render_parent: NodePath,
        chunk_size: Tuple[int, int, int],
        cube_size: float,
        origin_indices: Tuple[int, int, int],
        render_distance: int,
        *,
        bullet_world: Optional["BulletWorld"] = None,
        tick_hz: float = 8.0,
    ) -> None:
        self.render_parent = render_parent
        self.chunk_size = tuple(int(max(1, v)) for v in chunk_size)
        self.cube_size = float(cube_size) if cube_size else 1.0
        self.origin_indices = tuple(int(v) for v in origin_indices)
        self.render_distance = max(0, int(render_distance))
        self.bullet_world = bullet_world

        interval = 1.0 / max(0.1, float(tick_hz))
        self._tick_interval = interval
        self._last_update = 0.0

        self._render_chunks: Dict[ChunkKey, NodePath] = {}
        self._physics_chunks: Dict[ChunkKey, "BulletRigidBodyNode"] = {}
        self._active_render: Set[ChunkKey] = set()
        self._active_physics: Set[ChunkKey] = set()

        self._last_logged_counts: Tuple[int, int] = (-1, -1)

    # ------------------------------------------------------------------
    def register_chunk(
        self,
        key: ChunkKey,
        node: NodePath,
        body: Optional["BulletRigidBodyNode"] = None,
    ) -> None:
        """Register a chunk's render node (and optional physics body)."""
        self._render_chunks[key] = node
        # Ensure chunks start detached; they'll be reparented on update.
        if not node.isEmpty():
            node.detachNode()

        if body is not None:
            self._physics_chunks[key] = body
            if self.bullet_world is not None:
                try:
                    self.bullet_world.removeRigidBody(body)
                except Exception:
                    pass
        elif key in self._physics_chunks:
            self._physics_chunks.pop(key, None)

    # ------------------------------------------------------------------
    def update(
        self,
        focus_pos: Tuple[float, float, float],
        now: Optional[float] = None,
        *,
        force: bool = False,
    ) -> None:
        """Activate chunks near the focus position; detach the rest."""
        if now is None:
            now = time.time()
        if not force and (now - self._last_update) < self._tick_interval:
            return
        self._last_update = now

        focus_key = self._world_to_chunk(focus_pos)
        active_keys = self._select_active_keys(focus_key)

        # --- Render chunks ---
        newly_active = active_keys - self._active_render
        for key in newly_active:
            node = self._render_chunks.get(key)
            if node is None or node.isEmpty():
                continue
            node.reparentTo(self.render_parent)
        newly_inactive = self._active_render - active_keys
        for key in newly_inactive:
            node = self._render_chunks.get(key)
            if node is None or node.isEmpty():
                continue
            node.detachNode()
        self._active_render = active_keys

        # --- Physics chunks ---
        if self.bullet_world is not None:
            desired_phys = {key for key in active_keys if key in self._physics_chunks}
            phys_activate = desired_phys - self._active_physics
            for key in phys_activate:
                body = self._physics_chunks.get(key)
                if body is None:
                    continue
                try:
                    self.bullet_world.attachRigidBody(body)
                except Exception:
                    pass
            phys_deactivate = self._active_physics - desired_phys
            for key in phys_deactivate:
                body = self._physics_chunks.get(key)
                if body is None:
                    continue
                try:
                    self.bullet_world.removeRigidBody(body)
                except Exception:
                    pass
            self._active_physics = desired_phys

        self._log_counts()

    # ------------------------------------------------------------------
    def _select_active_keys(self, focus_key: ChunkKey) -> Set[ChunkKey]:
        if not self._render_chunks:
            return set()
        max_d = self.render_distance
        if max_d < 0:
            return set(self._render_chunks.keys())
        active: Set[ChunkKey] = set()
        for key in self._render_chunks.keys():
            dx = abs(key.x - focus_key.x)
            dy = abs(key.y - focus_key.y)
            dz = abs(key.z - focus_key.z)
            if max(dx, dy, dz) <= max_d:
                active.add(key)
        return active

    def _world_to_chunk(self, pos: Tuple[float, float, float]) -> ChunkKey:
        vx = int(math.floor(pos[0] / self.cube_size)) - self.origin_indices[0]
        vy = int(math.floor(pos[1] / self.cube_size)) - self.origin_indices[1]
        vz = int(math.floor(pos[2] / self.cube_size)) - self.origin_indices[2]
        cx = vx // self.chunk_size[0]
        cy = vy // self.chunk_size[1]
        cz = vz // self.chunk_size[2]
        return ChunkKey(cx, cy, cz)

    def _log_counts(self) -> None:
        render_count = len(self._active_render)
        total = len(self._render_chunks)
        if (render_count, total) != self._last_logged_counts:
            print(f"[perf] chunks active={render_count}/{total}")
            self._last_logged_counts = (render_count, total)

    # ------------------------------------------------------------------
    @property
    def render_chunks(self) -> Dict[ChunkKey, NodePath]:
        return self._render_chunks

    @property
    def physics_chunks(self) -> Dict[ChunkKey, "BulletRigidBodyNode"]:
        return self._physics_chunks

    @property
    def active_render_count(self) -> int:
        return len(self._active_render)

    @property
    def total_chunks(self) -> int:
        return len(self._render_chunks)
