from __future__ import annotations

"""Simple entity-component-system utilities for the laser tag server.

This module defines a very small ECS implementation used by the server.
It intentionally keeps the API tiny – enough to express movement, combat
and collision logic in systems while the authoritative server loop simply
iterates over them.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Tuple, Type, TypeVar, Generic, Optional, List

Entity = int

# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """World position and orientation for an entity."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass
class Physics:
    """Simple velocity container used by :class:`MovementSystem`."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    on_ground: bool = True


@dataclass
class Health:
    """Basic health/respawn information."""

    hp: int = 100
    max_hp: int = 100
    alive: bool = True
    respawn_at: float = 0.0


@dataclass
class Weapon:
    """Weapon state for players."""

    shots_remaining: int = 0
    last_fire_time: float = 0.0
    reload_end: float = 0.0
    recoil_accum: float = 0.0


# ---------------------------------------------------------------------------
# ECS World
# ---------------------------------------------------------------------------

T = TypeVar("T")


class World:
    """Minimal ECS world.

    Components are stored in dictionaries keyed by entity id.  Entities are
    just integers which makes serialisation cheap.
    """

    def __init__(self) -> None:
        self._next_entity: int = 1
        self._components: Dict[Type, Dict[Entity, object]] = {}

    # -- entity management -------------------------------------------------
    def create_entity(self) -> Entity:
        eid = self._next_entity
        self._next_entity += 1
        return eid

    def remove_entity(self, eid: Entity) -> None:
        for store in self._components.values():
            store.pop(eid, None)

    # -- component management ----------------------------------------------
    def add_component(self, eid: Entity, comp: object) -> None:
        store = self._components.setdefault(type(comp), {})
        store[eid] = comp

    def component_for_entity(self, eid: Entity, comp_type: Type[T]) -> Optional[T]:
        store = self._components.get(comp_type, {})
        return store.get(eid)  # type: ignore[return-value]

    def components_of_type(self, comp_type: Type[T]) -> Iterator[Tuple[Entity, T]]:
        store = self._components.get(comp_type, {})
        for eid, comp in store.items():
            yield eid, comp  # type: ignore[misc]

    def query(self, *comp_types: Type) -> Iterator[Tuple[Entity, Tuple]]:
        """Iterate over entities that have all ``comp_types``.

        ``query(Position, Physics)`` yields ``(eid, (pos, phys))`` for each
        entity that has both ``Position`` and ``Physics`` components.
        """

        if not comp_types:
            return iter(())
        stores = [self._components.get(t, {}) for t in comp_types]
        if not all(stores):
            return iter(())
        # Intersect keys of all stores
        common = set(stores[0].keys())
        for s in stores[1:]:
            common &= set(s.keys())
        for eid in common:
            comps = tuple(s[eid] for s in stores)
            yield eid, comps


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


class System:
    """Base class for systems."""

    def update(self, world: World, dt: float) -> None:
        raise NotImplementedError


class MovementSystem(System):
    """Integrate velocity into position each tick."""

    def update(self, world: World, dt: float) -> None:
        for eid, (pos, phys) in world.query(Position, Physics):
            pos.x += phys.vx * dt
            pos.y += phys.vy * dt
            pos.z += phys.vz * dt


class CombatSystem(System):
    """Very small system that simply handles weapon reload timers."""

    def update(self, world: World, dt: float) -> None:
        for eid, weapon in world.components_of_type(Weapon):
            if weapon.shots_remaining <= 0 and weapon.reload_end > 0.0:
                weapon.reload_end = max(0.0, weapon.reload_end - dt)
                if weapon.reload_end == 0.0:
                    weapon.shots_remaining = 0


class CollisionSystem(System):
    """Clamp entities inside the map bounds.

    The map is treated as an axis-aligned bounding box.
    """

    def __init__(self, width: float = 50.0, height: float = 50.0):
        self.width = width
        self.height = height

    def update(self, world: World, dt: float) -> None:  # noqa: D401 - simple
        half_w = self.width * 0.5
        half_h = self.height * 0.5
        for _, pos in world.components_of_type(Position):
            pos.x = max(-half_w, min(half_w, pos.x))
            pos.y = max(-half_h, min(half_h, pos.y))
            # z is left unclamped; gravity handled elsewhere


__all__ = [
    "World",
    "Position",
    "Physics",
    "Health",
    "Weapon",
    "System",
    "MovementSystem",
    "CombatSystem",
    "CollisionSystem",
]
