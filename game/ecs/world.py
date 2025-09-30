"""Lightweight ECS world implementation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class World:
    """Minimal entity-component storage with typed queries."""

    def __init__(self) -> None:
        self._next_entity_id: int = 1
        self._components: Dict[Type[Any], Dict[int, Any]] = defaultdict(dict)
        self._entity_components: Dict[int, set[Type[Any]]] = defaultdict(set)

    def create_entity(self) -> int:
        eid = self._next_entity_id
        self._next_entity_id += 1
        self._entity_components[eid]  # seed entry
        return eid

    def remove_entity(self, entity: int) -> None:
        types = self._entity_components.pop(entity, set())
        for comp_type in types:
            self._components[comp_type].pop(entity, None)

    def add_component(self, entity: int, component: Any) -> None:
        comp_type = type(component)
        self._components[comp_type][entity] = component
        self._entity_components[entity].add(comp_type)

    def remove_component(self, entity: int, comp_type: Type[T]) -> None:
        if entity in self._components.get(comp_type, {}):
            del self._components[comp_type][entity]
        if entity in self._entity_components:
            self._entity_components[entity].discard(comp_type)

    def get_component(self, entity: int, comp_type: Type[T]) -> Optional[T]:
        return self._components.get(comp_type, {}).get(entity)

    def has_component(self, entity: int, comp_type: Type[Any]) -> bool:
        return comp_type in self._entity_components.get(entity, set())

    def entities_with(self, *comp_types: Type[Any]) -> Iterable[int]:
        required = set(comp_types)
        for entity, types in self._entity_components.items():
            if required.issubset(types):
                yield entity

    def query(self, *comp_types: Type[T]) -> Generator[Tuple[int, Tuple[Any, ...]], None, None]:
        for entity in self.entities_with(*comp_types):
            yield entity, tuple(self._components[ct][entity] for ct in comp_types)

    def components_of_type(self, comp_type: Type[T]) -> Dict[int, T]:
        return self._components.get(comp_type, {})


class System:
    """Base class for systems run by the server loop."""

    def __init__(self, world: World) -> None:
        self.world = world

    # Systems may override either phase depending on their needs.
    def update(self, dt: float) -> None:  # pre-physics
        pass

    def late_update(self, dt: float) -> None:  # post-physics
        pass

