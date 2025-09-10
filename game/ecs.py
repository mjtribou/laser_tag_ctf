from dataclasses import dataclass, field
from typing import Dict, Type, Any

class Component:
    pass

@dataclass
class Entity:
    id: int
    components: Dict[Type[Component], Component] = field(default_factory=dict)

    def add(self, comp: Component) -> None:
        self.components[type(comp)] = comp

    def get(self, comp_type: Type[Component]) -> Any:
        return self.components.get(comp_type)

class System:
    def update(self, dt: float, entities: Dict[int, Entity]) -> None:
        raise NotImplementedError
