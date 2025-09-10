from typing import Callable, Dict, List, Any

class EventBus:
    """Simple pub/sub event bus."""
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[..., None]]] = {}

    def subscribe(self, event: str, cb: Callable[..., None]) -> None:
        self._subs.setdefault(event, []).append(cb)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for cb in self._subs.get(event, []):
            cb(*args, **kwargs)
