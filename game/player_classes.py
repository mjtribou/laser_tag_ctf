from dataclasses import dataclass
from typing import Dict, Optional
import json

@dataclass
class PlayerClass:
    id: str
    run_speed: float
    walk_speed: float
    crouch_speed: float
    jump_speed: float
    accuracy: float = 1.0
    special: Optional[str] = None


def load_player_classes(path: str) -> Dict[str, PlayerClass]:
    with open(path, 'r') as f:
        data = json.load(f)
    classes: Dict[str, PlayerClass] = {}
    for cid, cfg in data.items():
        classes[cid] = PlayerClass(id=cid,
                                   run_speed=cfg.get('run_speed', 0.0),
                                   walk_speed=cfg.get('walk_speed', 0.0),
                                   crouch_speed=cfg.get('crouch_speed', 0.0),
                                   jump_speed=cfg.get('jump_speed', 0.0),
                                   accuracy=cfg.get('accuracy', 1.0),
                                   special=cfg.get('special'))
    return classes
