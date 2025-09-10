from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class Weapon:
    id: str
    fire_rate: float
    recoil: float
    spread: float
    mag_capacity: int
    reload_seconds: float


def load_weapons(path: str) -> Dict[str, Weapon]:
    with open(path, 'r') as f:
        data = json.load(f)
    weapons: Dict[str, Weapon] = {}
    for wid, cfg in data.items():
        weapons[wid] = Weapon(id=wid,
                              fire_rate=cfg.get('fire_rate', 1.0),
                              recoil=cfg.get('recoil', 0.0),
                              spread=cfg.get('spread', 0.0),
                              mag_capacity=cfg.get('mag_capacity', 0),
                              reload_seconds=cfg.get('reload_seconds', 0.0))
    return weapons
