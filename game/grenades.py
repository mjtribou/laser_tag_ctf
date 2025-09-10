from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class GrenadeType:
    id: str
    fuse: float
    radius: float
    knockback: float


def load_grenades(path: str) -> Dict[str, GrenadeType]:
    with open(path, 'r') as f:
        data = json.load(f)
    grenades: Dict[str, GrenadeType] = {}
    for gid, cfg in data.items():
        grenades[gid] = GrenadeType(id=gid,
                                    fuse=cfg.get('fuse', 3.0),
                                    radius=cfg.get('radius', 0.0),
                                    knockback=cfg.get('knockback', 0.0))
    return grenades
