from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Mapa:
    puntos: List[str]
    conexiones: Dict[str, Dict[str, int]] 