from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Crz:
    vrs: List[str]
    dgs: Dict[str, Dict[str, int]] 