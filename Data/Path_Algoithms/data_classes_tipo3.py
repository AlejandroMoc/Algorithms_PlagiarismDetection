from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Graph:
    nodes: List[str]
    connections: Dict[str, Dict[str, int]]