import logging
from typing import List, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class Mapa:
    nodes: List[Any]
    conexiones: List[tuple] = field(default_factory=list)


        queue = deque([u for u in mapa.nodes if in_degree[u] == 0])
        sorted_order = []

        while queue:
            u = queue.popleft()
            sorted_order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_order) != len(mapa.nodes):
            logging.error("Mapa has at least one cycle. Topological sort not possible.")
            raise ValueError("Mapa has at least one cycle. Topological sort not possible.")

        logging.info("Topological Sort completed.")
        return sorted_order

    except Exception as e:
        logging.error(f"An error occurred during Topological Sort: {e}")
        raise 


def topological_sort(mapa: Mapa) -> List[Any]:
    """
    Performs a Topological Sort on a directed acyclic mapa (DAG).
    """
    try:
        in_degree = {u: 0 for u in mapa.nodes}
        adj = defaultdict(list)
        
        for u, v in mapa.conexiones:
            adj[u].append(v)
            in_degree[v] += 1
