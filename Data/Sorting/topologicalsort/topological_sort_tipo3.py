import logging
from typing import List, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class Graph:
    nodes_list: List[Any]
    edges_list: List[tuple] = field(default_factory=list)

def topological_order(graph_obj: Graph) -> List[Any]:
    try:
        in_degrees = {u: 0 for u in graph_obj.nodes_list}
        adj_list = defaultdict(list)
        
        for u, v in graph_obj.edges_list:
            adj_list[u].append(v)
            in_degrees[v] += 1

        queue = deque([u for u in graph_obj.nodes_list if in_degrees[u] == 0])
        ordered_list = []

        while queue:
            u = queue.popleft()
            ordered_list.append(u)
            for v in adj_list[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    queue.append(v)

        if len(ordered_list) != len(graph_obj.nodes_list):
            logging.error("Graph has at least one cycle. Topological sort not possible.")
            raise ValueError("Graph has at least one cycle. Topological sort not possible.")

        logging.info("Topological Sort completed.")
        return ordered_list

    except Exception as e:
        logging.error(f"An error occurred during Topological Sort: {e}")
        raise 