from collections import deque
from typing import Dict, List, Any

def breadth_first_search(adjacency_list: Dict[Any, List[Any]], start_node: Any) -> List[Any]:
    if start_node not in adjacency_list:
        raise ValueError(f"Start node {start_node} not found in the graph.")

    visited_nodes = set()
    queue = deque([start_node])
    visited_nodes.add(start_node)

    traversal_order = []

    while queue:
        current_node = queue.popleft()
        traversal_order.append(current_node)

        for neighbor_node in adjacency_list[current_node]:
            if neighbor_node not in visited_nodes:
                visited_nodes.add(neighbor_node)
                queue.append(neighbor_node)

    return traversal_order

if __name__ == "__main__":
    adjacency_list = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    result = breadth_first_search(adjacency_list, 'A')
    print(result)  # Output: ['A', 'B', 'C', 'D', 'E', 'F']