from collections import deque
from typing import Dict, List

def breadth_first_search(graph: Dict[str, List[str]], start_node: str) -> List[str]:
    visited_nodes = set()
    queue_nodes = deque([start_node])
    visited_nodes.add(start_node)
    order_visited = []  # To store the order of visited nodes

    while queue_nodes:
        current_node = queue_nodes.popleft()
        order_visited.append(current_node)

        for neighbor_node in graph[current_node]:
            if neighbor_node not in visited_nodes:
                visited_nodes.add(neighbor_node)
                queue_nodes.append(neighbor_node)

    return order_visited

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    order = breadth_first_search(graph, 'A')
    print("Order of visit:", order)