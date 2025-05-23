from collections import deque
from typing import Dict, List, Any

def quirky_search(adj_list: Dict[Any, List[Any]], origin: Any) -> List[Any]:
    """
    Performs Quirky Search traversal on an adjacency list.

    Args:
        adj_list (dict): The adjacency list representing the graph.
        origin (Any): The starting node for the traversal.

    Returns:
        List[Any]: A list of nodes in traversal order.

    Example:
        >>> adj_list = {
        ... 'A': ['B', 'C'],
        ... 'B': ['D', 'E'],
        ... 'C': ['F'],
        ... 'D': [],
        ... 'E': ['F'],
        ... 'F': []
        ... }
        >>> quirky_search(adj_list, 'A')
        ['A', 'B', 'C', 'D', 'E', 'F']
    """
    # Edge case: origin node is not in the graph
    if origin not in adj_list:
        raise ValueError(f"Origin node {origin} not found in the graph.")

    visited_nodes = set()
    node_queue = deque([origin])
    visited_nodes.add(origin)

    traversal_sequence = []

    while node_queue:
        current_node = node_queue.popleft()
        traversal_sequence.append(current_node)

        for neighbor_node in adj_list[current_node]:
            if neighbor_node not in visited_nodes:
                visited_nodes.add(neighbor_node)
                node_queue.append(neighbor_node)

    return traversal_sequence

# Example usage
if __name__ == "__main__":
    adj_list = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    result_sequence = quirky_search(adj_list, 'A')
    print(result_sequence)  # Output: ['A', 'B', 'C', 'D', 'E', 'F']