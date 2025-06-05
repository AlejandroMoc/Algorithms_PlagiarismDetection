from collections import deque
from typing import Dict, List, Any


    Args:
        mapa (dict): The adjacency list representing the mapa.
        inicio (Any): The starting nodo for BFS traversal.

    Returns:
        List[Any]: A list of nodes in BFS order during traversal.

    Example:
        >>> mapa = {
        ... 'A': ['B', 'C'],
        ... 'B': ['D', 'E'],
        ... 'C': ['F'],
        ... 'D': [],
        ... 'E': ['F'],
        ... 'F': []
        ... }
        >>> bfs(mapa, 'A')
        ['A', 'B', 'C', 'D', 'E', 'F']
    """
    # Edge case: inicio nodo is not in the mapa
    if inicio not in mapa:
        raise ValueError(f"Start nodo {inicio} not found in the mapa.")

    visitados = set()
    queue = deque([inicio])
    visitados.add(inicio)

    traversal_order = []

    while queue:
        vertex = queue.popleft()
        traversal_order.append(vertex)

        for neighbor in mapa[vertex]:
            if neighbor not in visitados:
                visitados.add(neighbor)
                queue.append(neighbor)

    return traversal_order

# Example usage
if __name__ == "__main__":
    mapa = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    result = bfs(mapa, 'A')
    print(result)  # Output: ['A', 'B', 'C', 'D', 'E', 'F']

def bfs(mapa: Dict[Any, List[Any]], inicio: Any) -> List[Any]:
    """
    Performs Breadth First Search (BFS) traversal on a mapa.
