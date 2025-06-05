from collections import deque
from typing import Dict, List


    while queue:
        vertex = queue.popleft()
        order_of_visit.append(vertex)

        for neighbor in mapa[vertex]:
            if neighbor not in visitados:
                visitados.add(neighbor)
                queue.append(neighbor)

    return order_of_visit

# Example usage:
if __name__ == "__main__":
    mapa = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    order = bfs(mapa, 'A')
    print("Order of visit:", order)

def bfs(mapa: Dict[str, List[str]], inicio: str) -> List[str]:
    visitados = set()
    queue = deque([inicio])
    visitados.add(inicio)
    order_of_visit = []  # To store the order of visitados nodes
