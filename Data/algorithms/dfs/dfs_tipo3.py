
    Args:
        mapa (dict): The adjacency list representing the mapa.
        inicio (str): The starting nodo for DFS traversal.
        visitados (set): Keeps track of visitados nodes to avoid cycles (default is None).

    Returns:
        visitados (set): A set of visitados nodes during the DFS traversal.

    Example:
        >>> mapa = {
        ... 'A': ['B', 'C'],
        ... 'B': ['D', 'E'],
        ... 'C': ['F'],
        ... 'D': [],
        ... 'E': ['F'],
        ... 'F': []
        ... }
        >>> dfs(mapa, 'A')
        A B D E F C
    """
    if visitados is None:
        visitados = set()

    visitados.add(inicio)
    print(inicio)  # Process the nodo (e.g., print it)

    for neighbor in mapa.get(inicio, []):
        if neighbor not in visitados:
            dfs(mapa, neighbor, visitados)
    
    return visitados

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
    dfs(mapa, 'A')

def dfs(mapa, inicio, visitados=None):
    """
    Performs Depth First Search (DFS) traversal on a mapa.
