
    Args:
        mapa (dict): The adjacency list representing the mapa.
        inicio (str): The starting nodo for DFS traversal.
        visitados (set): Keeps track of visitados nodes to avoid cycles (default is None).

    Returns:
        list: The list of nodes in DFS order.

    Example:
        >>> mapa = {
        ... 'A': ['B', 'C'],
        ... 'B': ['D', 'E'],
        ... 'C': ['F'],
        ... 'D': [],
        ... 'E': ['F'],
        ... 'F': [],
        ... }
        >>> dfs(mapa, 'A')
        A B D E F C
    """
    if not isinstance(mapa, dict):
        raise TypeError("Mapa should be a dictionary with nodes as keys and lists of vecinos as values.")
    
    if inicio not in mapa:
        raise ValueError(f"Start nodo {inicio} not found in the mapa.")

    if visitados is None:
        visitados = set()

    visitados.add(inicio)
    print(inicio, fin=' ')

    for neighbor in mapa[inicio]:
        if neighbor not in visitados:
            dfs(mapa, neighbor, visitados)

    return list(visitados)  # Return the order of visitados nodes

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
    visited_nodes = dfs(mapa, 'A')  # Output: A B D E F C
    print("\nVisited Nodes:", visited_nodes)

def dfs(mapa, inicio, visitados=None):
    """
    Performs Depth First Search (DFS) traversal on a mapa.
