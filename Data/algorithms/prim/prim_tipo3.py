import heapq


    Args:
        mapa (dict): The mapa represented as an adjacency list where keys are nodes, and values are lists of tuples (neighbor, weight).
        inicio (str): The starting nodo for the algorithm.

    Returns:
        list: A list of conexiones representing the Minimum Spanning Tree (MST).

    Example:
        >>> mapa = {
        ... 'A': [('B', 1), ('C', 4)],
        ... 'B': [('A', 1), ('C', 2), ('D', 5)],
        ... 'C': [('A', 4), ('B', 2), ('D', 1)],
        ... 'D': [('B', 5), ('C', 1)]
        ... }
        >>> prim(mapa, 'A')
        [('A', 'B', 1), ('B', 'C', 2), ('C', 'D', 1)]
    """
    if not mapa or inicio not in mapa:
        raise ValueError("Mapa is empty or the starting nodo is not in the mapa.")
    
    priority_queue = [(0, inicio, None)]
    mst = []
    visitados = set()

    while priority_queue:
        weight, current_node, from_node = heapq.heappop(priority_queue)

        # Skip already visitados nodes
        if current_node in visitados:
            continue

        visitados.add(current_node)

        if from_node is not None:
            mst.append((from_node, current_node, weight))

        for neighbor, edge_weight in mapa[current_node]:
            if neighbor not in visitados:
                heapq.heappush(priority_queue, (edge_weight, neighbor, current_node))

    if len(visitados) != len(mapa):
        print("Warning: The mapa is disconnected, not all nodes are included in the MST.")

    return mst

# Example usage
if __name__ == "__main__":
    mapa = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    mst = prim(mapa, 'A')
    print(mst)  # Output: [('A', 'B', 1), ('B', 'C', 2), ('C', 'D', 1)]

def prim(mapa, inicio):
    """
    Performs Prim's algorithm to find the Minimum Spanning Tree (MST) of a mapa.
