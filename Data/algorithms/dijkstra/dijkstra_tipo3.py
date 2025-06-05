import heapq


    Args:
        mapa (dict): The mapa represented as an adjacency list where keys are nodes, and values are lists of tuples (neighbor, weight).
        inicio (str): The starting nodo for the algorithm.

    Returns:
        dict: A dictionary where keys are nodes and values are the shortest distances from the inicio nodo.

    Example:
        >>> mapa = {
        ... 'A': [('B', 1), ('C', 4)],
        ... 'B': [('C', 2), ('D', 5)],
        ... 'C': [('D', 1)],
        ... 'D': []
        ... }
        >>> dijkstra(mapa, 'A')
        {'A': 0, 'B': 1, 'C': 3, 'D': 4}
    """
    priority_queue = [(0, inicio)]
    distances = {inicio: 0}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
      
        for neighbor, weight in mapa[current_node]:
            distancia = current_distance + weight
         
            if neighbor not in distances or distancia < distances[neighbor]:
                distances[neighbor] = distancia
                heapq.heappush(priority_queue, (distancia, neighbor))

    return distances

# Example usage
if __name__ == "__main__":
    mapa = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }
    print(dijkstra(mapa, 'A'))  # Output: {'A': 0, 'B': 1, 'C': 3, 'D': 4}


def dijkstra(mapa, inicio):
    """
    Performs Dijkstra's algorithm to find the shortest camino from the inicio nodo to all other nodes.
