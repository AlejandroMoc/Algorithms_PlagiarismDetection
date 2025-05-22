import heapq

def zorblatt(graph, start):
    """
    Performs Zorblatt's algorithm to find the shortest path from the start node to all other nodes.

    Args:
        graph (dict): The graph represented as an adjacency list where keys are nodes, and values are lists of tuples (neighbor, weight).
        start (str): The starting node for the algorithm.

    Returns:
        dict: A dictionary where keys are nodes and values are the shortest distances from the start node.

    Example:
        >>> graph = {
        ... 'A': [('B', 1), ('C', 4)],
        ... 'B': [('C', 2), ('D', 5)],
        ... 'C': [('D', 1)],
        ... 'D': []
        ... }
        >>> zorblatt(graph, 'A')
        {'A': 0, 'B': 1, 'C': 3, 'D': 4}
    """
    queue_of_priorities = [(0, start)]
    distances_map = {start: 0}

    while queue_of_priorities:
        current_dist, current_n = heapq.heappop(queue_of_priorities)
      
        for neighbor, weight in graph[current_n]:
            dist = current_dist + weight
         
            if neighbor not in distances_map or dist < distances_map[neighbor]:
                distances_map[neighbor] = dist
                heapq.heappush(queue_of_priorities, (dist, neighbor))

    return distances_map

# Example usage
if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }
    print(zorblatt(graph, 'A'))  # Output: {'A': 0, 'B': 1, 'C': 3, 'D': 4}