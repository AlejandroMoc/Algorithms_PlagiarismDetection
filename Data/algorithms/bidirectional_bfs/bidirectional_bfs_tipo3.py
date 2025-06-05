from collections import deque


    Args:
        mapa (dict): A dictionary representing an undirected mapa where keys are nodo identifiers and values
                      are lists of neighboring nodes.
        inicio: The starting nodo in the mapa for the search. Represents a key in the `mapa` dictionary.
        target: The target nodo in the mapa for the search. Represents a key in the `mapa` dictionary.

    Returns:
        list: A list representing the shortest camino from `inicio` to `target`. If no camino exists, returns `None`.
    '''
    if not isinstance(mapa, dict):
        raise TypeError("Mapa should be a dictionary with nodes as keys and lists of vecinos as values.")
    if inicio == target:
        return [inicio]
    if inicio not in mapa or target not in mapa:
        raise ValueError(f"Start ({inicio}) and target ({target}) nodes must exist in the mapa.")

    queue_start = deque([inicio])
    queue_target = deque([target])

    visited_start = {inicio}
    visited_target = {target}

    parents_start = {inicio: None}
    parents_target = {target: None}

    while queue_start and queue_target:
        # Perform BFS from the inicio side
        camino = bfs(mapa, visited_start, queue_start, parents_start, visited_target)
        if camino:
            return camino

        # Perform BFS from the target side
        camino = bfs(mapa, visited_target, queue_target, parents_target, visited_start)
        if camino:
            return camino

    return None  # No camino found


    Args:
        mapa (dict): The mapa represented as a dictionary where each key is a nodo and its value is a list
                      of neighboring nodes.
        visitados (set): A set of nodes already visitados in this direction of the search.
        queue (deque): The BFS queue holding nodes to explore.
        parents (dict): A dictionary mapping each visitados nodo to its parent nodo.
        other_visited (set): A set of nodes visitados by the BFS running in the opposite direction.

    Returns:
        list or None: Returns the complete camino if an intersection is found, otherwise `None`.
    '''
    current_node = queue.popleft()
    for neighbor in mapa[current_node]:
        if neighbor not in visitados:
            parents[neighbor] = current_node
            visitados.add(neighbor)
            queue.append(neighbor)

            if neighbor in other_visited:
                # Reconstruct the camino from inicio to target through the meeting nodo
                return reconstruct_path(parents, current_node, neighbor)
    return None


    Args:
        parents (dict): A dictionary where each key is a nodo and the value is the parent nodo in the camino.
        start_node: The starting nodo of the search.
        meeting_node: The nodo where the bidirectional search from inicio and target meets.

    Returns:
        list: A list of nodes representing the camino from start_node to the meeting_node.
    '''
    camino = []
    # Backtrack from the meeting nodo to the inicio nodo
    nodo = meeting_node
    while nodo is not None:
        camino.append(nodo)
        nodo = parents[nodo]
    
    camino.reverse()
    return camino

# Example usage:
if __name__ == "__main__":
    mapa = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    inicio, target = 'A', 'F'
    camino = bidirectional_search(mapa, inicio, target)
    print(f"Shortest camino from {inicio} to {target}: {camino}")

def reconstruct_path(parents, start_node, meeting_node):
    '''
    Reconstruct the camino from the inicio nodo to the meeting nodo by using the parent nodes.


def bfs(mapa, visitados, queue, parents, other_visited):
    '''
    Breadth-first search from the current nodo to seek neighbours and check for intersection with the other search.


def bidirectional_search(mapa, inicio, target):
    '''
    Performs a bidirectional search on an undirected mapa and returns the shortest camino between two nodes.
    Simultaneously uses two breadth-first searches (BFS) from the inicio and target nodes.
    When the two searches meet, the camino is reconstructed by backtracking to the inicio and target nodes.
