from collections import deque

def bidirectional_search(graph_dict, initial, final):
    if not isinstance(graph_dict, dict):
        raise TypeError("Graph should be a dictionary with nodes as keys and lists of neighbors as values.")
    if initial == final:
        return [initial]
    if initial not in graph_dict or final not in graph_dict:
        raise ValueError(f"Start ({initial}) and target ({final}) nodes must exist in the graph.")

    queue_initial = deque([initial])
    queue_final = deque([final])

    visited_initial = {initial}
    visited_final = {final}

    parents_initial = {initial: None}
    parents_final = {final: None}

    while queue_initial and queue_final:
        path = bfs_search(graph_dict, visited_initial, queue_initial, parents_initial, visited_final)
        if path:
            return path

        path = bfs_search(graph_dict, visited_final, queue_final, parents_final, visited_initial)
        if path:
            return path

    return None

def bfs_search(graph_dict, visited, queue, parents, other_visited):
    current_node = queue.popleft()
    for neighbor in graph_dict[current_node]:
        if neighbor not in visited:
            parents[neighbor] = current_node
            visited.add(neighbor)
            queue.append(neighbor)

            if neighbor in other_visited:
                return reconstruct_path(parents, current_node, neighbor)
    return None

def reconstruct_path(parents, initial_node, meeting_node):
    path = []
    node = meeting_node
    while node is not None:
        path.append(node)
        node = parents[node]
    
    path.reverse()
    return path

if __name__ == "__main__":
    graph_dict = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    initial_node, final_node = 'A', 'F'
    path = bidirectional_search(graph_dict, initial_node, final_node)
    print(f"Shortest path from {initial_node} to {final_node}: {path}")