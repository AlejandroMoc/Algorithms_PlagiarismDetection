def depth_first_search(adjacency_list, starting_node, visited_set=None):
    if not isinstance(adjacency_list, dict):
        raise TypeError("Graph should be a dictionary with nodes as keys and lists of neighbors as values.")

    if starting_node not in adjacency_list:
        raise ValueError(f"Start node {starting_node} not found in the graph.")

    if visited_set is None:
        visited_set = set()

    visited_set.add(starting_node)
    print(starting_node, end=' ')

    for neighbor_node in adjacency_list[starting_node]:
        if neighbor_node not in visited_set:
            depth_first_search(adjacency_list, neighbor_node, visited_set)

    return list(visited_set)

if __name__ == "__main__":
    adjacency_list = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    visited_nodes = depth_first_search(adjacency_list, 'A')
    print("\nVisited Nodes:", visited_nodes)