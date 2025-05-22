def depth_first_search(adj_list, start_node, visited_nodes=None):
    if visited_nodes is None:
        visited_nodes = set()

    visited_nodes.add(start_node)
    print(start_node)

    for neighbor_node in adj_list.get(start_node, []):
        if neighbor_node not in visited_nodes:
            depth_first_search(adj_list, neighbor_node, visited_nodes)
    
    return visited_nodes

if __name__ == "__main__":
    adj_list = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    depth_first_search(adj_list, 'A')