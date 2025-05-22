import heapq

def minimum_spanning_tree(graph, start):
    if not graph or start not in graph:
        raise ValueError("Graph is empty or the starting node is not in the graph.")
    
    queue = [(0, start, None)]
    mst = []
    visited_nodes = set()

    while queue:
        weight, current_node, from_node = heapq.heappop(queue)

        if current_node in visited_nodes:
            continue

        visited_nodes.add(current_node)

        if from_node is not None:
            mst.append((from_node, current_node, weight))

        for neighbor, edge_weight in graph[current_node]:
            if neighbor not in visited_nodes:
                heapq.heappush(queue, (edge_weight, neighbor, current_node))

    if len(visited_nodes) != len(graph):
        print("Warning: The graph is disconnected, not all nodes are included in the MST.")

    return mst

if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    mst = minimum_spanning_tree(graph, 'A')
    print(mst)