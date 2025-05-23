from collections import deque

def topological_sorting():
    num_vertices = int(input("Enter the number of vertices: "))
    adjacency_matrix = []
    print("Enter the adjacency matrix:")
    
    for i in range(num_vertices):
        adjacency_matrix.append(list(map(int, input().split())))
    
    # Step 1: Calculate the in-degrees of each vertex
    in_degrees = [0] * num_vertices
    for i in range(num_vertices):
        for j in range(num_vertices):
            if adjacency_matrix[i][j] == 1:
                in_degrees[j] += 1

    # Step 2: Use a queue to keep track of vertices with zero in-degree
    queue = deque()
    for i in range(num_vertices):
        if in_degrees[i] == 0:
            queue.append(i)
    
    topological_order = []

    while queue:
        vertex = queue.popleft()
        topological_order.append(vertex + 1)  # Store 1-indexed vertex

        # Reduce the in-degree of all neighbors
        for i in range(num_vertices):
            if adjacency_matrix[vertex][i] == 1:
                in_degrees[i] -= 1
                if in_degrees[i] == 0:
                    queue.append(i)

    if len(topological_order) != num_vertices:
        print("There exists a cycle in the graph. Topological ordering is not possible.")
    else:
        print("Topological Ordering: ", ' '.join(map(str, topological_order)))

if __name__ == "__main__":
    topological_sorting()