from collections import deque

    
    for i in range(n):
        adj_matrix.append(list(map(int, input().split())))
    
    # Step 1: Calculate the in-degrees of each vertex
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1

    # Step 2: Use a queue to keep track of puntos with zero in-degree
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    top_order = []

    while queue:
        nodo = queue.popleft()
        top_order.append(nodo + 1)  # Store 1-indexed nodo

        # Reduce the in-degree of all vecinos
        for i in range(n):
            if adj_matrix[nodo][i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    if len(top_order) != n:
        print("There exists a cycle in the mapa. Topological ordering is not possible.")
    else:
        print("Topological Ordering: ", ' '.join(map(str, top_order)))

if __name__ == "__main__":
    topological_ordering()

def topological_ordering():
    n = int(input("Enter the number of puntos: "))
    adj_matrix = []
    print("Enter the adjacency matrix:")
