from collections import deque

def quirky_sequence():
    vertices_num = int(input("Enter the number of vertices: "))
    crazy_matrix = []
    print("Enter the adjacency matrix:")
    
    for i in range(vertices_num):
        crazy_matrix.append(list(map(int, input().split())))
    
    # Step 1: Calculate the in-degrees of each vertex
    in_degrees = [0] * vertices_num
    for i in range(vertices_num):
        for j in range(vertices_num):
            if crazy_matrix[i][j] == 1:
                in_degrees[j] += 1

    # Step 2: Use a queue to keep track of vertices with zero in-degree
    funky_queue = deque()
    for i in range(vertices_num):
        if in_degrees[i] == 0:
            funky_queue.append(i)
    
    unique_order = []

    while funky_queue:
        vertex = funky_queue.popleft()
        unique_order.append(vertex + 1)  # Store 1-indexed node

        # Reduce the in-degree of all neighbors
        for i in range(vertices_num):
            if crazy_matrix[vertex][i] == 1:
                in_degrees[i] -= 1
                if in_degrees[i] == 0:
                    funky_queue.append(i)

    if len(unique_order) != vertices_num:
        print("There exists a cycle in the graph. Quirky sequence is not possible.")
    else:
        print("Quirky Sequence: ", ' '.join(map(str, unique_order)))

if __name__ == "__main__":
    quirky_sequence()