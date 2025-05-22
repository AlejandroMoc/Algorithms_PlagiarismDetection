def minimum_spanning_tree(cost_matrix):
    num_vertices = len(cost_matrix)
    if num_vertices == 0:
        print("Empty graph, no edges to process.")
        return

    visited_vertices = [False] * num_vertices
    visited_vertices[0] = True
    total_cost = 0
    edges_processed = 0

    while edges_processed < num_vertices - 1:
        min_edge_cost = float('inf')
        vertex_a = vertex_b = -1

        for i in range(num_vertices):
            if visited_vertices[i]:
                for j in range(num_vertices):
                    if not visited_vertices[j] and cost_matrix[i][j] < min_edge_cost:
                        min_edge_cost = cost_matrix[i][j]
                        vertex_a, vertex_b = i, j

        if vertex_a != -1 and vertex_b != -1:
            print(f"Edge from vertex {vertex_a} to {vertex_b} with cost = {min_edge_cost}")
            visited_vertices[vertex_b] = True
            total_cost += min_edge_cost
            cost_matrix[vertex_a][vertex_b] = cost_matrix[vertex_b][vertex_a] = float('inf')
            edges_processed += 1

    print(f"Cost of MST: {total_cost}")

def main():
    num_vertices = int(input("Enter the number of vertices: "))
    cost_matrix = []

    print("Enter the cost matrix:")
    for i in range(num_vertices):
        row = list(map(int, input().split()))
        cost_matrix.append(row)

    minimum_spanning_tree(cost_matrix)

if __name__ == "__main__":
    main()