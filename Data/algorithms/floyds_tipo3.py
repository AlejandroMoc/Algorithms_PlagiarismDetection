def floyds_algorithm(cost_matrix):
    n = len(cost_matrix)
    # Create a copy of the cost matrix to store shortest paths
    dist = [row[:] for row in cost_matrix]  # Make a deep copy

    k = 0
    while k < n:
        i = 0
        while i < n:
            j = 0
            while j < n:
                # Update the distance if a shorter path is found
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                j += 1
            i += 1
        k += 1

    return dist

def main_function():
    try:
        num_vertices = int(input("Enter the number of vertices: "))
        if num_vertices <= 0:
            raise ValueError("Number of vertices must be a positive integer.")

        print("Enter the cost matrix (use space-separated integers for each row):")
        cost_matrix = []
        i = 0
        while i < num_vertices:
            row = list(map(int, input(f"Enter row {i+1}: ").split()))
            if len(row) != num_vertices:
                raise ValueError("Each row must have the same number of elements as the number of vertices.")
            cost_matrix.append(row)
            i += 1

        # Floyd-Warshall algorithm to calculate shortest paths
        shortest_paths = floyds_algorithm(cost_matrix)

        # Output the shortest path matrix
        print("\nAll Pairs Shortest Paths:")
        i = 0
        while i < num_vertices:
            j = 0
            while j < num_vertices:
                if shortest_paths[i][j] == float('inf'):
                    print(f"{'∞':>3}", end="  ")
                else:
                    print(f"{shortest_paths[i][j]:>3}", end="  ")
                j += 1
            print()
            i += 1

    except ValueError as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main_function()