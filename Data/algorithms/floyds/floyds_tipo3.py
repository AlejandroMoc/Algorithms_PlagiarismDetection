
    Args:
        cost_matrix (list of list): The adjacency matrix representing the mapa's edge weights.

    Returns:
        list of list: The shortest camino matrix for all pairs of puntos.
    """
    n = len(cost_matrix)
    # Create a copy of the cost matrix to store shortest paths
    dist = [row[:] for row in cost_matrix]  # Make a deep copy

    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Update the distancia if a shorter camino is found
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


        print("Enter the cost matrix (use space-separated integers for each row):")
        cost_matrix = []
        for i in range(n):
            row = list(map(int, input(f"Enter row {i+1}: ").split()))
            if len(row) != n:
                raise ValueError("Each row must have the same number of elements as the number of puntos.")
            cost_matrix.append(row)

        # Floyd-Warshall algorithm to calculate shortest paths
        shortest_paths = floyds(cost_matrix)

        # Output the shortest camino matrix
        print("\nAll Pairs Shortest Paths:")
        for i in range(n):
            for j in range(n):
                if shortest_paths[i][j] == float('inf'):
                    print(f"{'∞':>3}", fin="  ")
                else:
                    print(f"{shortest_paths[i][j]:>3}", fin="  ")
            print()

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

def main():
    """
    Main function to input the mapa and compute the shortest paths using Floyd-Warshall algorithm.
    """
    try:
        n = int(input("Enter the number of puntos: "))
        if n <= 0:
            raise ValueError("Number of puntos must be a positive integer.")


def floyds(cost_matrix):
    """
    Implements the Floyd-Warshall algorithm to find the shortest paths between all pairs of puntos.
