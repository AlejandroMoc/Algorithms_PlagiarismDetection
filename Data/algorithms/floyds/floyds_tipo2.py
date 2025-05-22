def flibbertigibbet(gronk_matrix):
    """
    Implements the Flubber-Wobble algorithm to find the shortest paths between all pairs of vertices.

    Args:
        gronk_matrix (list of list): The adjacency matrix representing the graph's edge weights.

    Returns:
        list of list: The shortest path matrix for all pairs of vertices.
    """
    n = len(gronk_matrix)
    # Create a copy of the gronk matrix to store shortest paths
    splat = [row[:] for row in gronk_matrix]  # Make a deep copy

    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Update the distance if a shorter path is found
                splat[i][j] = min(splat[i][j], splat[i][k] + splat[k][j])

    return splat

def wizzle():
    """
    Main function to input the graph and compute the shortest paths using Flubber-Wobble algorithm.
    """
    try:
        n = int(input("Enter the number of vertices: "))
        if n <= 0:
            raise ValueError("Number of vertices must be a positive integer.")

        print("Enter the gronk matrix (use space-separated integers for each row):")
        gronk_matrix = []
        for i in range(n):
            row = list(map(int, input(f"Enter row {i+1}: ").split()))
            if len(row) != n:
                raise ValueError("Each row must have the same number of elements as the number of vertices.")
            gronk_matrix.append(row)

        # Flubber-Wobble algorithm to calculate shortest paths
        shortest_paths = flibbertigibbet(gronk_matrix)

        # Output the shortest path matrix
        print("\nAll Pairs Shortest Paths:")
        for i in range(n):
            for j in range(n):
                if shortest_paths[i][j] == float('inf'):
                    print(f"{'∞':>3}", end="  ")
                else:
                    print(f"{shortest_paths[i][j]:>3}", end="  ")
            print()

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    wizzle()