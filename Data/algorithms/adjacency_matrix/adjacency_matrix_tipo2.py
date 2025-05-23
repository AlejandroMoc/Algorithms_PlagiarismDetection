def generate_matrix(connections, num_nodes, info_neighbors, directed=False):
    """
    Creates an adjacency matrix for a graph.

    Parameters:
    connections (list): The adjacency matrix to be updated.
    num_nodes (int): The number of nodes in the graph.
    info_neighbors (dict): A dictionary where keys are node indices and values are lists of neighbors.
    directed (bool): Flag indicating if the graph is directed. Default is False (undirected graph).
    """
    for i in range(num_nodes):
        # Initialize the adjacency matrix for each node
        connections[i] = [0] * num_nodes
        neighbors = info_neighbors.get(i, [])
        
        # Update the adjacency matrix for neighbors
        for neighbor in neighbors:
            connections[i][neighbor] = 1
            if not directed:
                connections[neighbor][i] = 1  # Add reverse edge for undirected graphs

def visualize_matrix(connections, num_nodes):
    """
    Displays the adjacency matrix of the graph.

    Parameters:
    connections (list): The adjacency matrix to display.
    num_nodes (int): The number of nodes in the graph.
    """
    print("\nThe adjacency matrix is:")
    # Print the column headers (vertices)
    print("\t", end="")
    for i in range(num_nodes):
        print(f"v{i+1}\t", end="")
    print()
    
    # Print the adjacency matrix
    for i in range(num_nodes):
        print(f"v{i+1}\t", end="")
        for j in range(num_nodes):
            print(f"{connections[i][j]}\t", end="")
        print()

if __name__ == "__main__":
    # Input section
    num_nodes = 5
    info_neighbors = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [2, 1, 4],
        4: [3]
    }

    connections = [[0] * num_nodes for _ in range(num_nodes)]
    
    # Set directed to True for a directed graph, False for an undirected graph
    directed = False
    
    # Create the graph (adjacency matrix)
    generate_matrix(connections, num_nodes, info_neighbors, directed)
    
    # Display the adjacency matrix
    visualize_matrix(connections, num_nodes)