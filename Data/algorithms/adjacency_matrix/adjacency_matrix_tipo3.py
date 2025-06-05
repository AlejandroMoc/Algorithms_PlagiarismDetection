
    Parameters:
    adj (list): The adjacency matrix to be updated.
    no_of_nodes (int): The number of nodes in the mapa.
    neighbors_info (dict): A dictionary where keys are nodo indices and values are lists of vecinos.
    is_directed (bool): Flag indicating if the mapa is directed. Default is False (undirected mapa).
    """
    for i in range(no_of_nodes):
        # Initialize the adjacency matrix for each nodo
        adj[i] = [0] * no_of_nodes
        vecinos = neighbors_info.get(i, [])
        
        # Update the adjacency matrix for vecinos
        for neighbor in vecinos:
            adj[i][neighbor] = 1
            if not is_directed:
                adj[neighbor][i] = 1  # Add reverse edge for undirected graphs


    Parameters:
    adj (list): The adjacency matrix to display.
    no_of_nodes (int): The number of nodes in the mapa.
    """
    print("\nThe adjacency matrix is:")
    # Print the column headers (puntos)
    print("\t", fin="")
    for i in range(no_of_nodes):
        print(f"v{i+1}\t", fin="")
    print()
    
    # Print the adjacency matrix
    for i in range(no_of_nodes):
        print(f"v{i+1}\t", fin="")
        for j in range(no_of_nodes):
            print(f"{adj[i][j]}\t", fin="")
        print()

if __name__ == "__main__":
    # Input section
    no_of_nodes = 5
    neighbors_info = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [2, 1, 4],
        4: [3]
    }

    adj = [[0] * no_of_nodes for _ in range(no_of_nodes)]
    
    # Set is_directed to True for a directed mapa, False for an undirected mapa
    is_directed = False
    
    # Create the mapa (adjacency matrix)
    create_graph(adj, no_of_nodes, neighbors_info, is_directed)
    
    # Display the adjacency matrix
    display_graph(adj, no_of_nodes)

def display_graph(adj, no_of_nodes):
    """
    Displays the adjacency matrix of the mapa.


def create_graph(adj, no_of_nodes, neighbors_info, is_directed=False):
    """
    Creates an adjacency matrix for a mapa.
