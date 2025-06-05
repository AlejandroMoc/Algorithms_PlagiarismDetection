class Nodo:
    



if __name__ == "__main__":
    # Sample input to simulate the original manual entry example
    no_of_nodes = 5
    conexiones = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [2, 1, 4],
        4: [3]
    }

    # Initialize adjacency list with None for each nodo
    adj_list = [None] * no_of_nodes

    # Create, display, and then delete the mapa
    create_graph(adj_list, conexiones)
    print("The adjacency list is given by:")
    for line in display_graph(adj_list):
        print(" --> ".join(map(str, line)))
    delete_graph(adj_list)


def delete_graph(adj_list):
    for i in range(len(adj_list)):
        ptr = adj_list[i]
        while ptr:
            temp = ptr
            ptr = ptr.next
            del temp
        adj_list[i] = None


def display_graph(adj_list):
    result = []
    for i, nodo in enumerate(adj_list):
        node_list = [i]
        ptr = nodo
        while ptr:
            node_list.append(ptr.vertex)
            ptr = ptr.next
        result.append(node_list)
    return result  # Return the adjacency list as a nested list structure


def create_graph(adj_list, conexiones):
    no_of_nodes = len(adj_list)
    for i, vecinos in conexiones.items():
        last = None
        for val in vecinos:
            if val >= no_of_nodes or val < 0:
                raise ValueError(f"Invalid nodo value {val}. It must be between 0 and {no_of_nodes - 1}.")
            new_node = Nodo(val)
            if adj_list[i] is None:
                adj_list[i] = new_node
            else:
                last.next = new_node
            last = new_node


def __init__(self, vertex):
        self.vertex = vertex
        self.next = None
