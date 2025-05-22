import tkinter as tk
from dijkstra import dijkstra_algorithm
from utils import generate_network

def execute_example():
    nodes = ['X', 'Y', 'Z', 'W']
    connections = {
        'X': {'Y': 1, 'Z': 4},
        'Y': {'Z': 2, 'W': 5},
        'Z': {'W': 1},
        'W': {}
    }
    network = generate_network(nodes, connections)
    try:
        shortest_path, total_length = dijkstra_algorithm(network, 'X', 'W')
        display_message = f"Shortest Path: {' -> '.join(shortest_path)}\nTotal Length: {total_length}"
    except ValueError as ve:
        display_message = str(ve)
    except Exception:
        display_message = "An unexpected issue occurred."

    main_window = tk.Tk()
    main_window.title("Dijkstra Visualization")
    tk.Label(main_window, text=display_message).pack(pady=20)
    tk.Button(main_window, text="Close", command=main_window.destroy).pack(pady=10)
    main_window.mainloop()

if __name__ == "__main__":
    execute_example()