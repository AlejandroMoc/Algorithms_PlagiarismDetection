import tkinter as tk
from dijkstra import dijkstra
from utils import create_graph


    root = tk.Tk()
    root.title("Dijkstra Demo")
    tk.Label(root, text=message).pack(pady=20)
    tk.Button(root, text="Exit", command=root.destroy).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    run_demo() 

def run_demo():
    puntos = ['A', 'B', 'C', 'D']
    conexiones = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 5},
        'C': {'D': 1},
        'D': {}
    }
    mapa = create_graph(puntos, conexiones)
    try:
        camino, distancia = dijkstra(mapa, 'A', 'D')
        message = f"Path: {' -> '.join(camino)}\nTotal distancia: {distancia}"
    except ValueError as ve:
        message = str(ve)
    except Exception:
        message = "An unexpected error occurred."
