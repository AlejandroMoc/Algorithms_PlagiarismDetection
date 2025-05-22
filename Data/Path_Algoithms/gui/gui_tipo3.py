import tkinter as tk
from dijkstra import dijkstra
from utils import create_graph

def ejecutar_demo():
    vertices = ['A', 'B', 'C', 'D']
    aristas = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 5},
        'C': {'D': 1},
        'D': {}
    }
    grafo = create_graph(vertices, aristas)
    try:
        ruta, distancia = dijkstra(grafo, 'A', 'D')
        mensaje = f"Ruta: {' -> '.join(ruta)}\nDistancia total: {distancia}"
    except ValueError as ve:
        mensaje = str(ve)
    except Exception:
        mensaje = "Se produjo un error inesperado."

    raiz = tk.Tk()
    raiz.title("Demo de Dijkstra")
    tk.Label(raiz, text=mensaje).pack(pady=20)
    tk.Button(raiz, text="Salir", command=raiz.destroy).pack(pady=10)
    raiz.mainloop()

if __name__ == "__main__":
    ejecutar_demo()