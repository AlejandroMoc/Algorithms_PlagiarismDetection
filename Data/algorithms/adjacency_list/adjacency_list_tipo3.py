class Nodo:
    def __init__(self, vertice):
        self.vertice = vertice
        self.siguiente = None

def crear_grafo(lista_ady, aristas):
    cant_nodos = len(lista_ady)
    for i, vecinos in aristas.items():
        ultimo = None
        for val in vecinos:
            if val >= cant_nodos or val < 0:
                raise ValueError(f"Valor de nodo inválido {val}. Debe estar entre 0 y {cant_nodos - 1}.")
            nuevo_nodo = Nodo(val)
            if lista_ady[i] is None:
                lista_ady[i] = nuevo_nodo
            else:
                ultimo.siguiente = nuevo_nodo
            ultimo = nuevo_nodo

def mostrar_grafo(lista_ady):
    resultado = []
    for i, nodo in enumerate(lista_ady):
        lista_nodo = [i]
        ptr = nodo
        while ptr:
            lista_nodo.append(ptr.vertice)
            ptr = ptr.siguiente
        resultado.append(lista_nodo)
    return resultado

def borrar_grafo(lista_ady):
    for i in range(len(lista_ady)):
        ptr = lista_ady[i]
        while ptr:
            temp = ptr
            ptr = ptr.siguiente
            del temp
        lista_ady[i] = None

if __name__ == "__main__":
    # Entrada de ejemplo para simular el ejemplo original de entrada manual
    cant_nodos = 5
    aristas = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [2, 1, 4],
        4: [3]
    }

    # Inicializar lista de adyacencia con None para cada nodo
    lista_ady = [None] * cant_nodos

    # Crear, mostrar y luego borrar el grafo
    crear_grafo(lista_ady, aristas)
    print("La lista de adyacencia es:")
    for linea in mostrar_grafo(lista_ady):
        print(" --> ".join(map(str, linea)))
    borrar_grafo(lista_ady)