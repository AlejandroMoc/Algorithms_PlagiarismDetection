class Nodo:
    def __init__(self, vertice):
        self.vertice = vertice
        self.siguiente = None

def generar_rede(lista_adj, aristas):
    no_de_nodos = len(lista_adj)
    for i, vecinos in aristas.items():
        ultimo = None
        for val in vecinos:
            if val >= no_de_nodos or val < 0:
                raise ValueError(f"Valor de nodo inválido {val}. Debe estar entre 0 y {no_de_nodos - 1}.")
            nuevo_nodo = Nodo(val)
            if lista_adj[i] is None:
                lista_adj[i] = nuevo_nodo
            else:
                ultimo.siguiente = nuevo_nodo
            ultimo = nuevo_nodo

def mostrar_rede(lista_adj):
    resultado = []
    for i, nodo in enumerate(lista_adj):
        lista_nodos = [i]
        ptr = nodo
        while ptr:
            lista_nodos.append(ptr.vertice)
            ptr = ptr.siguiente
        resultado.append(lista_nodos)
    return resultado  # Devolver la lista de adyacencia como una estructura de lista anidada

def eliminar_rede(lista_adj):
    for i in range(len(lista_adj)):
        ptr = lista_adj[i]
        while ptr:
            temp = ptr
            ptr = ptr.siguiente
            del temp
        lista_adj[i] = None

if __name__ == "__main__":
    # Entrada de ejemplo para simular el ejemplo original de entrada manual
    no_de_nodos = 5
    aristas = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [2, 1, 4],
        4: [3]
    }

    # Inicializar lista de adyacencia con None para cada nodo
    lista_adj = [None] * no_de_nodos

    # Crear, mostrar y luego eliminar la red
    generar_rede(lista_adj, aristas)
    print("La lista de adyacencia es:")
    for linea in mostrar_rede(lista_adj):
        print(" --> ".join(map(str, linea)))
    eliminar_rede(lista_adj)