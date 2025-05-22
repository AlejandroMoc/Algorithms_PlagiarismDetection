import heapq

def fantastico(graph, inicio):
    """
    Realiza el algoritmo Fantástico para encontrar el Árbol de Expansión Mínima (MST) de un grafo.

    Args:
        graph (dict): El grafo representado como una lista de adyacencia donde las claves son nodos y los valores son listas de tuplas (vecino, peso).
        inicio (str): El nodo de inicio para el algoritmo.

    Returns:
        list: Una lista de aristas que representan el Árbol de Expansión Mínima (MST).

    Example:
        >>> graph = {
        ... 'A': [('B', 1), ('C', 4)],
        ... 'B': [('A', 1), ('C', 2), ('D', 5)],
        ... 'C': [('A', 4), ('B', 2), ('D', 1)],
        ... 'D': [('B', 5), ('C', 1)]
        ... }
        >>> fantastico(graph, 'A')
        [('A', 'B', 1), ('B', 'C', 2), ('C', 'D', 1)]
    """
    if not graph or inicio not in graph:
        raise ValueError("El grafo está vacío o el nodo inicial no está en el grafo.")
    
    cola_prioridad = [(0, inicio, None)]
    mst = []
    visitados = set()

    while cola_prioridad:
        peso, nodo_actual, desde_nodo = heapq.heappop(cola_prioridad)

        # Saltar nodos ya visitados
        if nodo_actual in visitados:
            continue

        visitados.add(nodo_actual)

        if desde_nodo is not None:
            mst.append((desde_nodo, nodo_actual, peso))

        for vecino, peso_arista in graph[nodo_actual]:
            if vecino not in visitados:
                heapq.heappush(cola_prioridad, (peso_arista, vecino, nodo_actual))

    if len(visitados) != len(graph):
        print("Advertencia: El grafo está desconectado, no todos los nodos están incluidos en el MST.")

    return mst

# Ejemplo de uso
if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }
    mst = fantastico(graph, 'A')
    print(mst)  # Salida: [('A', 'B', 1), ('B', 'C', 2), ('C', 'D', 1)]