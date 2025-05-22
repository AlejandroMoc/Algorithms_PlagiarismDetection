import heapq

class Nodo:
    def __init__(self, posicion, padre=None):
        self.posicion = posicion
        self.padre = padre
        self.g = 0  # Costo desde el inicio hasta el nodo actual
        self.h = 0  # Costo heurístico desde el nodo actual hasta el final
        self.f = 0  # Costo total (g + h)

    def __eq__(self, otro):
        return self.posicion == otro.posicion

    def __lt__(self, otro):
        return self.f < otro.f

def astar(maze, inicio, final, permitir_diagonal=False):
    nodo_inicio = Nodo(inicio)
    nodo_final = Nodo(final)

    lista_abierta = []
    lista_cerrada = set()
    diccionario_abierto = {}  # Diccionario para hacer seguimiento de nodos y sus valores 'g'

    heapq.heappush(lista_abierta, nodo_inicio)
    diccionario_abierto[nodo_inicio.posicion] = nodo_inicio

    while lista_abierta:
        nodo_actual = heapq.heappop(lista_abierta)
        lista_cerrada.add(nodo_actual.posicion)

        if nodo_actual == nodo_final:
            # Reconstruir el camino
            camino = []
            while nodo_actual:
                camino.append(nodo_actual.posicion)
                nodo_actual = nodo_actual.padre
            return camino[::-1]  # Devolver el camino en el orden correcto

        hijos = []
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Arriba, Abajo, Izquierda, Derecha
        if permitir_diagonal:
            # Incluir movimientos diagonales (arriba-izquierda, arriba-derecha, abajo-izquierda, abajo-derecha)
            direcciones += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Generar hijos (vecinos)
        for nueva_posicion in direcciones:
            posicion_nodo = (nodo_actual.posicion[0] + nueva_posicion[0], nodo_actual.posicion[1] + nueva_posicion[1])

            # Saltar celdas fuera de límites o bloqueadas (valor diferente de 0 significa pared)
            if posicion_nodo[0] > (len(maze) - 1) or posicion_nodo[0] < 0 or posicion_nodo[1] > (len(maze[0]) - 1) or posicion_nodo[1] < 0:
                continue
            if maze[posicion_nodo[0]][posicion_nodo[1]] != 0:
                continue

            nuevo_nodo = Nodo(posicion_nodo, nodo_actual)
            hijos.append(nuevo_nodo)

        for hijo in hijos:
            if hijo.posicion in lista_cerrada:
                continue  # Ignorar nodos evaluados

            hijo.g = nodo_actual.g + 1  # El valor g es la distancia desde el inicio
            # Usar la distancia de Manhattan como heurística (para movimientos no diagonales)
            hijo.h = abs(hijo.posicion[0] - nodo_final.posicion[0]) + abs(hijo.posicion[1] - nodo_final.posicion[1])
            hijo.f = hijo.g + hijo.h

            # Verificar si este nodo debe ser agregado a lista_abierta o saltado
            if hijo.posicion not in diccionario_abierto or hijo.g < diccionario_abierto[hijo.posicion].g:
                diccionario_abierto[hijo.posicion] = hijo
                heapq.heappush(lista_abierta, hijo)

    return None  # Devolver None si no se encuentra un camino

# Uso de ejemplo:
laberinto = [
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

inicio = (0, 0)
final = (4, 5)

# Llamada al algoritmo A* con movimiento diagonal permitido
camino = astar(laberinto, inicio, final, permitir_diagonal=True)
print("Camino:", camino)