class Grafo:
    def __init__(self, vertices):
        self.V = vertices
        self.grafo = []

    def agregar_arista(self, u, v, w):
        self.grafo.append([u, v, w])

    def imprimir_solucion(self, dist):
        print("Vértice Distancia desde el Origen:")
        for i in range(self.V):
            print(f"{i}\t\t{dist[i]}")

    def imprimir_ciclo_negativo(self, padre, inicio):
        print("Ciclo de peso negativo detectado:")
        ciclo = []
        ciclo.append(inicio)
        actual = padre[inicio]
        while actual != inicio:
            ciclo.append(actual)
            actual = padre[actual]
        ciclo.append(inicio)
        ciclo.reverse()
        print(" -> ".join(map(str, ciclo)))

    def bellman_ford(self, src):
        dist = [float("Inf")] * self.V
        padre = [-1] * self.V
        dist[src] = 0

        for _ in range(self.V - 1):
            for u, v, w in self.grafo:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    padre[v] = u

        for u, v, w in self.grafo:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("El grafo contiene un ciclo de peso negativo")
                self.imprimir_ciclo_negativo(padre, v)
                return

        self.imprimir_solucion(dist)

# Uso de ejemplo:
g = Grafo(5)
g.agregar_arista(0, 1, -1)
g.agregar_arista(0, 2, 4)
g.agregar_arista(1, 2, 3)
g.agregar_arista(1, 3, 2)
g.agregar_arista(1, 4, 2)
g.agregar_arista(3, 2, 5)
g.agregar_arista(3, 1, 1)
g.agregar_arista(4, 3, -3)

g.bellman_ford(0)