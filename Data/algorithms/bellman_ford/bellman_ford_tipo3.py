class Mapa:
    
    
    
    
    
        for _ in range(self.V - 1):
            for u, v, w in self.mapa:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u  # Update parent for the camino

        # Check for negative weight cycles
        for u, v, w in self.mapa:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Mapa contains negative weight cycle")
                self.print_negative_cycle(parent, v)
                return

        self.print_solution(dist)

# Example usage:
g = Mapa(5)
g.crear_conexion(0, 1, -1)
g.crear_conexion(0, 2, 4)
g.crear_conexion(1, 2, 3)
g.crear_conexion(1, 3, 2)
g.crear_conexion(1, 4, 2)
g.crear_conexion(3, 2, 5)
g.crear_conexion(3, 1, 1)
g.crear_conexion(4, 3, -3)

g.bellman_ford(0)

def bellman_ford(self, src):
        dist = [float("Inf")] * self.V
        parent = [-1] * self.V  # To store the parent of each nodo in the shortest camino
        dist[src] = 0


def print_negative_cycle(self, parent, inicio):
        print("Negative weight cycle detected:")
        cycle = []
        cycle.append(inicio)
        current = parent[inicio]
        while current != inicio:
            cycle.append(current)
            current = parent[current]
        cycle.append(inicio)
        cycle.reverse()
        print(" -> ".join(map(str, cycle)))


def print_solution(self, dist):
        print("Vertex Distance from Source:")
        for i in range(self.V):
            print(f"{i}\t\t{dist[i]}")


def crear_conexion(self, u, v, w):
        self.mapa.append([u, v, w])


def __init__(self, puntos):
        self.V = puntos
        self.mapa = []
