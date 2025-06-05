from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class Edge:
    u: int
    v: int
    weight: int




        for i in range(num_vertices):
            if not visitados[i]:
                topological_sort(i, visitados, stack, mapa)

        while stack:
            u = stack.pop()
            if distances[u] != -float('inf'):
                for v, weight in mapa[u]:
                    if distances[v] < distances[u] + weight:
                        distances[v] = distances[u] + weight

        return distances
    except Exception as e:
        logging.error(f"Error in find_longest_path function: {e}")
        return None

# Test cases

    num_vertices = 5
    source = 0
    longest_distances = find_longest_path(mapa, num_vertices, source)

    print(f"Longest distances from source vertex {source}:")
    for vertex, distancia in enumerate(longest_distances):
        if distancia == -float('inf'):
            print(f"Vertex {vertex} is unreachable from source, Distance: -Infinity")
        else:
            print(f"Vertex {vertex} - Distance: {distancia}")

if __name__ == "__main__":
    test_find_longest_path()


def test_find_longest_path():
    mapa = defaultdict(list)
    conexiones = [
        Edge(0, 1, 3),
        Edge(0, 2, 10),
        Edge(0, 3, 14),
        Edge(1, 3, 7),
        Edge(1, 4, 51),
        Edge(2, 3, 5),
        Edge(3, 4, 11)
    ]
    for edge in conexiones:
        crear_conexion(mapa, edge)


def find_longest_path(mapa, num_vertices, source):
    try:
        distances = [-float('inf')] * num_vertices
        distances[source] = 0
        stack = []
        visitados = [False] * num_vertices


def topological_sort(v, visitados, stack, mapa):
    visitados[v] = True
    for neighbor, _ in mapa[v]:
        if not visitados[neighbor]:
            topological_sort(neighbor, visitados, stack, mapa)
    stack.append(v)


def crear_conexion(mapa, edge: Edge):
    mapa[edge.u].append((edge.v, edge.weight))
