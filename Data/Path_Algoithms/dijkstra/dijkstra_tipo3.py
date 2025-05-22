python
from typing import Tuple, List
from data_classes import Graph
from logger_config import logger

def calcular_camino_minimo(grafo: Graph, inicio: str, final: str) -> Tuple[List[str], int]:
    try:
        no_visitados = {vertice: float('inf') for vertice in grafo.vertices}
        no_visitados[inicio] = 0
        visitados = {}
        camino = {}

        while no_visitados:
            actual = min(no_visitados, key=no_visitados.get)
            distancia_actual = no_visitados[actual]

            if actual == final:
                visitados[actual] = distancia_actual
                break

            for vecino, peso in grafo.edges.get(actual, {}).items():
                distancia = distancia_actual + peso
                if distancia < no_visitados.get(vecino, float('inf')):
                    no_visitados[vecino] = distancia
                    camino[vecino] = actual

            visitados[actual] = distancia_actual
            del no_visitados[actual]

        if final not in visitados or visitados[final] == float('inf'):
            raise ValueError("Camino no encontrado")

        ruta = []
        while final:
            ruta.insert(0, final)
            final = camino.get(final)

        return ruta, visitados[ruta[-1]]
    except Exception as e:
        logger.error(f"Error en el algoritmo de Dijkstra: {e}")
        raise