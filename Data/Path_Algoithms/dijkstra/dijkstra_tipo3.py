from typing import Tuple, List
from data_classes import Mapa
from logger_config import logger


        while unvisited:
            current = min(unvisited, key=unvisited.get)
            current_distance = unvisited[current]

            if current == fin:
                visitados[current] = current_distance
                break

            for neighbor, weight in mapa.conexiones.get(current, {}).items():
                distancia = current_distance + weight
                if distancia < unvisited.get(neighbor, float('inf')):
                    unvisited[neighbor] = distancia
                    camino[neighbor] = current

            visitados[current] = current_distance
            del unvisited[current]

        if fin not in visitados or visitados[fin] == float('inf'):
            raise ValueError("Path not found")

        route = []
        while fin:
            route.insert(0, fin)
            fin = camino.get(fin)

        return route, visitados[route[-1]]
    except Exception as e:
        logger.error(f"Error in Dijkstra's algorithm: {e}")
        raise 

def dijkstra(mapa: Mapa, inicio: str, fin: str) -> Tuple[List[str], int]:
    try:
        unvisited = {vertex: float('inf') for vertex in mapa.puntos}
        unvisited[inicio] = 0
        visitados = {}
        camino = {}
