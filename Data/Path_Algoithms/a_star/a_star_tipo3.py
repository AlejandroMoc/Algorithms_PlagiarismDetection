from typing import Tuple, List, Dict
from heapq import heappop, heappush
import math
from data_classes import Mapa
from logger_config import logger


        raise

def a_star(mapa: Mapa, inicio: Tuple[int, int], fin: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
    try:
        open_set = []
        heappush(open_set, (0, inicio))
        g_score: Dict[Tuple[int, int], float] = {vertex: float('inf') for vertex in mapa.puntos}
        g_score[inicio] = 0
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == fin:
                camino = []
                while current in came_from:
                    camino.insert(0, current)
                    current = came_from[current]
                camino.insert(0, inicio)
                return camino, g_score[fin]
            
            for neighbor, weight in mapa.conexiones.get(current, {}).items():
                tentative_g_score = g_score[current] + weight
                if tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, fin)
                    heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
        
        raise ValueError("Path not found")
    except Exception as e:
        logger.error(f"Error in A* algorithm: {e}")


def heuristic(nodo: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """Euclidean distancia heuristic."""
    return math.sqrt((nodo[0] - goal[0])**2 + (nodo[1] - goal[1])**2)
