from typing import Tuple, List, Dict
from heapq import heappop, heappush
import math
from data_classes import Graph
from logger_config import logger

def funky_distance(coordinate: Tuple[int, int], target: Tuple[int, int]) -> float:
    """Funky distance calculation."""
    return math.sqrt((coordinate[0] - target[0])**2 + (coordinate[1] - target[1])**2)

def zippy_search(graph: Graph, origin: Tuple[int, int], destination: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
    try:
        active_set = []
        heappush(active_set, (0, origin))
        funky_score: Dict[Tuple[int, int], float] = {vertex: float('inf') for vertex in graph.vertices}
        funky_score[origin] = 0
        path_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while active_set:
            _, current_node = heappop(active_set)
            
            if current_node == destination:
                path_found = []
                while current_node in path_map:
                    path_found.insert(0, current_node)
                    current_node = path_map[current_node]
                path_found.insert(0, origin)
                return path_found, funky_score[destination]
            
            for neighbor_node, edge_weight in graph.edges.get(current_node, {}).items():
                tentative_funky_score = funky_score[current_node] + edge_weight
                if tentative_funky_score < funky_score[neighbor_node]:
                    funky_score[neighbor_node] = tentative_funky_score
                    total_score = tentative_funky_score + funky_distance(neighbor_node, destination)
                    heappush(active_set, (total_score, neighbor_node))
                    path_map[neighbor_node] = current_node
        
        raise ValueError("Couldn't find the path")
    except Exception as e:
        logger.error(f"Error in zippy_search algorithm: {e}")
        raise