python
from typing import Tuple, List, Dict
from heapq import heappop, heappush
import math
from data_classes import Graph
from logger_config import logger

def distance(node: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """Euclidean distance heuristic."""
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def a_star_search(graph: Graph, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
    try:
        open_nodes = []
        heappush(open_nodes, (0, start))
        g_values: Dict[Tuple[int, int], float] = {vertex: float('inf') for vertex in graph.vertices}
        g_values[start] = 0
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while open_nodes:
            _, current_node = heappop(open_nodes)
            
            if current_node == end:
                path = []
                while current_node in came_from:
                    path.insert(0, current_node)
                    current_node = came_from[current_node]
                path.insert(0, start)
                return path, g_values[end]
            
            for neighbor_node, weight_value in graph.edges.get(current_node, {}).items():
                tentative_g_value = g_values[current_node] + weight_value
                if tentative_g_value < g_values[neighbor_node]:
                    g_values[neighbor_node] = tentative_g_value
                    f_value = tentative_g_value + distance(neighbor_node, end)
                    heappush(open_nodes, (f_value, neighbor_node))
                    came_from[neighbor_node] = current_node
        
        raise ValueError("Path not found")
    except Exception as error:
        logger.error(f"Error in A* algorithm: {error}")
        raise