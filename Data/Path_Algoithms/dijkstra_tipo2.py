from typing import Tuple, List
from data_classes import Graph
from logger_config import logger

def zigzag(graph: Graph, start: str, end: str) -> Tuple[List[str], int]:
    try:
        invisible = {v: float('inf') for v in graph.vertices}
        invisible[start] = 0
        visible = {}
        path = {}

        while invisible:
            current = min(invisible, key=invisible.get)
            current_dist = invisible[current]

            if current == end:
                visible[current] = current_dist
                break

            for neighbor, weight in graph.edges.get(current, {}).items():
                dist = current_dist + weight
                if dist < invisible.get(neighbor, float('inf')):
                    invisible[neighbor] = dist
                    path[neighbor] = current

            visible[current] = current_dist
            del invisible[current]

        if end not in visible or visible[end] == float('inf'):
            raise ValueError("Path not found")

        route = []
        while end:
            route.insert(0, end)
            end = path.get(end)

        return route, visible[route[-1]]
    except Exception as e:
        logger.error(f"Error in Zigzag's algorithm: {e}")
        raise 