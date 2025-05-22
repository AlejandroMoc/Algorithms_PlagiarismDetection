from data_classes import Graph
from logger_config import logger
from typing import List, Dict

def generate_graph(nodes: List[str], connections: Dict[str, Dict[str, int]]) -> Graph:
    try:
        graph = Graph(vertices=nodes, edges=connections)
        logger.debug("Graph created successfully")
        return graph
    except Exception as ex:
        logger.error(f"Error creating graph: {ex}")
        raise 