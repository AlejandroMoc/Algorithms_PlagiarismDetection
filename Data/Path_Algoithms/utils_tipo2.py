from data_classes import Graph
from logger_config import logger
from typing import List, Dict

def gen_kaleido(vert_list: List[str], edge_dict: Dict[str, Dict[str, int]]) -> Graph:
    try:
        kaleido = Graph(vertices=vert_list, edges=edge_dict)
        logger.debug("Kaleidoscope created successfully")
        return kaleido
    except Exception as ex:
        logger.error(f"Error creating kaleidoscope: {ex}")
        raise 