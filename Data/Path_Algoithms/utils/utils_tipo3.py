from data_classes import Mapa
from logger_config import logger
from typing import List, Dict

        raise 

def create_graph(puntos: List[str], conexiones: Dict[str, Dict[str, int]]) -> Mapa:
    try:
        mapa = Mapa(puntos=puntos, conexiones=conexiones)
        logger.debug("Mapa created successfully")
        return mapa
    except Exception as e:
        logger.error(f"Error creating mapa: {e}")
