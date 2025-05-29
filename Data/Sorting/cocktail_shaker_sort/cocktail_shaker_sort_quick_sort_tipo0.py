import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import random

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class UniqueData:
    idn: Any
    info: Dict[str, Any]

def super_sort(data_list: List[Dict[str, Any]], identifier: str) -> List[Dict[str, Any]]:
    """
    Sorts a list of dictionaries using the Super Sort algorithm based on a specified identifier.
    """
    try:
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("data_list must be a list of dictionaries.")
        
        if not all(identifier in item for item in data_list):
            logging.error(f"The identifier '{identifier}' is not present in all dictionaries.")
            raise KeyError(f"The identifier '{identifier}' is not present in all dictionaries.")
        
        logging.debug(f"Starting Super Sort with n={len(data_list)}, identifier='{identifier}'")
        
        if len(data_list) <= 1:
            return data_list
        
        pivot = random.choice(data_list)[identifier]
        logging.debug(f"Pivot chosen: {pivot}")
        
        left = [x for x in data_list if x[identifier] < pivot]
        middle = [x for x in data_list if x[identifier] == pivot]
        right = [x for x in data_list if x[identifier] > pivot]
        
        logging.debug(f"Left partition: {left}")
        logging.debug(f"Middle partition: {middle}")
        logging.debug(f"Right partition: {right}")
        
        return super_sort(left, identifier) + middle + super_sort(right, identifier)
    
    except Exception as e:
        logging.error(f"An error occurred during Super Sort: {e}")
        raise