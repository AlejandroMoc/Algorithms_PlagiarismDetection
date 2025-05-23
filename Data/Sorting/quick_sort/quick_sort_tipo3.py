import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import random

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortedItem:
    key: Any
    value: Dict[str, Any]

def custom_sort(data_list: List[Dict[str, Any]], sorting_key: str) -> List[Dict[str, Any]]:
    """
    Sorts a list of dictionaries using a custom sorting algorithm based on a specified key.
    """
    try:
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("data_list must be a list of dictionaries.")
        
        if not all(sorting_key in item for item in data_list):
            logging.error(f"The key '{sorting_key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{sorting_key}' is not present in all dictionaries.")
        
        logging.debug(f"Starting Custom Sort with n={len(data_list)}, key='{sorting_key}'")
        
        if len(data_list) <= 1:
            return data_list
        
        pivot = random.choice(data_list)[sorting_key]
        logging.debug(f"Pivot chosen: {pivot}")
        
        less = [x for x in data_list if x[sorting_key] < pivot]
        equal = [x for x in data_list if x[sorting_key] == pivot]
        greater = [x for x in data_list if x[sorting_key] > pivot]
        
        logging.debug(f"Less partition: {less}")
        logging.debug(f"Equal partition: {equal}")
        logging.debug(f"Greater partition: {greater}")
        
        return custom_sort(less, sorting_key) + equal + custom_sort(greater, sorting_key)
    
    except Exception as e:
        logging.error(f"An error occurred during Custom Sort: {e}")
        raise