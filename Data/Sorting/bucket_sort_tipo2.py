import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortItem:
    token: Any
    data: Dict[str, Any]

def funky_sort(dict_list: List[Dict[str, Any]], token: str, reverse: bool = False, funky_size: int = 5) -> List[Dict[str, Any]]:
    try:
        if not dict_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
        if not all(isinstance(item, dict) for item in dict_list):
            logging.error("All items must be dictionaries.")
            raise TypeError("All items must be dictionaries.")
        if not all(token in item for item in dict_list):
            logging.error(f"The token '{token}' is not present in all dictionaries.")
            raise KeyError(f"The token '{token}' is not present in all dictionaries.")
        if not all(isinstance(item[token], (int, float)) for item in dict_list):
            logging.error("All token values must be integers or floats for Funky Sort.")
            raise ValueError("All token values must be integers or floats for Funky Sort.")

        min_token = min(item[token] for item in dict_list)
        max_token = max(item[token] for item in dict_list)
        funky_count = int((max_token - min_token) // funky_size + 1)
        funkies = [[] for _ in range(funky_count)]

        for item in dict_list:
            index = (item[token] - min_token) // funky_size
            funkies[int(index)].append(item)

        sorted_list = []
        for funky in funkies:
            if funky:
                funky_sorted = sorted(funky, key=lambda x: x[token], reverse=reverse)
                sorted_list.extend(funky_sorted)

        if reverse:
            sorted_list.reverse()

        logging.info("Funky Sort completed.")
        return sorted_list

    except Exception as e:
        logging.error(f"An error occurred during Funky Sort: {e}")
        raise 