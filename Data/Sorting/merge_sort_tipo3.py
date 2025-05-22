import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class SortItem:
    key: Any
    value: Dict[str, Any]

def sort_dict_list(dict_list: List[Dict[str, Any]], sort_key: str) -> List[Dict[str, Any]]:
    try:
        if not isinstance(dict_list, list) or not all(isinstance(d, dict) for d in dict_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("dict_list must be a list of dictionaries.")
    
        if not all(sort_key in d for d in dict_list):
            logging.error(f"The key '{sort_key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{sort_key}' is not present in all dictionaries.")
        
        logging.debug(f"Starting Merge Sort with n={len(dict_list)}, sort_key='{sort_key}'")
    
        if len(dict_list) > 1:
            mid = len(dict_list) // 2
            left_half = dict_list[:mid]
            right_half = dict_list[mid:]
    
            logging.debug(f"Dividing list into left_half={left_half} and right_half={right_half}")
            
            sort_dict_list(left_half, sort_key)
            sort_dict_list(right_half, sort_key)
    
            i = j = k = 0
    
            while i < len(left_half) and j < len(right_half):
                if left_half[i][sort_key] < right_half[j][sort_key]:
                    dict_list[k] = left_half[i]
                    i += 1
                else:
                    dict_list[k] = right_half[j]
                    j += 1
                k += 1
    
            while i < len(left_half):
                dict_list[k] = left_half[i]
                i += 1
                k += 1
    
            while j < len(right_half):
                dict_list[k] = right_half[j]
                j += 1
                k += 1
    
        logging.info("Merge Sort completed.")
        return dict_list
    
    except Exception as ex:
        logging.error(f"An error occurred during Merge Sort: {ex}")
        raise