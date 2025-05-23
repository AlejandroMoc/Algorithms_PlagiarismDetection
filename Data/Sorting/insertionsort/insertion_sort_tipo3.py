import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class ItemSort:
    key: Any
    value: Dict[str, Any]

def sort_insertion(list_dict: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    try:
        if not isinstance(list_dict, list) or not all(isinstance(element, dict) for element in list_dict):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("list_dict must be a list of dictionaries.")
        
        if not all(isinstance(element[key], (int, float, str)) for element in list_dict):
            logging.error(f"Invalid type for key '{key}' in dictionaries. Must be int, float, or string.")
            raise TypeError(f"Key '{key}' must have a valid comparable type (int, float, string).")
        
        if not all(key in element for element in list_dict):
            logging.error(f"The key '{key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{key}' is not present in all dictionaries.")
        
        logging.debug(f"Starting Insertion Sort with n={len(list_dict)}, key='{key}', reverse={reverse}")
        
        if not list_dict:
            logging.info("Empty list provided. Returning empty list.")
            return []
    
        for i in range(1, len(list_dict)):
            key_element = list_dict[i]
            j = i - 1
            
            while j >= 0 and (key_element[key] < list_dict[j][key]) != reverse:
                logging.debug(f"Shifting index {j} to {j + 1}: {list_dict[j]} -> {key_element}")
                list_dict[j + 1] = list_dict[j]
                j -= 1
                
            list_dict[j + 1] = key_element
        
        logging.info("Insertion Sort completed.")
        return list_dict

    except Exception as e:
        logging.error(f"An error occurred during Insertion Sort: {e}")
        raise