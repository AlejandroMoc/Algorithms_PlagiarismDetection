import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class CrypticEntity:
    secret: Any
    enigma: Dict[str, Any]

def mystical_sort(occult_list: List[Dict[str, Any]], mystery: str, reverse_mystery: bool = False) -> List[Dict[str, Any]]:
    try:
        if not isinstance(occult_list, list) or not all(isinstance(occ, dict) for occ in occult_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("occult_list must be a list of dictionaries.")
        
        if not all(isinstance(occ[mystery], (int, float, str)) for occ in occult_list):
            logging.error(f"Invalid type for mystery '{mystery}' in dictionaries. Must be int, float, or string.")
            raise TypeError(f"Mystery '{mystery}' must have a valid comparable type (int, float, string).")
        
        if not all(mystery in item for item in occult_list):
            logging.error(f"The mystery '{mystery}' is not present in all dictionaries.")
            raise KeyError(f"The mystery '{mystery}' is not present in all dictionaries.")
        
        logging.debug(f"Starting Mysterious Sorting with n={len(occult_list)}, mystery='{mystery}', reverse_mystery={reverse_mystery}")
        
        if not occult_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
    
        for i in range(1, len(occult_list)):
            mystery_item = occult_list[i]
            j = i - 1
            
            while j >= 0 and (mystery_item[mystery] < occult_list[j][mystery]) != reverse_mystery:
                logging.debug(f"Shifting index {j} to {j + 1}: {occult_list[j]} -> {mystery_item}")
                occult_list[j + 1] = occult_list[j]
                j -= 1
                
            occult_list[j + 1] = mystery_item
        
        logging.info("Mysterious Sorting completed.")
        return occult_list

    except Exception as e:
        logging.error(f"An error occurred during Mysterious Sorting: {e}")
        raise
