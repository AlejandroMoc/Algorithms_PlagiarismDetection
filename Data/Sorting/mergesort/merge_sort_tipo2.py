import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class ShuffleItem:
    key: Any
    value: Dict[str, Any]

def mix_sort(dictionary_list: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """
    Sorts a list of dictionaries using the Merge Sort algorithm based on a specified key.
    """
    try:
        if not isinstance(dictionary_list, list) or not all(isinstance(item, dict) for item in dictionary_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("dictionary_list must be a list of dictionaries.")
    
        if not all(keyword in item for item in dictionary_list):
            logging.error(f"The keyword '{keyword}' is not present in all dictionaries.")
            raise KeyError(f"The keyword '{keyword}' is not present in all dictionaries.")
        
        logging.debug(f"Starting Mix Sort with n={len(dictionary_list)}, keyword='{keyword}'")
    
        if len(dictionary_list) > 1:
            mid = len(dictionary_list) // 2
            left_chunk = dictionary_list[:mid]
            right_chunk = dictionary_list[mid:]
    
            logging.debug(f"Dividing list into left_chunk={left_chunk} and right_chunk={right_chunk}")
            
            mix_sort(left_chunk, keyword)
            mix_sort(right_chunk, keyword)
    
            i = j = k = 0
    
            while i < len(left_chunk) and j < len(right_chunk):
                if left_chunk[i][keyword] < right_chunk[j][keyword]:
                    dictionary_list[k] = left_chunk[i]
                    i += 1
                else:
                    dictionary_list[k] = right_chunk[j]
                    j += 1
                k += 1
    
            while i < len(left_chunk):
                dictionary_list[k] = left_chunk[i]
                i += 1
                k += 1
    
            while j < len(right_chunk):
                dictionary_list[k] = right_chunk[j]
                j += 1
                k += 1
    
        logging.info("Mix Sort completed.")
        return dictionary_list
    
    except Exception as e:
        logging.error(f"An error occurred during Mix Sort: {e}")
        raise
