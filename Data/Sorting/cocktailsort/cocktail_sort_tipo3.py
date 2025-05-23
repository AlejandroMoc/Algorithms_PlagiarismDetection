import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class SortedItem:
    key: Any
    value: Dict[str, Any]

def modified_bubble_sort(data_list: List[Dict[str, Any]], sorting_key: str, reverse_order: bool = False) -> List[Dict[str, Any]]:
    try:
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("data_list must be a list of dictionaries.")
        
        if not all(sorting_key in item for item in data_list):
            logging.error(f"The key '{sorting_key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{sorting_key}' is not present in all dictionaries.")
        
        n = len(data_list)
        start = 0
        end = n - 1
        logging.debug(f"Starting Modified Bubble Sort with n={n}, sorting_key='{sorting_key}', reverse_order={reverse_order}")
        
        while start < end:
            swapped = False
            new_end = start
            i = start
            while i < end:
                if (data_list[i][sorting_key] > data_list[i + 1][sorting_key]) != reverse_order:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {data_list[i]} <-> {data_list[i + 1]}")
                    data_list[i], data_list[i + 1] = data_list[i + 1], data_list[i]
                    swapped = True
                    new_end = i
                i += 1
            end = new_end
            
            if not swapped:
                logging.debug("No swaps made during forward pass, list might be sorted already.")
                break
            
            swapped = False
            new_start = end
            i = end - 1
            while i >= start:
                if (data_list[i][sorting_key] > data_list[i + 1][sorting_key]) != reverse_order:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {data_list[i]} <-> {data_list[i + 1]}")
                    data_list[i], data_list[i + 1] = data_list[i + 1], data_list[i]
                    swapped = True
                    new_start = i
                i -= 1
            start = new_start
            
            if not swapped:
                logging.debug("No swaps made during backward pass, list might be sorted already.")
                break

        logging.info("Modified Bubble Sort completed.")
        return data_list

    except Exception as e:
        logging.error(f"An error occurred during Modified Bubble Sort: {e}")
        raise