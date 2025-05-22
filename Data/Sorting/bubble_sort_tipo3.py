import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortingItem:
    key: Any
    value: Dict[str, Any]

def bubble_sort(data_list: List[Dict[str, Any]], sorting_key: str, reverse_order: bool = False) -> List[Dict[str, Any]]:
    try:
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("data_list must be a list of dictionaries.")

        if not all(sorting_key in item for item in data_list):
            logging.error(f"The key '{sorting_key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{sorting_key}' is not present in all dictionaries.")

        n = len(data_list)
        logging.debug(f"Starting Bubble Sort with n={n}, key='{sorting_key}', reverse={reverse_order}")

        for i in range(n):
            swapped = False
            j = 0
            while j < n - i - 1:
                if (data_list[j][sorting_key] > data_list[j + 1][sorting_key]) != reverse_order:
                    logging.debug(f"Swapping indices {j} and {j + 1}: {data_list[j]} <-> {data_list[j + 1]}")
                    data_list[j], data_list[j + 1] = data_list[j + 1], data_list[j]
                    swapped = True
                j += 1
            if not swapped:
                logging.debug("No swaps made, the list is already sorted.")
                break

        logging.info("Bubble Sort completed.")
        return data_list

    except Exception as e:
        logging.error(f"An error occurred during Bubble Sort: {e}")
        raise