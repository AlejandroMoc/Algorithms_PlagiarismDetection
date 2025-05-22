import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortItem:
    key: Any
    value: Dict[str, Any]

def sort_radix(items_list: List[Dict[str, Any]], sorting_key: str, descending: bool = False) -> List[Dict[str, Any]]:
    try:
        if not items_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
        if not all(isinstance(item, dict) for item in items_list):
            logging.error("All items must be dictionaries.")
            raise TypeError("All items must be dictionaries.")
        if not all(sorting_key in item for item in items_list):
            logging.error(f"The key '{key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{key}' is not present in all dictionaries.")
        if not all(isinstance(item[sorting_key], int) and item[sorting_key] >= 0 for item in items_list):
            logging.error("All key values must be non-negative integers for Radix Sort.")
            raise ValueError("All key values must be non-negative integers for Radix Sort.")

        max_key_value = max(item[sorting_key] for item in items_list)
        exponent = 1
        sorted_items = items_list.copy()
        logging.debug(f"Maximum key value: {max_key_value}")

        while max_key_value // exponent > 0:
            logging.debug(f"Sorting by exponent: {exponent}")
            count = [0] * 10
            output = [None] * len(sorted_items)

            for item in sorted_items:
                index = (item[sorting_key] // exponent) % 10
                count[index] += 1

            if descending:
                for i in range(8, -1, -1):
                    count[i] += count[i + 1]
            else:
                for i in range(1, 10):
                    count[i] += count[i - 1]

            for i in range(len(sorted_items) - 1, -1, -1):
                index = (sorted_items[i][sorting_key] // exponent) % 10
                output[count[index] - 1] = sorted_items[i]
                count[index] -= 1

            sorted_items = output
            logging.debug(f"List after sorting by exponent {exponent}: {sorted_items}")
            exponent *= 10

        if descending:
            sorted_items.reverse()

        logging.info("Radix Sort completed.")
        return sorted_items

    except Exception as e:
        logging.error(f"An error occurred during Radix Sort: {e}")
        raise