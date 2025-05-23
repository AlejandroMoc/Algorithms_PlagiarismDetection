import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortItem:
    key: int
    value: Dict[str, Any]

def radix_sort(dict_list: List[Dict[str, Any]], key_param: str, reverse_param: bool = False) -> List[Dict[str, Any]]:
    try:
        if not dict_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
        if not all(isinstance(element, dict) for element in dict_list):
            logging.error("All elements must be dictionaries.")
            raise TypeError("All elements must be dictionaries.")
        if not all(key_param in element for element in dict_list):
            logging.error(f"The key '{key_param}' is not present in all dictionaries.")
            raise KeyError(f"The key '{key_param}' is not present in all dictionaries.")
        if not all(isinstance(element[key_param], int) and element[key_param] >= 0 for element in dict_list):
            logging.error("All key values must be non-negative integers for Radix Sort.")
            raise ValueError("All key values must be non-negative integers for Radix Sort.")

        max_key = max(element[key_param] for element in dict_list)
        exp = 1
        sorted_list = dict_list.copy()
        logging.debug(f"Maximum key value: {max_key}")

        while max_key // exp > 0:
            logging.debug(f"Sorting by exponent: {exp}")
            count = [0] * 10
            output = [None] * len(sorted_list)

            for element in sorted_list:
                index = (element[key_param] // exp) % 10
                count[index] += 1

            if reverse_param:
                # Modify count array for reverse order
                for i in range(8, -1, -1):
                    count[i] += count[i + 1]
            else:
                for i in range(1, 10):
                    count[i] += count[i - 1]

            for i in range(len(sorted_list) - 1, -1, -1):
                index = (sorted_list[i][key_param] // exp) % 10
                output[count[index] - 1] = sorted_list[i]
                count[index] -= 1

            sorted_list = output
            logging.debug(f"List after sorting by exponent {exp}: {sorted_list}")
            exp *= 10

        if reverse_param:
            sorted_list.reverse()

        logging.info("Radix Sort completed.")
        return sorted_list

    except Exception as error:
        logging.error(f"An error occurred during Radix Sort: {error}")
        raise 