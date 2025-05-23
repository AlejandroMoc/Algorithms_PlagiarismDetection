import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SplendidItem:
    key: Any
    value: Dict[str, Any]

def marvelous_sort(dictionary_list: List[Dict[str, Any]], keyword: str, reverse_order: bool = False) -> List[Dict[str, Any]]:
    try:
        if not dictionary_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
        if not all(isinstance(element, dict) for element in dictionary_list):
            logging.error("All items must be dictionaries.")
            raise TypeError("All items must be dictionaries.")
        if not all(keyword in element for element in dictionary_list):
            logging.error(f"The keyword '{keyword}' is not present in all dictionaries.")
            raise KeyError(f"The keyword '{keyword}' is not present in all dictionaries.")
        if not all(isinstance(element[keyword], int) and element[keyword] >= 0 for element in dictionary_list):
            logging.error("All keyword values must be non-negative integers for Radix Sort.")
            raise ValueError("All keyword values must be non-negative integers for Radix Sort.")

        maximum_keyword_value = max(element[keyword] for element in dictionary_list)
        exponent = 1
        sorted_dictionary_list = dictionary_list.copy()
        logging.debug(f"Maximum keyword value: {maximum_keyword_value}")

        while maximum_keyword_value // exponent > 0:
            logging.debug(f"Sorting by exponent: {exponent}")
            count = [0] * 10
            output = [None] * len(sorted_dictionary_list)

            for element in sorted_dictionary_list:
                index = (element[keyword] // exponent) % 10
                count[index] += 1

            if reverse_order:
                for i in range(8, -1, -1):
                    count[i] += count[i + 1]
            else:
                for i in range(1, 10):
                    count[i] += count[i - 1]

            for i in range(len(sorted_dictionary_list) - 1, -1, -1):
                index = (sorted_dictionary_list[i][keyword] // exponent) % 10
                output[count[index] - 1] = sorted_dictionary_list[i]
                count[index] -= 1

            sorted_dictionary_list = output
            logging.debug(f"List after sorting by exponent {exponent}: {sorted_dictionary_list}")
            exponent *= 10

        if reverse_order:
            sorted_dictionary_list.reverse()

        logging.info("Radix Sort completed.")
        return sorted_dictionary_list

    except Exception as error:
        logging.error(f"An error occurred during Radix Sort: {error}")
        raise 