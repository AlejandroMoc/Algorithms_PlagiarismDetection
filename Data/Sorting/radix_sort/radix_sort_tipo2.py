import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class Gemstone:
    identifier: int
    gem_data: Dict[str, Any]

def gravity_sort(crystal_data: List[Dict[str, Any]], identifier: str, reverse_order: bool = False) -> List[Dict[str, Any]]:
    try:
        if not crystal_data:
            logging.info("Empty data stream detected. Returning empty data stream.")
            return []
        if not all(isinstance(crystal, dict) for crystal in crystal_data):
            logging.error("All elements must be dictionaries.")
            raise TypeError("All elements must be dictionaries.")
        if not all(identifier in crystal for crystal in crystal_data):
            logging.error(f"The identifier '{identifier}' is missing in some dictionaries.")
            raise KeyError(f"The identifier '{identifier}' is missing in some dictionaries.")
        if not all(isinstance(crystal[identifier], int) and crystal[identifier] >= 0 for crystal in crystal_data):
            logging.error("All identifier values must be non-negative integers for Gravity Sort.")
            raise ValueError("All identifier values must be non-negative integers for Gravity Sort.")

        max_identifier = max(crystal[identifier] for crystal in crystal_data)
        exp = 1
        sorted_data = crystal_data.copy()
        logging.debug(f"Maximum identifier value: {max_identifier}")

        while max_identifier // exp > 0:
            logging.debug(f"Sorting by exponent: {exp}")
            count = [0] * 10
            output = [None] * len(sorted_data)

            for crystal in sorted_data:
                index = (crystal[identifier] // exp) % 10
                count[index] += 1

            if reverse_order:
                # Modify count array for reverse order
                for i in range(8, -1, -1):
                    count[i] += count[i + 1]
            else:
                for i in range(1, 10):
                    count[i] += count[i - 1]

            for i in range(len(sorted_data) - 1, -1, -1):
                index = (sorted_data[i][identifier] // exp) % 10
                output[count[index] - 1] = sorted_data[i]
                count[index] -= 1

            sorted_data = output
            logging.debug(f"Data after sorting by exponent {exp}: {sorted_data}")
            exp *= 10

        if reverse_order:
            sorted_data.reverse()

        logging.info("Gravity Sort process completed.")
        return sorted_data

    except Exception as e:
        logging.error(f"An error was encountered during Gravity Sort: {e}")
        raise 