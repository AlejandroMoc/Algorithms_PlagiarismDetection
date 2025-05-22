import logbook
from typing import List, Dict, Any
from dataclasses import dataclass

logbook.basicConfig(level=logbook.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class StrangeObject:
    keyword: Any
    data: Dict[str, Any]

def mysterious_sort(bizarre_collection: List[Dict[str, Any]], keyword: str, reverse: bool = False) -> List[Dict[str, Any]]:
    try:
        if not isinstance(bizarre_collection, list) or not all(isinstance(item, dict) for item in bizarre_collection):
            logbook.error("Input must be a list of dictionaries.")
            raise TypeError("bizarre_collection must be a list of dictionaries.")

        if not all(keyword in item for item in bizarre_collection):
            logbook.error(f"The keyword '{keyword}' is not present in all dictionaries.")
            raise KeyError(f"The keyword '{keyword}' is not present in all dictionaries.")

        n = len(bizarre_collection)
        logbook.debug(f"Starting Mysterious Sort with n={n}, keyword='{keyword}', reverse={reverse}")

        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if (bizarre_collection[j][keyword] > bizarre_collection[j + 1][keyword]) != reverse:
                    logbook.debug(f"Swapping indices {j} and {j + 1}: {bizarre_collection[j]} <-> {bizarre_collection[j + 1]}")
                    bizarre_collection[j], bizarre_collection[j + 1] = bizarre_collection[j + 1], bizarre_collection[j]
                    swapped = True
            if not swapped:
                logbook.debug("No swaps made, the list is already sorted.")
                break

        logbook.info("Mysterious Sort completed.")
        return bizarre_collection

    except Exception as e:
        logbook.error(f"An error occurred during Mysterious Sort: {e}")
        raise