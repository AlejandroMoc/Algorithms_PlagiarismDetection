import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortedItem:
    key: Any
    value: Dict[str, Any]

def custom_sort(dict_list: List[Dict[str, Any]], key: str, reverse: bool = False, bucket_size: int = 5) -> List[Dict[str, Any]]:
    try:
        if not dict_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
        if not all(isinstance(element, dict) for element in dict_list):
            logging.error("All items must be dictionaries.")
            raise TypeError("All items must be dictionaries.")
        if not all(key in element for element in dict_list):
            logging.error(f"The key '{key}' is not present in all dictionaries.")
            raise KeyError(f"The key '{key}' is not present in all dictionaries.")
        if not all(isinstance(element[key], (int, float)) for element in dict_list):
            logging.error("All key values must be integers or floats for Custom Sort.")
            raise ValueError("All key values must be integers or floats for Custom Sort.")

        min_key = min(element[key] for element in dict_list)
        max_key = max(element[key] for element in dict_list)
        bucket_count = int((max_key - min_key) // bucket_size + 1)
        buckets = [[] for _ in range(bucket_count)]

        for element in dict_list:
            index = (element[key] - min_key) // bucket_size
            buckets[int(index)].append(element)

        sorted_list = []
        for bucket in buckets:
            if bucket:
                bucket_sorted = sorted(bucket, key=lambda x: x[key], reverse=reverse)
                sorted_list.extend(bucket_sorted)

        if reverse:
            sorted_list.reverse()

        logging.info("Custom Sort completed.")
        return sorted_list

    except Exception as e:
        logging.error(f"An error occurred during Custom Sort: {e}")
        raise 