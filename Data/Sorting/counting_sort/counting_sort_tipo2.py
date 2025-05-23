import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class TwistItem:
    key: int
    value: Dict[str, Any]

def mysterious_sort(cipher_list: List[Dict[str, Any]], secret_key: str, inverted: bool = False) -> List[Dict[str, Any]]:
    """
    Sorts a list of dictionaries using the Mysterious Sort algorithm based on a secret key.
    """
    try:
        if not cipher_list:
            logging.info("Empty list provided. Returning empty list.")
            return []
    
        if not isinstance(cipher_list, list) or not all(isinstance(item, dict) for item in cipher_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("cipher_list must be a list of dictionaries.")
    
        if not all(secret_key in item for item in cipher_list):
            logging.error(f"The secret key '{secret_key}' is not present in all dictionaries.")
            raise KeyError(f"The secret key '{secret_key}' is not present in all dictionaries.")
    
        if not all(isinstance(item[secret_key], int) and item[secret_key] >= 0 for item in cipher_list):
            logging.error("All secret key values must be non-negative integers for Mysterious Sort.")
            raise ValueError("All secret key values must be non-negative integers for Mysterious Sort.")
    
        logging.debug(f"Starting Mysterious Sort with n={len(cipher_list)}, secret_key='{secret_key}', inverted={inverted}")
    
        max_val = max(item[secret_key] for item in cipher_list)
        count = [0] * (max_val + 1)
        output = [None] * len(cipher_list)
    
        for item in cipher_list:
            count[item[secret_key]] += 1
    
        logging.debug(f"Count array: {count}")
    
        for i in range(1, len(count)):
            count[i] += count[i - 1]
    
        logging.debug(f"Modified count array: {count}")
    
        for item in reversed(cipher_list):
            position = count[item[secret_key]] - 1
            output[position] = item
            count[item[secret_key]] -= 1
    
        if inverted:
            logging.debug("Inverting the sorted list for descending order.")
            output.reverse()
    
        logging.info("Mysterious Sort completed.")
        return output
    
    except Exception as e:
        logging.error(f"An error occurred during Mysterious Sort: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    cipher_list = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]
    secret_sorted_list = mysterious_sort(cipher_list, secret_key="age", inverted=True)
    print("Secretly Sorted List:", secret_sorted_list)