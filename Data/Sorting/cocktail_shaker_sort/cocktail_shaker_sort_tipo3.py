import logging

from typing import List, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortItem:
    key: Any
    value: Any

    
    except Exception as e:
        logging.error(f"An error occurred during Cocktail Shaker Sort: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    sample_list = [5, 3, 8, 4, 2]
    sorted_list = cocktail_shaker_sort(sample_list)
    print("Sorted List:", sorted_list)


def cocktail_shaker_sort(arr: List[Any]) -> List[Any]:
    try:
        if not isinstance(arr, list):
            logging.error("Input must be a list.")
            raise TypeError("arr must be a list.")
        
        n = len(arr)
        logging.debug(f"Starting Cocktail Shaker Sort with n={n}")
        
        inicio = 0
        fin = n - 1
        swapped = True
        
        while swapped:
            swapped = False
            
            # Forward pass
            for i in range(inicio, fin):
                if arr[i] > arr[i + 1]:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {arr[i]} <-> {arr[i + 1]}")
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            
            if not swapped:
                logging.debug("No swaps made in the forward pass, list might be sorted already.")
                break
            
            fin -= 1
            swapped = False
            
            # Backward pass
            for i in range(fin, inicio, -1):
                if arr[i] < arr[i - 1]:
                    logging.debug(f"Swapping indices {i - 1} and {i}: {arr[i - 1]} <-> {arr[i]}")
                    arr[i], arr[i - 1] = arr[i - 1], arr[i]
                    swapped = True
            
            if not swapped:
                logging.debug("No swaps made in the backward pass, list might be sorted already.")
            
            inicio += 1
        
        logging.info("Cocktail Shaker Sort completed.")
        return arr
