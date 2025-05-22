import logging

from typing import List, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class ShufflerItem:
    key: Any
    value: Any

def funky_mixer_sort(lst: List[Any]) -> List[Any]:
    try:
        if not isinstance(lst, list):
            logging.error("Input must be a list.")
            raise TypeError("lst must be a list.")
        
        n = len(lst)
        logging.debug(f"Starting Funky Mixer Sort with n={n}")
        
        start_idx = 0
        end_idx = n - 1
        swapped = True
        
        while swapped:
            swapped = False
            
            # Forward pass
            for i in range(start_idx, end_idx):
                if lst[i] > lst[i + 1]:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {lst[i]} <-> {lst[i + 1]}")
                    lst[i], lst[i + 1] = lst[i + 1], lst[i]
                    swapped = True
            
            if not swapped:
                logging.debug("No swaps made in the forward pass, list might be sorted already.")
                break
            
            end_idx -= 1
            swapped = False
            
            # Backward pass
            for i in range(end_idx, start_idx, -1):
                if lst[i] < lst[i - 1]:
                    logging.debug(f"Swapping indices {i - 1} and {i}: {lst[i - 1]} <-> {lst[i]}")
                    lst[i], lst[i - 1] = lst[i - 1], lst[i]
                    swapped = True
            
            if not swapped:
                logging.debug("No swaps made in the backward pass, list might be sorted already.")
            
            start_idx += 1
        
        logging.info("Funky Mixer Sort completed.")
        return lst
    
    except Exception as e:
        logging.error(f"An error occurred during Funky Mixer Sort: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    sample_lst = [5, 3, 8, 4, 2]
    sorted_lst = funky_mixer_sort(sample_lst)
    print("Sorted List:", sorted_lst)