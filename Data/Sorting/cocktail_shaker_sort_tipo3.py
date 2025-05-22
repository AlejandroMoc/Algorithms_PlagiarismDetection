import logging

from typing import List, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class SortItem:
    key: Any
    value: Any

def cocktail_shaker_sort_modified(input_list: List[Any]) -> List[Any]:
    try:
        if not isinstance(input_list, list):
            logging.error("Input must be a list.")
            raise TypeError("input_list must be a list.")
        
        n = len(input_list)
        logging.debug(f"Starting Cocktail Shaker Sort with n={n}")
        
        start_index = 0
        end_index = n - 1
        is_swapped = True
        
        while is_swapped:
            is_swapped = False
            
            # Forward pass
            for i in range(start_index, end_index):
                if input_list[i] > input_list[i + 1]:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {input_list[i]} <-> {input_list[i + 1]}")
                    input_list[i], input_list[i + 1] = input_list[i + 1], input_list[i]
                    is_swapped = True
            
            if not is_swapped:
                logging.debug("No swaps made in the forward pass, list might be sorted already.")
                break
            
            end_index -= 1
            is_swapped = False
            
            # Backward pass
            for i in range(end_index, start_index, -1):
                if input_list[i] < input_list[i - 1]:
                    logging.debug(f"Swapping indices {i - 1} and {i}: {input_list[i - 1]} <-> {input_list[i]}")
                    input_list[i], input_list[i - 1] = input_list[i - 1], input_list[i]
                    is_swapped = True
            
            if not is_swapped:
                logging.debug("No swaps made in the backward pass, list might be sorted already.")
            
            start_index += 1
        
        logging.info("Cocktail Shaker Sort completed.")
        return input_list
    
    except Exception as e:
        logging.error(f"An error occurred during Cocktail Shaker Sort: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    sample_input = [5, 3, 8, 4, 2]
    sorted_output = cocktail_shaker_sort_modified(sample_input)
    print("Sorted List:", sorted_output)