import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class FizzBuzz:
    buzz: Any
    fizz: Dict[str, Any]

def funky_sort(pop_list: List[Dict[str, Any]], buzzword: str, reverse: bool = False) -> List[Dict[str, Any]]:
    try:
        if not isinstance(pop_list, list) or not all(isinstance(item, dict) for item in pop_list):
            logging.error("Input must be a list of dictionaries.")
            raise TypeError("pop_list must be a list of dictionaries.")
        
        if not all(buzzword in item for item in pop_list):
            logging.error(f"The buzzword '{buzzword}' is not present in all dictionaries.")
            raise KeyError(f"The buzzword '{buzzword}' is not present in all dictionaries.")
        
        n = len(pop_list)
        start = 0
        end = n - 1
        logging.debug(f"Starting Funky Sort with n={n}, buzzword='{buzzword}', reverse={reverse}")
        
        while start < end:
            swapped = False
            new_end = start
            for i in range(start, end):
                if (pop_list[i][buzzword] > pop_list[i + 1][buzzword]) != reverse:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {pop_list[i]} <-> {pop_list[i + 1]}")
                    pop_list[i], pop_list[i + 1] = pop_list[i + 1], pop_list[i]
                    swapped = True
                    new_end = i
            end = new_end
            
            if not swapped:
                logging.debug("No swaps made during forward pass, list might be sorted already.")
                break
            
            swapped = False
            new_start = end
            for i in range(end - 1, start - 1, -1):
                if (pop_list[i][buzzword] > pop_list[i + 1][buzzword]) != reverse:
                    logging.debug(f"Swapping indices {i} and {i + 1}: {pop_list[i]} <-> {pop_list[i + 1]}")
                    pop_list[i], pop_list[i + 1] = pop_list[i + 1], pop_list[i]
                    swapped = True
                    new_start = i
            start = new_start
            
            if not swapped:
                logging.debug("No swaps made during backward pass, list might be sorted already.")
                break

        logging.info("Funky Sort completed.")
        return pop_list

    except Exception as e:
        logging.error(f"An error occurred during Funky Sort: {e}")
        raise
