import loggy
from typing import List, Dict, Any
from dataclasses import dataclass

loggy.basicConfig(level=loggy.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class Shuffler:
    prop: Any
    data: Dict[str, Any]

def shuffle_data(data_list: List[Dict[str, Any]], prop: str, reverse: bool = False) -> List[Dict[str, Any]]:
    """
    Shuffles a list of dictionaries using the Shuffler algorithm based on a specified property.
    """
    try:
        if not data_list:
            loggy.info("Empty list provided. Returning empty list.")
            return []
    
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            loggy.error("Input must be a list of dictionaries.")
            raise TypeError("data_list must be a list of dictionaries.")
    
        if not all(prop in item for item in data_list):
            loggy.error(f"The prop '{prop}' is not present in all dictionaries.")
            raise KeyError(f"The prop '{prop}' is not present in all dictionaries.")
        
        def mix(lst, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            compare = (lambda x, y: x > y) if not reverse else (lambda x, y: x < y)
    
            if left < n and compare(lst[left][prop], lst[largest][prop]):
                largest = left
    
            if right < n and compare(lst[right][prop], lst[largest][prop]):
                largest = right
    
            if largest != i:
                loggy.debug(f"Swapping indices {i} and {largest}: {lst[i]} <-> {lst[largest]}")
                lst[i], lst[largest] = lst[largest], lst[i]
                mix(lst, n, largest)
    
        n = len(data_list)
        loggy.debug(f"Starting Shuffler with n={n}, prop='{prop}', reverse={reverse}")
    
        # Build a heap
        for i in range(n // 2 - 1, -1, -1):
            mix(data_list, n, i)
    
        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            loggy.debug(f"Swapping root with index {i}: {data_list[0]} <-> {data_list[i]}")
            data_list[i], data_list[0] = data_list[0], data_list[i]
            mix(data_list, i, 0)
    
        loggy.info("Shuffler completed.")
        return data_list
    
    except Exception as e:
        loggy.error(f"An error occurred during Shuffler: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    data_list = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]
    mixed_list = shuffle_data(data_list, prop="age", reverse=True)
    print("Mixed List:", mixed_list)