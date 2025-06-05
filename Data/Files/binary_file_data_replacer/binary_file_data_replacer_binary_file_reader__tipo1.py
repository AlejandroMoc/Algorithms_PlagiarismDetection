import pickle
from typing import Any, Optional

def read_wacky_file(path_to_file: str) -> Optional[Any]:
    """
    Reads and deserializes all data from a binary file using pickle.

    Args:
        path_to_file (str): The path to the binary file.

    Returns:
        Optional[Any]: A list of deserialized objects from the file or None if an error occurs.
    """
    if not isinstance(path_to_file, str):
        raise ValueError("File path must be a string.")
    
    wacky_list = []
    try:
        with open(path_to_file, 'rb') as funky_file:
            while True:
                try:
                    # Load each object one by one
                    item = pickle.load(funky_file)
                    wacky_list.append(item)
                except EOFError:
                    # End of file reached
                    break
        return wacky_list
    except FileNotFoundError:
        print(f"Error: File '{path_to_file}' not found. Please check the file path.")
    except pickle.PickleError as e:
        print(f"Error: Failed to deserialize data. Details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {e}")
        return None

if __name__ == "__main__":
    # Example Usage
    path_to_file = r"pysnippets\Files\sample_data.pkl"
    result = read_wacky_file(path_to_file)
    if result is not None:
        print("Data read successfully:")
        print(result)
    else:
        print("Failed to read data from the file.")