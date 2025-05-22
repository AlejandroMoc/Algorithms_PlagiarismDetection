python
import pickle
from typing import Any, Optional

def read_binary_data(file_path: str) -> Optional[Any]:
    if not isinstance(file_path, str):
        raise ValueError("File path must be a string.")
    
    data_objects = []
    try:
        with open(file_path, 'rb') as file:
            while True:
                try:
                    obj = pickle.load(file)
                    data_objects.append(obj)
                except EOFError:
                    break
        return data_objects
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
    except pickle.PickleError as e:
        print(f"Error: Failed to deserialize data. Details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {e}")
        return None

if __name__ == "__main__":
    file_path = r"pysnippets\Files\sample_data.pkl"
    result_data = read_binary_data(file_path)
    if result_data is not None:
        print("Data read successfully:")
        print(result_data)
    else:
        print("Failed to read data from the file.")