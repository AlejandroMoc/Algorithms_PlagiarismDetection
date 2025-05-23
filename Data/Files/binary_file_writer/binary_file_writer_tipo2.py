import pickle
from typing import Any

def save_bin_file(path_to_file: str, info: Any) -> None:
    """
    Serializes and saves information to a binary file using pickle.

    Args:
        path_to_file (str): The path to the binary file.
        info (Any): The data to serialize and save.

    Returns:
        None
    """
    if not isinstance(path_to_file, str):
        raise ValueError("File path must be a string.")
    try:
        with open(path_to_file, 'wb') as bin_file:
            pickle.dump(info, bin_file)
        print(f"Data successfully saved to '{path_to_file}'")
    except pickle.PickleError as e:
        print(f"Error: Failed to serialize data. Details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {e}")

if __name__ == "__main__":
    # Example Usage
    path_to_file = r"pysnippets\Files\sample_data.pkl"
    data_to_store = {
        'name': 'Aryan Sharma',
        'age': 24,
        'is_student': True,
        'scores': [55, 88, 26],
        'details': {
            'course': 'Computer Science',
            'year': 2025
        }
    }
    
    save_bin_file(path_to_file, data_to_store)
    print("Save operation complete.")