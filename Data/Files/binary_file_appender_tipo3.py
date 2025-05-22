import pickle
from typing import Any

def add_data_to_binary_file(file_path_str: str, data_to_add: Any) -> None:
    if not isinstance(file_path_str, str):
        raise ValueError("File path must be a string.")
    
    try:
        with open(file_path_str, 'ab') as file:
            pickle.dump(data_to_add, file)
        print(f"Data successfully appended to '{file_path_str}'")
    except pickle.PickleError as e:
        print(f"Error: Failed to serialize data. Details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {e}")

if __name__ == "__main__":
    file_path_str = r"pysnippets\Files\sample_data.pkl"
    data_to_append = {
        'name': 'Jaya Mehra',
        'age': 25,
        'is_student': True,
        'scores': [85, 90, 92],
        'details': {
            'course': 'Data Science',
            'year': 2026
        }
    }
    
    add_data_to_binary_file(file_path_str, data_to_append)
    print("Append operation complete.")