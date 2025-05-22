import pickle
from typing import Any

def data_transformer_magic(file_path_name: str, magic_search_data: Any, magic_replace_data: Any) -> bool:
    """
    Finds specific data in a binary file and replaces it with new data using pickle.

    Args:
        file_path_name (str): The path to the binary file.
        magic_search_data (Any): The data to search for within the binary file.
        magic_replace_data (Any): The new data to replace the matching data.

    Returns:
        bool: True if replacement is successful, False otherwise.
    """
    if not isinstance(file_path_name, str):
        raise ValueError("File path name must be a string.")
    
    try:
        # Read all data from the binary file
        magic_objects = []
        magic_found = False
        with open(file_path_name, 'rb') as magic_file:
            while True:
                try:
                    magic_obj = pickle.load(magic_file)
                    if magic_obj == magic_search_data:
                        magic_objects.append(magic_replace_data)  # Replace matching object
                        magic_found = True
                    else:
                        magic_objects.append(magic_obj)
                except EOFError:
                    break

        if not magic_found:
            print("No matching data found to replace.")
            return False

        # Write the updated data back to the binary file
        with open(file_path_name, 'wb') as magic_file:
            for magic_obj in magic_objects:
                pickle.dump(magic_obj, magic_file)
        
        print("Data successfully replaced.")
        return True

    except FileNotFoundError:
        print(f"Error: File '{file_path_name}' not found. Please check the file path.")
    except pickle.PickleError as magic_error:
        print(f"Error: Failed to process data. Details: {magic_error}")
    except Exception as magic_error:
        print(f"Error: An unexpected error occurred. Details: {magic_error}")
    return False

if __name__ == "__main__":
    # Example Usage
    magic_file_path = r"pysnippets\Files\sample_data.pkl"
    magic_search_data = {
        'name': 'Aryan Sharma',
        'age': 24,
        'is_student': True,
        'scores': [55, 88, 26],
        'details': {
            'course': 'Computer Science',
            'year': 2025
        }
    }
    magic_replace_data = {
        'name': 'Aaryan Sharma',
        'age': 24,
        'is_student': True,
        'scores': [55, 88, 32],
        'details': {
            'course': 'AWS Solution Architect',
            'year': 2025
        }
    }
    
    magic_result = data_transformer_magic(magic_file_path, magic_search_data, magic_replace_data)
    if magic_result:
        print("Find and replace operation completed successfully.")
    else:
        print("Find and replace operation failed.")