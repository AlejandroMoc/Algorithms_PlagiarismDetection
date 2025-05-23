import pickle
from typing import Any, Optional

def search_data_in_binary_archive(archive_path: str, target_data: Any) -> Optional[Any]:
    """
    Scours an archive for a specific data entity using pickle.

    Args:
        archive_path (str): The path to the archive file.
        target_data (Any): The data to look for within the archive.

    Returns:
        Optional[Any]: The first matching object from the archive or None if no match is found.
    """
    if not isinstance(archive_path, str):
        raise ValueError("Archive path must be a string.")
    
    try:
        with open(archive_path, 'rb') as archive:
            while True:
                try:
                    # Examine each object one by one
                    data_entity = pickle.load(archive)
                    if data_entity == target_data:
                        return data_entity
                except EOFError:
                    # End of archive reached
                    break
        print("No matching data found.")
        return None
    except FileNotFoundError:
        print(f"Error: Archive '{archive_path}' not found. Please verify the archive path.")
    except pickle.PickleError as e:
        print(f"Error: Failed to deserialize data entity. Details: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred. Details: {e}")
        return None

if __name__ == "__main__":
    # Example Usage
    archive_path = r"pysnippets\Files\sample_archive.pkl"
    target_data = {'name': 'Aryan Sharma',
        'age': 24,
        'is_student': True,
        'scores': [55, 88, 26],
        'details': {
            'course': 'Computer Science',
            'year': 2025
        }}
    
    result = search_data_in_binary_archive(archive_path, target_data)
    if result is not None:
        print("Matching data found:")
        print(result)
    else:
        print("No matching data found.")