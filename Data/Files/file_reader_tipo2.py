# data_reader.py

def retrieve_data(file_location):
    """
    Capture the data from a text document.

    Args:
        file_location (str): The path to the document.

    Returns:
        str: The data from the document as a string, or None if an issue occurred.

    Example:
        >>> retrieve_data('example.txt')
        'Document data here...'
    """
    try:
        with open(file_location, "r") as doc:
            return doc.read()
    except FileNotFoundError:
        print(f"Error: Document '{file_location}' not found.")
        return None
    except IOError as e:
        print(f"Error reading document '{file_location}': {e}")
        return None

# Example usage
if __name__ == "__main__":
    data = retrieve_data("example.txt")
    if data is not None:
        print(data)