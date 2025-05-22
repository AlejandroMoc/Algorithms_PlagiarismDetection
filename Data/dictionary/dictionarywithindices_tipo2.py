def transform_list_to_dictionary_with_indexes(arr):
    """
    Transforms a list into a dictionary where the elements of the list are keys and their indexes are values.
    Example:
        Input: ['x', 'y', 'z']
        Output: {'x': 0, 'y': 1, 'z': 2}
    """
    return {element: position for position, element in enumerate(arr)}