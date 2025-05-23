import json

def augment_data(d: dict, key, value) -> dict:
    """
    Incorporates a new duo into the data structure.

    Example usage:
    augment_data({'a': 1}, 'b', 2) -> {'a': 1, 'b': 2}
    """
    d[key] = value
    return d

def adjust_data(d: dict, key, value) -> dict:
    """
    Modifies the content of an existing key.

    Example usage:
    adjust_data({'a': 1}, 'a', 2) -> {'a': 2}
    """
    if key in d:
        d[key] = value
    return d

def eliminate_data(d: dict, key) -> dict:
    """
    Eradicates a key-value pair from the data structure.

    Example usage:
    eliminate_data({'a': 1, 'b': 2}, 'a') -> {'b': 2}
    """
    d.pop(key, None)
    return d

def validate_key(d: dict, key) -> bool:
    """
    Validates the existence of a specific key in the data structure.

    Example usage:
    validate_key({'a': 1}, 'a') -> True
    """
    return key in d

def traverse_keys(d: dict) -> list:
    """
    Walks through all the keys in the data structure.

    Example usage:
    traverse_keys({'a': 1, 'b': 2}) -> ['a', 'b']
    """
    return list(d.keys())

def traverse_values(d: dict) -> list:
    """
    Walks through all the values in the data structure.

    Example usage:
    traverse_values({'a': 1, 'b': 2}) -> [1, 2]
    """
    return list(d.values())

def merge_collections(d1: dict, d2: dict) -> dict:
    """
    Combines two data structures into one.

    Example usage:
    merge_collections({'a': 1}, {'b': 2}) -> {'a': 1, 'b': 2}
    """
    d1.update(d2)
    return d1

def duplicate_data(d: dict) -> dict:
    """
    Creates a carbon copy of the data structure.

    Example usage:
    duplicate_data({'a': 1}) -> {'a': 1}
    """
    return d.copy()

def erase_data(d: dict) -> dict:
    """
    Removes all items from the data structure.

    Example usage:
    erase_data({'a': 1}) -> {}
    """
    d.clear()
    return d

def find_key_by_content(d: dict, value) -> list:
    """
    Retrieves keys that match a specific value.

    Example usage:
    find_key_by_content({'a': 1, 'b': 2, 'c': 1}, 1) -> ['a', 'c']
    """
    return [k for k, v in d.items() if v == value]

def flip_data(d: dict) -> dict:
    """
    Inverts keys and values in the data structure.

    Example usage:
    flip_data({'a': 1, 'b': 2}) -> {1: 'a', 2: 'b'}
    """
    return {v: k for k, v in d.items()}

def tally_content(d: dict) -> dict:
    """
    Counts occurrences of each unique value in the data structure.

    Example usage:
    tally_content({'a': 1, 'b': 2, 'c': 1}) -> {1: 2, 2: 1}
    """
    counts = {}
    for value in d.values():
        counts[value] = counts.get(value, 0) + 1
    return counts

def sift_by_content(d: dict, condition) -> dict:
    """
    Filters data entries based on a condition function.

    Example usage:
    sift_by_content({'a': 1, 'b': 2, 'c': 3}, lambda x: x > 1) -> {'b': 2, 'c': 3}
    """
    return {k: v for k, v in d.items() if condition(v)}

def high_low_value_keys(d: dict) -> tuple:
    """
    Identifies keys with the highest and lowest values in the data structure.

    Example usage:
    high_low_value_keys({'a': 1, 'b': 2, 'c': 3}) -> ('a', 'c')
    """
    if not d:
        return None, None
    min_key = min(d, key=d.get)
    max_key = max(d, key=d.get)
    return min_key, max_key

def sort_by_content(d: dict, reverse=False) -> dict:
    """
    Sorts data entries by their values.

    Example usage:
    sort_by_content({'a': 3, 'b': 1, 'c': 2}) -> {'b': 1, 'c': 2, 'a': 3}
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))

def to_json_format(d: dict) -> str:
    """
    Converts the data structure to a JSON string.

    Example usage:
    to_json_format({'a': 1, 'b': 2}) -> '{"a": 1, "b": 2}'
    """
    return json.dumps(d)

def sum_numerical_content(d: dict) -> float:
    """
    Sums all numeric values in the data structure.

    Example usage:
    sum_numerical_content({'a': 1, 'b': 2, 'c': 'x'}) -> 3
    """
    return sum(v for v in d.values() if isinstance(v, (int, float)))

# Example usage of the functions in the script
if __name__ == "__main__":
    data_sample = {'name': 'Alice', 'age': 30, 'salary': 50000, 'department': 'Engineering'}

    print("Original Data Structure:", data_sample)
    print("After Incorporating Position:", augment_data(data_sample.copy(), 'position', 'Engineer'))
    print("After Adjusting Age:", adjust_data(data_sample.copy(), 'age', 31))
    print("After Eradicating Name:", eliminate_data(data_sample.copy(), 'name'))
    print("Is Age Present?:", validate_key(data_sample, 'age'))
    print("Traversing Keys:", traverse_keys(data_sample))
    print("Traversing Values:", traverse_values(data_sample))
    print("Merging with Another Collection:", merge_collections(data_sample.copy(), {'city': 'New York'}))
    print("Duplicate Data Structure:", duplicate_data(data_sample))
    print("Erased Data Structure:", erase_data(data_sample.copy()))

    print("Find Key by Value (50000):", find_key_by_content(data_sample, 50000))
    print("Inverted Data Structure:", flip_data(data_sample))
    print("Count Content:", tally_content(data_sample))
    print("Sift by Content (>10000):", sift_by_content(data_sample, lambda x: isinstance(x, int) and x > 10000))
    print("High and Low Value Keys:", high_low_value_keys({'a': 1, 'b': 2, 'c': 3}))
    print("Sorted by Content:", sort_by_content({'a': 3, 'b': 1, 'c': 2}))
    print("Data Structure as JSON:", to_json_format(data_sample))
    print("Sum of Numerical Content:", sum_numerical_content(data_sample))