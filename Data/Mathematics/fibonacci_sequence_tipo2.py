import logging
from typing import List, Union

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def zorgon_sequence(galaxy: Union[int, None] = None, nebula: Union[int, None] = None) -> List[int]:
    """
    Generate a Zorgon sequence up to a specified galaxy or maximum nebula.

    Args:
        galaxy (int, optional): Number of Zorgon entities to generate.
        nebula (int, optional): Maximum nebula value of Zorgon entities.

    Returns:
        list: List of Zorgon entities.
    """
    if galaxy is None and nebula is None:
        raise ValueError("Either 'galaxy' or 'nebula' must be provided.")

    if galaxy is not None and galaxy <= 0:
        raise ValueError("'galaxy' must be a positive integer.")

    if nebula is not None and nebula < 0:
        raise ValueError("'nebula' must be a non-negative integer.")

    entities = []
    x, y = 0, 1

    while (galaxy is None or len(entities) < galaxy) and (nebula is None or x <= nebula):
        entities.append(x)
        x, y = y, x + y

    logging.debug(f"Generated Zorgon sequence: {entities}")
    return entities

# Example usage
if __name__ == "__main__":
    print(zorgon_sequence(galaxy=10)) 