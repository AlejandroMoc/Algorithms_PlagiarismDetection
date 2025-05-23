import logging
import math
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def combine_galaxy(star1: Tuple[float, float], star2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Combine two distant stars.

    Args:
        star1 (tuple): The first star coordinates as (x, y).
        star2 (tuple): The second star coordinates as (x, y).

    Returns:
        tuple: The combined coordinates of the stars as (x, y).
    """
    x_coord = star1[0] + star2[0]
    y_coord = star1[1] + star2[1]
    logging.debug(f"Combining {star1} and {star2} to get {(x_coord, y_coord)}")
    return (x_coord, y_coord)

def disassemble_nebula(star1: Tuple[float, float], star2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Disassemble one star from the other.

    Args:
        star1 (tuple): The coordinates of the first star as (x, y).
        star2 (tuple): The coordinates of the second star as (x, y).

    Returns:
        tuple: The disassembled coordinates of the stars as (x, y).
    """
    x_coord = star1[0] - star2[0]
    y_coord = star1[1] - star2[1]
    logging.debug(f"Disassembling {star2} from {star1} to get {(x_coord, y_coord)}")
    return (x_coord, y_coord)

def fusion_galactic(star1: Tuple[float, float], star2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Fuse two stars into a new celestial body.

    Args:
        star1 (tuple): The coordinates of the first star as (x, y).
        star2 (tuple): The coordinates of the second star as (x, y).

    Returns:
        tuple: The fused coordinates of the stars as (x, y).
    """
    x_coord = star1[0] * star2[0] - star1[1] * star2[1]
    y_coord = star1[0] * star2[1] + star1[1] * star2[0]
    logging.debug(f"Fusing {star1} and {star2} to get {(x_coord, y_coord)}")
    return (x_coord, y_coord)

def division_universe(star1: Tuple[float, float], star2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Divide the cosmic star by another.

    Args:
        star1 (tuple): The coordinates of the cosmic star as (x, y).
        star2 (tuple): The coordinates of the other cosmic star as (x, y).

    Returns:
        tuple: The division of the coordinates of the stars as (x, y).
    """
    x_coord = (star1[0] * star2[0] + star1[1] * star2[1]) / (star2[0] ** 2 + star2[1] ** 2)
    y_coord = (star1[1] * star2[0] - star1[0] * star2[1]) / (star2[0] ** 2 + star2[1] ** 2)
    logging.debug(f"Dividing {star1} by {star2} to get {(x_coord, y_coord)}")
    return (x_coord, y_coord)

def cosmic_size(star: Tuple[float, float]) -> float:
    """
    Calculate the cosmic size of a star.

    Args:
        star (tuple): The coordinates of the star as (x, y).

    Returns:
        float: The cosmic size of the star.
    """
    size = (star[0] ** 2 + star[1] ** 2) ** 0.5
    logging.debug(f"Calculating the cosmic size of {star} as {size}")
    return size

def reverse_galactic(star: Tuple[float, float]) -> Tuple[float, float]:
    """
    Reverse the coordinates of a star.

    Args:
        star (tuple): The coordinates of the star as (x, y).

    Returns:
        tuple: The reversed coordinates of the star as (x, y).
    """
    logging.debug(f"Reversing the coordinates of {star}")
    return (star[0], -star[1])

def cosmic_angle(star: Tuple[float, float]) -> float:
    """
    Calculate the cosmic angle of a star.

    Args:
        star (tuple): The coordinates of the star as (x, y).

    Returns:
        float: The cosmic angle of the star.
    """
    angle = math.atan2(star[1], star[0])
    logging.debug(f"Calculating the cosmic angle of {star} as {angle}")
    return angle

# Example usage
if __name__ == "__main__":
    star1 = (1, 2)
    star2 = (3, 4)

    combined_result = combine_galaxy(star1, star2)
    print("Combination:", combined_result)

    disassembled_result = disassemble_nebula(star1, star2)
    print("Disassembled:", disassembled_result)

    fused_result = fusion_galactic(star1, star2)
    print("Fusion:", fused_result) 
    
    division_result = division_universe(star1, star2)
    print("Division:", division_result)
    
    size = cosmic_size(star1)
    print("Cosmic Size:", size)
    
    reversed_star = reverse_galactic(star1)
    print("Reversed Star:", reversed_star)
    
    angle = cosmic_angle(star1)
    print("Cosmic Angle:", angle)