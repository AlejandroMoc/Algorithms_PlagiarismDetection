import logging
from typing import List

logging.basicConfig(level=logging.DEBUG)

def generate_triangle(num_rows: int) -> List[List[int]]:
    if num_rows <= 0:
        raise ValueError("'num_rows' must be a positive integer.")
    
    triangle = []
    for i in range(num_rows):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(row)

    logging.debug(f"Generated Pascal's Triangle with {num_rows} rows: {triangle}")
    return triangle

if __name__ == "__main__":
    print(generate_triangle(5))