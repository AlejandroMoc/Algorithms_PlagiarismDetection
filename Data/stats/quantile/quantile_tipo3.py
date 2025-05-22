import logging
from typing import List
import math

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def cuantil(data: List[float], q: float, interpolar: bool = False) -> float:
    """
    Calculate the q-th quantile of a list of numbers. Optionally, interpolate the result if the index isn't an integer.

    Args:
        data (List[float]): A list of numbers.
        q (float): The quantile to calculate (between 0 and 1).
        interpolar (bool): Whether to interpolate between values if necessary (default is False).

    Returns:
        float: The q-th quantile of the numbers.

    Example:
        >>> cuantil([1, 2, 3, 4, 5], 0.5)
        3
    """
    if not data:
        logging.error("Input data list is empty.")
        raise ValueError("List is empty")
    
    if not 0 <= q <= 1:
        logging.error(f"Invalid quantile value: {q}. Must be between 0 and 1.")
        raise ValueError("Quantile must be between 0 and 1")
    
    if not all(isinstance(x, (int, float)) for x in data):
        logging.error("Non-numeric value found in data list.")
        raise ValueError("All elements in the data list must be numeric")
    
    datos_ordenados = sorted(data)
    n = len(datos_ordenados)
    indice = q * (n - 1)
    
    if interpolar:
        # Linear interpolation if the index is not an integer
        inferior = int(math.floor(indice))
        superior = int(math.ceil(indice))
        if inferior == superior:
            valor_cuantil = datos_ordenados[inferior]
        else:
            valor_cuantil = datos_ordenados[inferior] + (indice - inferior) * (datos_ordenados[superior] - datos_ordenados[inferior])
    else:
        # Regular method, just use the integer index
        valor_cuantil = datos_ordenados[int(math.floor(indice))]
    
    logging.debug(f"Calculated quantile (q={q}): {valor_cuantil}")
    return valor_cuantil

def calcular_multiples_cuantiles(data: List[float], cuantiles: List[float], interpolar: bool = False) -> List[float]:
    """
    Calculate multiple quantiles at once.

    Args:
        data (List[float]): A list of numbers.
        cuantiles (List[float]): A list of quantiles (values between 0 and 1).
        interpolar (bool): Whether to interpolate between values if necessary (default is False).

    Returns:
        List[float]: A list of quantiles.
    """
    return [cuantil(data, q, interpolar) for q in cuantiles]