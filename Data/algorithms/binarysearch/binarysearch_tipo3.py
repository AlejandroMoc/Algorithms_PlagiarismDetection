python
from typing import List, Union

def busqueda_binaria(lista: List[Union[int, float]], objetivo: Union[int, float]) -> int:
    low, high = 0, len(lista) - 1

    while low <= high:
        medio = low + (high - low) // 2
        if lista[medio] == objetivo:
            return medio
        elif lista[medio] < objetivo:
            low = medio + 1
        else:
            high = medio - 1

    return -1

if __name__ == "__main__":
    resultado = busqueda_binaria([1, 2, 3, 4, 5], 3)
    print(resultado)  # Output: 2