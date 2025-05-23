import logging
import math
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def salto_busqueda(lista: List[int], objetivo: int) -> int:
    logging.info("Comenzando búsqueda por salto")
    if not lista:
        logging.error("Se proporcionó una lista vacía para la búsqueda.")
        return -1
    
    n = len(lista)
    salto = int(math.sqrt(n))  # El tamaño del salto es la raíz cuadrada de la longitud de la lista
    previo = 0

    # Saltar al bloque correcto
    while previo < n and lista[min(salto, n) - 1] < objetivo:
        logging.debug(f"Saltando desde el índice {previo} al índice {salto}")
        previo = salto
        salto += int(math.sqrt(n))
        if previo >= n:
            logging.warning(f"El elemento {objetivo} no está presente en la lista")
            return -1
    
    # Realizar una búsqueda lineal dentro del bloque encontrado
    for i in range(previo, min(salto, n)):
        logging.debug(f"Revisando índice {i}, valor={lista[i]}")
        if lista[i] == objetivo:
            logging.info(f"Objetivo {objetivo} encontrado en el índice {i}")
            return i

    logging.warning(f"El elemento {objetivo} no está presente en la lista")
    return -1

# Código de ejecución
if __name__ == "__main__":
    lista = [2, 3, 4, 10, 40]
    objetivo = 10
    resultado = salto_busqueda(lista, objetivo)
    if resultado == -1:
        logging.info("El elemento no está presente en la lista")
    else:
        logging.info(f"El elemento está presente en el índice {resultado}")