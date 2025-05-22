import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def es_fuerte(numero: int) -> bool:
    if numero < 0:
        logging.error("El número debe ser un entero no negativo.")
        raise ValueError("El número debe ser un entero no negativo.")

    num_str = str(numero)
    suma_fuerte = sum(math.factorial(int(digito)) for digito in num_str)
    logging.debug(f"Suma de factoriales para {numero}: {suma_fuerte}")

    if suma_fuerte == numero:
        logging.info(f"{numero} es un número fuerte.")
        return True
    else:
        logging.info(f"{numero} no es un número fuerte.")
        return False

if __name__ == "__main__":
    try:
        es_fuerte(145)
        es_fuerte(134)
    except ValueError as ve:
        logging.error(f"Error: {ve}")