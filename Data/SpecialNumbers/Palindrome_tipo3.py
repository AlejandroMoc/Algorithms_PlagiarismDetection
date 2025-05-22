import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def es_palindromo(numero: int) -> bool:
    if numero < 0:
        logging.error("El número debe ser un entero no negativo.")
        raise ValueError("El número debe ser un entero no negativo.")

    str_numero = str(numero)
    es_palin = str_numero == str_numero[::-1]
    logging.debug(f"Número original: {str_numero}, Invertido: {str_numero[::-1]}")

    if es_palin:
        logging.info(f"{numero} es un palíndromo.")
        return True
    else:
        logging.info(f"{numero} no es un palíndromo.")
        return False

if __name__ == "__main__":
    try:
        es_palindromo(101)
        es_palindromo(10)
    except ValueError as ve:
        logging.error(f"Error: {ve}")