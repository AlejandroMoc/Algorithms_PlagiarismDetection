import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def verificar_automorfico(numero: int) -> bool:
    if numero < 0:
        logging.error("El número debe ser un entero no negativo.")
        raise ValueError("El número debe ser un entero no negativo.")

    cuadrado = numero ** 2
    es_automorfico = str(cuadrado).endswith(str(numero))
    logging.debug(f"Cuadrado de {numero}: {cuadrado}")

    if es_automorfico:
        logging.info(f"{numero} es un número automórfico.")
        return True
    else:
        logging.info(f"{numero} no es un número automórfico.")
        return False

if __name__ == "__main__":
    try:
        verificar_automorfico(25)
        verificar_automorfico(16)
    except ValueError as ve:
        logging.error(f"Error: {ve}")