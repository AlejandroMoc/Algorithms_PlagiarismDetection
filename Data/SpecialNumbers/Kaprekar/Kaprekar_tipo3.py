import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def es_kaprekar(numero: int) -> bool:
    if numero < 1:
        logging.error("El número debe ser un entero positivo.")
        raise ValueError("El número debe ser un entero positivo.")

    cuadrado = numero ** 2
    str_cuadrado = str(cuadrado)
    longitud = len(str_cuadrado)

    i = 1
    while i < longitud:
        parte_izquierda = int(str_cuadrado[:i])
        parte_derecha = int(str_cuadrado[i:])
        logging.debug(f"Dividiendo {cuadrado} en {parte_izquierda} y {parte_derecha}")

        if parte_derecha > 0 and (parte_izquierda + parte_derecha) == numero:
            logging.info(f"{numero} es un número de Kaprekar.")
            return True
        i += 1

    logging.info(f"{numero} no es un número de Kaprekar.")
    return False

if __name__ == "__main__":
    try:
        es_kaprekar(297)
        es_kaprekar(10)
    except ValueError as ve:
        logging.error(f"Error: {ve}")