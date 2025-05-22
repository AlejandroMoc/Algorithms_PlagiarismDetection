import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def es_armstrong(numero: int) -> bool:
    if numero < 0:
        logging.error("El número debe ser un entero no negativo.")
        raise ValueError("El número debe ser un entero no negativo.")

    num_str = str(numero)
    num_digitos = len(num_str)
    suma_armstrong = sum(int(digito) ** num_digitos for digito in num_str)
    logging.debug(f"Suma Armstrong para {numero}: {suma_armstrong}")

    if suma_armstrong == numero:
        logging.info(f"{numero} es un número Armstrong.")
        return True
    else:
        logging.info(f"{numero} no es un número Armstrong.")
        return False

if __name__ == "__main__":
    try:
        es_armstrong(153)
        es_armstrong(13)
    except ValueError as ve:
        logging.error(f"Error: {ve}")