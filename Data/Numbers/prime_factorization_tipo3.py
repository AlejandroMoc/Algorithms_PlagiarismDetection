def factorizacion_primos(numero: int) -> list:
    if not isinstance(numero, int):
        raise TypeError("El input debe ser un entero.")
    if numero <= 0:
        raise ValueError("El input debe ser un entero positivo.")

    factores = []
    divisor = 2
    while numero > 1:
        while numero % divisor == 0:
            factores.append(divisor)
            numero //= divisor
        divisor += 1
    return factores


if __name__ == "__main__":
    # Ejemplo de uso
    resultado = factorizacion_primos(28)
    print(resultado)  # Salida: [2, 2, 7]