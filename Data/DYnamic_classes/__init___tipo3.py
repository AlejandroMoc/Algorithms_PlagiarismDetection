def obtener_suma_cuadrados(n):
    suma = 0
    for i in range(1, n + 1):
        suma += i ** 2
    return suma

def main():
    numero = 5
    resultado = obtener_suma_cuadrados(numero)
    print(f"La suma de los cuadrados hasta {numero} es: {resultado}")

main()