def suma_cubos(num):
    suma = 0
    for i in range(1, num + 1):
        suma += i ** 3
    return suma

def main():
    numero = int(input("Ingresa un número entero positivo: "))
    resultado = suma_cubos(numero)
    print(f"La suma de los cubos de los primeros {numero} números enteros es: {resultado}")

if __name__ == "__main__":
    main()