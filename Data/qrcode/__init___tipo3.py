def suma_cubos(n):
    return sum(i**3 for i in range(1, n+1))

def main():
    numero = int(input("Ingresa un número entero positivo: "))
    resultado = suma_cubos(numero)
    print(f"La suma de los cubos de los primeros {numero} números enteros es: {resultado}")

if __name__ == "__main__":
    main()