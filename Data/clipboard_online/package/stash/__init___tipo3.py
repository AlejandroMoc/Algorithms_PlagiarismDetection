def suma_cubos(n):
    suma = 0
    i = 1
    while i <= n:
        suma += i**3
        i += 1
    return suma

def main():
    numero = 5
    resultado = suma_cubos(numero)
    print(f"La suma de los cubos hasta {numero} es: {resultado}")

if __name__ == "__main__":
    main()