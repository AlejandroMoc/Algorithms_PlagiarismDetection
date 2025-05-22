def suma_cuadrados(lista_numeros):
    suma = 0
    for num in lista_numeros:
        suma += num ** 2
    return suma

def main():
    numeros = [1, 2, 3, 4, 5]
    resultado = suma_cuadrados(numeros)
    print("La suma de los cuadrados de los números es:", resultado)

if __name__ == "__main__":
    main()