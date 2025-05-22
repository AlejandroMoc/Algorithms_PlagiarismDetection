def suma_cuadrados(numeros):
    suma = 0
    for num in numeros:
        suma += num ** 2
    return suma

def main():
    lista_numeros = [1, 2, 3, 4, 5]
    resultado = suma_cuadrados(lista_numeros)
    print("La suma de los cuadrados es:", resultado)

if __name__ == "__main__":
    main()