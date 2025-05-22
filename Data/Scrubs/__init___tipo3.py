def suma_cuadrados_lista(lista_numeros):
    suma = 0
    for numero in lista_numeros:
        suma += numero ** 2
    return suma

def main():
    numeros = [1, 2, 3, 4, 5]
    suma_cuadrados = suma_cuadrados_lista(numeros)
    print("La suma de los cuadrados de los números es:", suma_cuadrados)

if __name__ == "__main__":
    main()