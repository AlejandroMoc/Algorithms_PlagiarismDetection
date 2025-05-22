def suma_cuadrados_lista(lista_numeros):
    suma = 0
    for num in lista_numeros:
        suma += num ** 2
    return suma

def main():
    lista = [1, 2, 3, 4, 5]
    resultado = suma_cuadrados_lista(lista)
    print("La suma de los cuadrados de los números en la lista es:", resultado)

if __name__ == "__main__":
    main()