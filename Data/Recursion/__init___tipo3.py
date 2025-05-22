def suma_pares(lista_numeros):
    suma = 0
    for num in lista_numeros:
        if num % 2 == 0:
            suma += num
    return suma

def main():
    numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    resultado = suma_pares(numeros)
    print("La suma de los números pares en la lista es:", resultado)

if __name__ == "__main__":
    main()