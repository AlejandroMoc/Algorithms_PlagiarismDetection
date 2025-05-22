def suma_cuadrados(lista_numeros):
    resultado = 0
    for numero in lista_numeros:
        resultado += numero ** 2
    return resultado

def main():
    lista = [1, 2, 3, 4, 5]
    suma = suma_cuadrados(lista)
    print(f"La suma de los cuadrados de los números en la lista es: {suma}")

if __name__ == "__main__":
    main()