def suma_cuadrados(lista):
    suma = 0
    for num in lista:
        suma += num ** 2
    return suma

def multiplicar_y_sumar(num1, num2):
    return num1 * num2 + num1 + num2

def main():
    numeros = [1, 2, 3, 4, 5]
    resultado = suma_cuadrados(numeros)
    print("La suma de los cuadrados de los números es:", resultado)

    resultado_final = multiplicar_y_sumar(resultado, len(numeros))
    print("El resultado final es:", resultado_final)

if __name__ == "__main__":
    main()