def suma_cuadrados(numeros):
    suma = 0
    for num in numeros:
        suma += num ** 2
    return suma

def producto_factorial(numeros):
    producto = 1
    for num in numeros:
        factorial = 1
        for i in range(1, num + 1):
            factorial *= i
        producto *= factorial
    return producto

numeros = [1, 2, 3, 4, 5]

suma_resultado = suma_cuadrados(numeros)
print("La suma de los cuadrados es:", suma_resultado)

producto_resultado = producto_factorial(numeros)
print("El producto de los factoriales es:", producto_resultado)