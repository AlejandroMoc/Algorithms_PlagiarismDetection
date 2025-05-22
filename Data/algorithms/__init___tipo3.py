def suma_cuadrados(numeros):
    suma = 0
    for num in numeros:
        suma += num ** 2
    return suma

def es_par(numero):
    return numero % 2 == 0

def filtrar_pares(lista_numeros):
    pares = []
    for num in lista_numeros:
        if es_par(num):
            pares.append(num)
    return pares

numeros = [1, 2, 3, 4, 5]
resultado = suma_cuadrados(numeros)
print("La suma de los cuadrados de los números es:", resultado)

numeros = [10, 15, 20, 25, 30]
pares = filtrar_pares(numeros)
print("Números pares de la lista:", pares)