def obtener_lista_pares(numeros):
    lista_pares = []
    for num in numeros:
        if num % 2 == 0:
            lista_pares.append(num)
    return lista_pares

def calcular_promedio(lista):
    suma = 0
    for num in lista:
        suma += num
    return suma / len(lista)

numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
numeros_pares = obtener_lista_pares(numeros)
promedio_pares = calcular_promedio(numeros_pares)

print("Lista original:", numeros)
print("Lista de números pares:", numeros_pares)
print("Promedio de los números pares:", promedio_pares)