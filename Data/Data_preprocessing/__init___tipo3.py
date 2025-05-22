def suma_cuadrados_lista(lista):
    resultado = 0
    for num in lista:
        resultado += num ** 2
    return resultado

def es_primo(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def filtrar_primos(lista_numeros):
    primos = []
    for num in lista_numeros:
        if es_primo(num):
            primos.append(num)
    return primos

numeros = [2, 3, 5, 7, 9, 11, 13, 17, 19, 23]
resultado = suma_cuadrados_lista(numeros)
print("La suma de los cuadrados de los números en la lista es:", resultado)

numeros_primos = filtrar_primos(numeros)
print("Números primos en la lista:", numeros_primos)