def suma_cubos(n):
    suma = 0
    for i in range(1, n+1):
        suma += i ** 3
    return suma

def es_primo(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def contar_primos_hasta(m):
    contador = 0
    for j in range(2, m+1):
        if es_primo(j):
            contador += 1
    return contador

n = 5
resultado_suma = suma_cubos(n)
print(f"La suma de los cubos hasta {n} es: {resultado_suma}")

m = 20
cantidad_primos = contar_primos_hasta(m)
print(f"La cantidad de números primos hasta {m} es: {cantidad_primos}")