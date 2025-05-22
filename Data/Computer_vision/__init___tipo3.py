def suma_cubos(n):
    suma = 0
    for i in range(1, n+1):
        suma += i**3
    return suma

def es_primo(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def imprimir_primos(n):
    contador = 0
    num = 2
    while contador < n:
        if es_primo(num):
            print(num)
            contador += 1
        num += 1

n = 5
resultado = suma_cubos(n)
print(f"La suma de los cubos de los primeros {n} números naturales es: {resultado}")

imprimir_primos(10)