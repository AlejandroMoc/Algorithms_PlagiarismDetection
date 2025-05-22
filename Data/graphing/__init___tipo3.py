__all__ = []

def es_primo(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True

def imprimir_primos_hasta(n):
    for num in range(2, n+1):
        if es_primo(num):
            print(num)

imprimir_primos_hasta(20)