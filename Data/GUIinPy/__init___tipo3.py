def verificar_primo(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def imprimir_primos_hasta(max_num):
    for numero in range(2, max_num + 1):
        if verificar_primo(numero):
            print(numero)

limite = 20
imprimir_primos_hasta(limite)