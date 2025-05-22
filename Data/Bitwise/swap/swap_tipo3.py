def intercambiar_numeros(x, y):
    x ^= y
    y ^= x
    x ^= y
    return x, y