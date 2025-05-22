def encontrar_unico(lista):
    unico = 0
    for numero in lista:
        unico ^= numero
    return unico