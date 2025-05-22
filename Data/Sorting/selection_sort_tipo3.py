def orden_seleccion(lista):
    longitud = len(lista)
    for i in range(longitud):
        indice_min = i
        for j in range(i + 1, longitud):
            if lista[j] < lista[indice_min]:
                indice_min = j
        lista[i], lista[indice_min] = lista[indice_min], lista[i]
    return lista

if __name__ == "__main__":
    # Entrada: enteros separados por espacios
    datos_entrada = input("Ingresa los números a ordenar, separados por espacios: ")
    lista = list(map(int, datos_entrada.split()))
    
    lista_ordenada = orden_seleccion(lista)
    print("Lista ordenada:", lista_ordenada)