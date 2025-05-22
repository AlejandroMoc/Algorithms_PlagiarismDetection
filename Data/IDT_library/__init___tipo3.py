def suma_cuadrados_lista(lista):
    total = 0
    for num in lista:
        total += num**2
    return total

def main():
    mi_lista = [1, 2, 3, 4, 5]
    resultado = suma_cuadrados_lista(mi_lista)
    print(f"La suma de los cuadrados de la lista es: {resultado}")

if __name__ == "__main__":
    main()