def sumar_elementos(lista):
    suma = 0
    for elemento in lista:
        suma += elemento
    return suma

def duplicar_elementos(lista):
    lista_duplicada = []
    for elemento in lista:
        lista_duplicada.append(elemento * 2)
    return lista_duplicada

def main():
    numeros = [1, 2, 3, 4, 5]
    
    suma_total = sumar_elementos(numeros)
    print("La suma de los elementos es:", suma_total)
    
    numeros_duplicados = duplicar_elementos(numeros)
    print("Lista con elementos duplicados:", numeros_duplicados)

if __name__ == "__main__":
    main()