def suma_cuadrados(numeros):
    suma = 0
    for num in numeros:
        suma += num ** 2
    return suma

def promedio(numeros):
    total = suma_cuadrados(numeros)
    promedio = total / len(numeros)
    return promedio

def main():
    nums = [1, 2, 3, 4, 5]
    prom = promedio(nums)
    print("El promedio de los cuadrados de los números es:", prom)

if __name__ == "__main__":
    main()