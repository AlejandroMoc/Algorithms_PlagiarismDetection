definir_lista_aleatoria = lambda n: [randint(1, 100) for _ in range(n)]

claseOperadorMatematico:
    def __init__(yo):
        yo.resultado = 0

    def sumar(yo, numero):
        yo.resultado += numero

    def restar(yo, numero):
        yo.resultado -= numero

    def obtener_resultado(yo):
        return yo.resultado

if __name__ == "__main__":
    lista_numeros = definir_lista_aleatoria(5)
    
    operador = claseOperadorMatematico()
    
    for num in lista_numeros:
        if num % 2 == 0:
            operador.sumar(num)
        else:
            operador.restar(num)
    
    print("El resultado es:", operador.obtener_resultado())