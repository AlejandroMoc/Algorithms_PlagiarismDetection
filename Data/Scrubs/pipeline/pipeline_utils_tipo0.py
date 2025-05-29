import time

def calcular_tiempo(funcion):
    def cronometrado(*args, **kwargs):
        """
        Decora una función para imprimir su tiempo de ejecución.
        """
        mensaje_superior = "\nIniciando {}".format(funcion.__name__)
        print(mensaje_superior)
        print("-" * len(mensaje_superior))
        t0 = time.time()

        resultado = funcion(*args, **kwargs)

        mensaje_inferior = "\nCompletado en {:.2f} minutos.".format((time.time() - t0)/60)
        print(mensaje_inferior)
        print('-' * len(mensaje_inferior))
        return resultado
    return cronometrado