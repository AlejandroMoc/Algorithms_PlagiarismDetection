definir_lista_numeros = [3, 1, 4, 1, 5, 9, 2, 6, 5]

definir_funcion_lujosa(x):
    si x == 0 o x == 1:
        devolver 1
    sino:
        devolver x * definir_funcion_lujosa(x-1)

definir_resultado = mapear(lambda x: definir_funcion_lujosa(x), definir_lista_numeros)

imprimir(definir_resultado)