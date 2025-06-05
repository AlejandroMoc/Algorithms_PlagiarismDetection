def encontrar_caminos_unicos(filas, columnas):
    if not (isinstance(filas, int) and isinstance(columnas, int)) or filas <= 0 or columnas <= 0:
        raise ValueError("Ambos filas y columnas deben ser enteros positivos.")

    dp = [1] * columnas

    for i in range(1, filas):
        for j in range(1, columnas):
            dp[j] += dp[j - 1]

    return dp[-1]

def test_encontrar_caminos_unicos():
    casos_prueba = [
        (3, 7, 28),
        (3, 2, 3),
        (7, 3, 28)
    ]
    
    for filas, columnas, esperado in casos_prueba:
        resultado = encontrar_caminos_unicos(filas, columnas)
        assert resultado == esperado, f"Esperado {esperado}, obtenido {resultado}"

if __name__ == "__main__":
    casos_prueba = [
        (3, 7),
        (3, 2),
        (1, 1),
        (5, 5),
        (2, 3),
        (7, 3),
        (10, 10)
    ]
    
    for filas, columnas in casos_prueba:
        num_caminos = encontrar_caminos_unicos(filas, columnas)
        print(f"El número de caminos únicos en una cuadrícula de tamaño {filas} x {columnas} es: {num_caminos}")