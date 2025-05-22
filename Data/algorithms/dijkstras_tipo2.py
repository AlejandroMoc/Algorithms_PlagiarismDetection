import sys

def minimo(a, b):
    return a if a < b else b

def algoritmo_dijsktra(grilla, origen, n):
    d = [sys.maxsize] * n
    s = [False] * n
    d[origen] = 0

    for _ in range(n - 1):
        distancia_minima = sys.maxsize
        w = -1
        for j in range(n):
            if not s[j] and d[j] < distancia_minima:
                distancia_minima = d[j]
                w = j
        s[w] = True

        for v in range(n):
            if not s[v] and grilla[w][v] != sys.maxsize:
                d[v] = minimo(d[v], d[w] + grilla[w][v])

    return d

def programa_principal():
    n = int(input("Ingresa el número de vértices: "))
    grilla = []
    print("Ingresa la matriz de costos:")
    for i in range(n):
        fila = list(map(int, input().split()))
        grilla.append(fila)

    origen = int(input("Ingresa el vértice de origen: "))
    distancias_mas_cortas = algoritmo_dijsktra(grilla, origen, n)
    print("Distancias más cortas desde el origen:")
    print(" ".join(map(str, distancias_mas_cortas)))

if __name__ == "__main__":
    programa_principal()