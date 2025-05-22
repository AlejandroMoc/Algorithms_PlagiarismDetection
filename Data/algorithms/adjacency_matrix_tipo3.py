def crear_grafo(matriz_adj, cant_nodos, info_vecinos, es_dirigido=False):
    for i in range(cant_nodos):
        matriz_adj[i] = [0] * cant_nodos
        vecinos = info_vecinos.get(i, [])
        for vecino in vecinos:
            matriz_adj[i][vecino] = 1
            if not es_dirigido:
                matriz_adj[vecino][i] = 1

def mostrar_grafo(matriz_adj, cant_nodos):
    print("\nLa matriz de adyacencia es:")
    print("\t", end="")
    for i in range(cant_nodos):
        print(f"v{i+1}\t", end="")
    print()
    
    for i in range(cant_nodos):
        print(f"v{i+1}\t", end="")
        for j in range(cant_nodos):
            print(f"{matriz_adj[i][j]}\t", end="")
        print()

if __name__ == "__main__":
    cant_nodos = 5
    info_vecinos = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [2, 1, 4],
        4: [3]
    }

    matriz_adj = [[0] * cant_nodos for _ in range(cant_nodos)]
    
    es_dirigido = False
    
    crear_grafo(matriz_adj, cant_nodos, info_vecinos, es_dirigido)
    
    mostrar_grafo(matriz_adj, cant_nodos)