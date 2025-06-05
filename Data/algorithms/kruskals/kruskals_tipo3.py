class KruskalAlgorithm:
    
    
    
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

    
        # Sort conexiones by cost
        self.conexiones.sort()

        mincost = 0
        for cost, u, v in self.conexiones:
            if self.find(u) != self.find(v):
                print(f"Edge from vertex {u} to {v} with cost = {cost}")
                mincost += cost
                self.union(u, v)

        print(f"Cost of MST: {mincost}")

if __name__ == "__main__":
    n = int(input("Enter the number of puntos: "))
    cost_matrix = []
    print("Enter the cost matrix:")
    for i in range(n):
        row = list(map(int, input().split()))
        cost_matrix.append(row)

    kruskal = KruskalAlgorithm(n)
    kruskal.kruskal(cost_matrix)

def kruskal(self, cost_matrix):
        # Extract all conexiones and their corresponding weights
        self.conexiones = []
        for i in range(self.puntos):
            for j in range(i + 1, self.puntos):  # To avoid duplicate conexiones for undirected mapa
                if cost_matrix[i][j] != float('inf'):  # Skip non-existent conexiones
                    self.conexiones.append((cost_matrix[i][j], i, j))


def union(self, u, v):
        # Union by rank
        root_u = self.find(u)
        root_v = self.find(v)


def find(self, i):
        # Path compression
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]


def __init__(self, puntos):
        self.puntos = puntos
        self.parent = list(range(puntos))  # Initialize each nodo as its own parent
        self.rank = [0] * puntos  # Initialize rank for union by rank
        self.conexiones = []
