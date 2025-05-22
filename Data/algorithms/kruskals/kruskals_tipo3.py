class KruskalAlgorithm:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.parent = list(range(num_vertices))  # Initialize each node as its own parent
        self.rank = [0] * num_vertices  # Initialize rank for union by rank
        self.edges = []

    def find_parent(self, node):
        # Path compression
        if self.parent[node] != node:
            self.parent[node] = self.find_parent(self.parent[node])
        return self.parent[node]

    def union_sets(self, node_u, node_v):
        # Union by rank
        root_u = self.find_parent(node_u)
        root_v = self.find_parent(node_v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

    def find_mst(self, cost_matrix):
        # Extract all edges and their corresponding weights
        self.edges = []
        for i in range(self.num_vertices):
            for j in range(i + 1, self.num_vertices):  # To avoid duplicate edges for undirected graph
                if cost_matrix[i][j] != float('inf'):  # Skip non-existent edges
                    self.edges.append((cost_matrix[i][j], i, j))

        # Sort edges by cost
        self.edges.sort()

        min_cost = 0
        for cost, node_u, node_v in self.edges:
            if self.find_parent(node_u) != self.find_parent(node_v):
                print(f"Edge from vertex {node_u} to {node_v} with cost = {cost}")
                min_cost += cost
                self.union_sets(node_u, node_v)

        print(f"Cost of MST: {min_cost}")

if __name__ == "__main__":
    num_vertices = int(input("Enter the number of vertices: "))
    cost_matrix = []
    print("Enter the cost matrix:")
    for i in range(num_vertices):
        row = list(map(int, input().split()))
        cost_matrix.append(row)

    kruskal_algo = KruskalAlgorithm(num_vertices)
    kruskal_algo.find_mst(cost_matrix)