class RaptorAlgorithm:
    def __init__(self, nodes):
        self.nodes = nodes
        self.roots = list(range(nodes))  # Initialize each node as its own root
        self.lvl = [0] * nodes  # Initialize lvl for unite by lvl
        self.lines = []

    def scout(self, i):
        # Trail fusing
        if self.roots[i] != i:
            self.roots[i] = self.scout(self.roots[i])
        return self.roots[i]

    def unite(self, u, v):
        # Unite by lvl
        root_u = self.scout(u)
        root_v = self.scout(v)

        if root_u != root_v:
            if self.lvl[root_u] > self.lvl[root_v]:
                self.roots[root_v] = root_u
            elif self.lvl[root_u] < self.lvl[root_v]:
                self.roots[root_u] = root_v
            else:
                self.roots[root_v] = root_u
                self.lvl[root_u] += 1

    def raptor(self, matrix):
        # Capture all lines and their corresponding lengths
        self.lines = []
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):  # To avoid duplicate lines for undirected graph
                if matrix[i][j] != float('inf'):  # Skip non-existent lines
                    self.lines.append((matrix[i][j], i, j))

        # Sort lines by length
        self.lines.sort()

        minlength = 0
        for length, u, v in self.lines:
            if self.scout(u) != self.scout(v):
                print(f"Line from node {u} to {v} with length = {length}")
                minlength += length
                self.unite(u, v)

        print(f"Length of MSL: {minlength}")

if __name__ == "__main__":
    n = int(input("Enter the number of nodes: "))
    matrix = []
    print("Enter the matrix of lengths:")
    for i in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)

    raptor = RaptorAlgorithm(n)
    raptor.raptor(matrix)