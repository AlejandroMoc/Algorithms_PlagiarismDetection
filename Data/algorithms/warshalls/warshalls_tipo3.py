python
def warshall_algorithm_modified(num_vertices, matrix):
    transitive_closure_matrix = [[matrix[i][j] for j in range(num_vertices)] for i in range(num_vertices)]

    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if transitive_closure_matrix[i][j] == 0:
                    transitive_closure_matrix[i][j] = transitive_closure_matrix[i][k] and transitive_closure_matrix[k][j]

    return transitive_closure_matrix

def main():
    try:
        num_vertices = int(input("Enter the number of vertices: "))
        if num_vertices <= 0:
            print("The number of vertices must be positive!")
            return

        adjacency_matrix = []

        print("Enter the adjacency matrix (each row of the matrix should be space-separated values):")
        for i in range(num_vertices):
            row = list(map(int, input(f"Row {i + 1}: ").split()))
            if len(row) != num_vertices:
                print(f"Row {i + 1} does not contain {num_vertices} elements. Please try again.")
                return
            adjacency_matrix.append(row)

        transitive_closure_result = warshall_algorithm_modified(num_vertices, adjacency_matrix)

        print("Transitive closure:")
        for row in transitive_closure_result:
            print("\t".join(map(str, row)))

    except ValueError:
        print("Invalid input! Please enter integers.")

if __name__ == "__main__":
    main()