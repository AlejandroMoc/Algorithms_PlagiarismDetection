import sys

def min_val(a, b):
    return a if a < b else b

def dijkstra_algorithm(cost_matrix, source, num_vertices):
    distances = [sys.maxsize] * num_vertices
    visited = [False] * num_vertices
    distances[source] = 0

    for _ in range(num_vertices - 1):
        min_dist = sys.maxsize
        w_vertex = -1
        for j_idx in range(num_vertices):
            if not visited[j_idx] and distances[j_idx] < min_dist:
                min_dist = distances[j_idx]
                w_vertex = j_idx
        visited[w_vertex] = True

        for v_idx in range(num_vertices):
            if not visited[v_idx] and cost_matrix[w_vertex][v_idx] != sys.maxsize:
                distances[v_idx] = min_val(distances[v_idx], distances[w_vertex] + cost_matrix[w_vertex][v_idx])

    return distances

def main():
    num_vertices = int(input("Enter number of vertices: "))
    cost_matrix = []
    print("Enter cost matrix:")
    for i in range(num_vertices):
        row_data = list(map(int, input().split()))
        cost_matrix.append(row_data)

    source_vertex = int(input("Enter source vertex: "))
    shortest_distances = dijkstra_algorithm(cost_matrix, source_vertex, num_vertices)
    print("Shortest distances from the source:")
    print(" ".join(map(str, shortest_distances)))

if __name__ == "__main__":
    main()