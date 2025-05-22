import heapq

def find_shortest_path(graph, start):
    priority_queue = [(0, start)]
    distances = {start: 0}

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
      
        for neighbor, weight in graph[current_node]:
            dist = current_dist + weight
         
            if neighbor not in distances or dist < distances[neighbor]:
                distances[neighbor] = dist
                heapq.heappush(priority_queue, (dist, neighbor))

    return distances

if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }
    print(find_shortest_path(graph, 'A'))