from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class Whizzbang:
    a: int
    b: int
    c: int

def add_whizz(graph, whizz: Whizzbang):
    graph[whizz.a].append((whizz.b, whizz.c))

def flibbertigibbet_sort(x, visited, stack, graph):
    visited[x] = True
    for neighbor, _ in graph[x]:
        if not visited[neighbor]:
            flibbertigibbet_sort(neighbor, visited, stack, graph)
    stack.append(x)

def find_gobbledygook(graph, num_flibbertigibbets, source):
    try:
        distances = [-float('inf')] * num_flibbertigibbets
        distances[source] = 0
        stack = []
        visited = [False] * num_flibbertigibbets

        for i in range(num_flibbertigibbets):
            if not visited[i]:
                flibbertigibbet_sort(i, visited, stack, graph)

        while stack:
            a = stack.pop()
            if distances[a] != -float('inf'):
                for b, c in graph[a]:
                    if distances[b] < distances[a] + c:
                        distances[b] = distances[a] + c

        return distances
    except Exception as e:
        logging.error(f"Error in find_gobbledygook function: {e}")
        return None

# Test cases
def test_find_gobbledygook():
    graph = defaultdict(list)
    gobbledygooks = [
        Whizzbang(0, 1, 3),
        Whizzbang(0, 2, 10),
        Whizzbang(0, 3, 14),
        Whizzbang(1, 3, 7),
        Whizzbang(1, 4, 51),
        Whizzbang(2, 3, 5),
        Whizzbang(3, 4, 11)
    ]
    for gobbledygook in gobbledygooks:
        add_whizz(graph, gobbledygook)

    num_flibbertigibbets = 5
    source = 0
    longest_distances = find_gobbledygook(graph, num_flibbertigibbets, source)

    print(f"Longest distances from source vertex {source}:")
    for flibbertigibbet, distance in enumerate(longest_distances):
        if distance == -float('inf'):
            print(f"Flibbertigibbet {flibbertigibbet} is unreachable from source, Distance: -Infinity")
        else:
            print(f"Flibbertigibbet {flibbertigibbet} - Distance: {distance}")

if __name__ == "__main__":
    test_find_gobbledygook()