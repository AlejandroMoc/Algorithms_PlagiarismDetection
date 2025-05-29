import unittest
from helpers import create_graph
from dijkstra import dijkstra_algorithm

class TestDijkstra(unittest.TestCase):
    def setUp(self):
        nodes = ['A', 'B', 'C', 'D']
        connections = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }
        self.graph = create_graph(nodes, connections)

    def test_shortest_route(self):
        route, distance = dijkstra_algorithm(self.graph, 'A', 'D')
        self.assertEqual(route, ['A', 'B', 'C', 'D'])
        self.assertEqual(distance, 4)

    def test_no_route(self):
        graph = create_graph(['A', 'B'], {'A': {}, 'B': {}})
        with self.assertRaises(ValueError):
            dijkstra_algorithm(graph, 'A', 'B')

    def test_same_start_and_end(self):
        route, distance = dijkstra_algorithm(self.graph, 'A', 'A')
        self.assertEqual(route, ['A'])
        self.assertEqual(distance, 0)

if __name__ == '__main__':
    unittest.main()