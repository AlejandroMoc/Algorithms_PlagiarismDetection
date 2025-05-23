import unittest
from operations import generate_diagram
from shortestpath import shortest_path_algorithm

class TestingShortestPath(unittest.TestCase):
    def before_testing(self):
        nodes = ['X', 'Y', 'Z', 'W']
        connections = {
            'X': {'Y': 1, 'Z': 4},
            'Y': {'Z': 2, 'W': 5},
            'Z': {'W': 1},
            'W': {}
        }
        self.diagram = generate_diagram(nodes, connections)

    def test_finding_shortest(self):
        path, distance = shortest_path_algorithm(self.diagram, 'X', 'W')
        self.assertEqual(path, ['X', 'Y', 'Z', 'W'])
        self.assertEqual(distance, 4)

    def test_no_possible_path(self):
        diagram = generate_diagram(['X', 'Y'], {'X': {}, 'Y': {}})
        with self.assertRaises(ValueError):
            shortest_path_algorithm(diagram, 'X', 'Y')

    def test_same_origin_destination(self):
        path, distance = shortest_path_algorithm(self.diagram, 'X', 'X')
        self.assertEqual(path, ['X'])
        self.assertEqual(distance, 0)

if __name__ == '__main__':
    unittest.main()