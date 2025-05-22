import unittest
from Sorting.topological_sort import topological_sort, Graph

class TestTopologicalSort(unittest.TestCase):
    def test_topological_sort_valid_dag(self):
        vertices = ["A", "B", "C", "D", "E", "F"]
        arcs = [("A", "D"), ("F", "B"), ("B", "D"), ("F", "A"), ("D", "C")]
        graph = Graph(nodes=vertices, edges=arcs)
        result = topological_sort(graph)
        valid_orders = [
            ['F', 'E', 'B', 'A', 'D', 'C'],
            ['E', 'F', 'B', 'A', 'D', 'C'],
            ['F', 'B', 'E', 'A', 'D', 'C'],
            # Additional valid topological orders can be added here
        ]
        self.assertIn(result, valid_orders)

    def test_topological_sort_empty_graph(self):
        graph = Graph(nodes=[], edges=[])
        expected = []
        result = topological_sort(graph)
        self.assertEqual(result, expected)

    def test_topological_sort_single_node(self):
        graph = Graph(nodes=["A"], edges=[])
        expected = ["A"]
        result = topological_sort(graph)
        self.assertEqual(result, expected)

    def test_topological_sort_cycle(self):
        vertices = ["A", "B", "C"]
        arcs = [("A", "B"), ("B", "C"), ("C", "A")]
        graph = Graph(nodes=vertices, edges=arcs)
        with self.assertRaises(ValueError):
            topological_sort(graph)

    def test_topological_sort_disconnected_graph(self):
        vertices = ["A", "B", "C", "D"]
        arcs = [("A", "B"), ("C", "D")]
        graph = Graph(nodes=vertices, edges=arcs)
        result = topological_sort(graph)
        # Verify that dependencies are maintained
        self.assertTrue(
            result.index("A") < result.index("B") and
            result.index("C") < result.index("D")
        )

    def test_topological_sort_multiple_valid_orders(self):
        vertices = ["1", "2", "3", "4"]
        arcs = [("1", "2"), ("1", "3"), ("3", "4")]
        graph = Graph(nodes=vertices, edges=arcs)
        result = topological_sort(graph)
        valid_orders = [
            ["1", "2", "3", "4"],
            ["1", "3", "2", "4"],
            # Additional valid orders can be included
        ]
        self.assertIn(result, valid_orders)

if __name__ == "__main__":
    unittest.main()