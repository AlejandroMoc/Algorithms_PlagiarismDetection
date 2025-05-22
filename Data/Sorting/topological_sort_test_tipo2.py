import unittest
from Sorting.topological_sort import topological_sort, Graph

class TestRenamedAlgorithms(unittest.TestCase):
    def test_renamed_algorithms_valid_dag(self):
        nodes_list = ["X", "Y", "Z", "W", "V", "U"]
        edges_list = [("X", "W"), ("U", "Y"), ("Y", "W"), ("U", "X"), ("W", "V")]
        graph_obj = Graph(nodes=nodes_list, edges=edges_list)
        result_list = topological_sort(graph_obj)
        valid_sequences = [
            ['U', 'V', 'Y', 'X', 'W', 'Z'],
            ['V', 'U', 'Y', 'X', 'W', 'Z'],
            ['U', 'Y', 'V', 'X', 'W', 'Z'],
            # Additional valid topological orders can be added here
        ]
        self.assertIn(result_list, valid_sequences)

    def test_renamed_algorithms_empty_structure(self):
        graph_obj = Graph(nodes=[], edges=[])
        expected_list = []
        result_list = topological_sort(graph_obj)
        self.assertEqual(result_list, expected_list)

    def test_renamed_algorithms_single_vertex(self):
        graph_obj = Graph(nodes=["X"], edges=[])
        expected_list = ["X"]
        result_list = topological_sort(graph_obj)
        self.assertEqual(result_list, expected_list)

    def test_renamed_algorithms_cycle(self):
        nodes_list = ["X", "Y", "Z"]
        edges_list = [("X", "Y"), ("Y", "Z"), ("Z", "X")]
        graph_obj = Graph(nodes=nodes_list, edges=edges_list)
        with self.assertRaises(ValueError):
            topological_sort(graph_obj)

    def test_renamed_algorithms_disconnected_structure(self):
        nodes_list = ["X", "Y", "Z", "W"]
        edges_list = [("X", "Y"), ("Z", "W")]
        graph_obj = Graph(nodes=nodes_list, edges=edges_list)
        result_list = topological_sort(graph_obj)
        # Verify that dependencies are maintained
        self.assertTrue(
            result_list.index("X") < result_list.index("Y") and
            result_list.index("Z") < result_list.index("W")
        )

    def test_renamed_algorithms_multiple_valid_orders(self):
        nodes_list = ["1", "2", "3", "4"]
        edges_list = [("1", "2"), ("1", "3"), ("3", "4")]
        graph_obj = Graph(nodes=nodes_list, edges=edges_list)
        result_list = topological_sort(graph_obj)
        valid_orders = [
            ["1", "2", "3", "4"],
            ["1", "3", "2", "4"],
            # Additional valid orders can be included
        ]
        self.assertIn(result_list, valid_orders)

if __name__ == "__main__":
    unittest.main()