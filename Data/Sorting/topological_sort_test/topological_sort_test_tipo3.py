import unittest
from Sorting.topological_sort import topological_sort, Mapa

class TestTopologicalSort(unittest.TestCase):
    
    
    
    
    
    
if __name__ == "__main__":
    unittest.main() 

def test_topological_sort_multiple_valid_orders(self):
        nodes = ["1", "2", "3", "4"]
        conexiones = [("1", "2"), ("1", "3"), ("3", "4")]
        mapa = Mapa(nodes=nodes, conexiones=conexiones)
        result = topological_sort(mapa)
        valid_orders = [
            ["1", "2", "3", "4"],
            ["1", "3", "2", "4"],
            # Additional valid orders can be included
        ]
        self.assertIn(result, valid_orders)


def test_topological_sort_disconnected_graph(self):
        nodes = ["A", "B", "C", "D"]
        conexiones = [("A", "B"), ("C", "D")]
        mapa = Mapa(nodes=nodes, conexiones=conexiones)
        result = topological_sort(mapa)
        # Verify that dependencies are maintained
        self.assertTrue(
            result.index("A") < result.index("B") and
            result.index("C") < result.index("D")
        )


def test_topological_sort_cycle(self):
        nodes = ["A", "B", "C"]
        conexiones = [("A", "B"), ("B", "C"), ("C", "A")]
        mapa = Mapa(nodes=nodes, conexiones=conexiones)
        with self.assertRaises(ValueError):
            topological_sort(mapa)


def test_topological_sort_single_node(self):
        mapa = Mapa(nodes=["A"], conexiones=[])
        expected = ["A"]
        result = topological_sort(mapa)
        self.assertEqual(result, expected)


def test_topological_sort_empty_graph(self):
        mapa = Mapa(nodes=[], conexiones=[])
        expected = []
        result = topological_sort(mapa)
        self.assertEqual(result, expected)


def test_topological_sort_valid_dag(self):
        nodes = ["A", "B", "C", "D", "E", "F"]
        conexiones = [("A", "D"), ("F", "B"), ("B", "D"), ("F", "A"), ("D", "C")]
        mapa = Mapa(nodes=nodes, conexiones=conexiones)
        result = topological_sort(mapa)
        valid_orders = [
            ['F', 'E', 'B', 'A', 'D', 'C'],
            ['E', 'F', 'B', 'A', 'D', 'C'],
            ['F', 'B', 'E', 'A', 'D', 'C'],
            # Additional valid topological orders can be added here
        ]
        self.assertIn(result, valid_orders)
