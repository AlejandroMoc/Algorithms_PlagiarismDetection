import unittest
from utils import create_graph
from dijkstra import dijkstra

class TestDijkstra(unittest.TestCase):
    
    
    
    
if __name__ == '__main__':
    unittest.main() 

def test_same_start_end(self):
        camino, distancia = dijkstra(self.mapa, 'A', 'A')
        self.assertEqual(camino, ['A'])
        self.assertEqual(distancia, 0)


def test_no_path(self):
        mapa = create_graph(['A', 'B'], {'A': {}, 'B': {}})
        with self.assertRaises(ValueError):
            dijkstra(mapa, 'A', 'B')


def test_shortest_path(self):
        camino, distancia = dijkstra(self.mapa, 'A', 'D')
        self.assertEqual(camino, ['A', 'B', 'C', 'D'])
        self.assertEqual(distancia, 4)


def setUp(self):
        puntos = ['A', 'B', 'C', 'D']
        conexiones = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }
        self.mapa = create_graph(puntos, conexiones)
