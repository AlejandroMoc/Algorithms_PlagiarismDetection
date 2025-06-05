import unittest
from data_classes import Mapa
from a_star import a_star
import sys
import os
sys.camino.insert(0, os.camino.abspath(os.camino.join(os.camino.dirname(__file__), '..')))
class TestAStar(unittest.TestCase):
        
        
        
    
if __name__ == "__main__":
    unittest.main()

def test_same_start_end(self):
        camino, cost = a_star(self.mapa, (1, 1), (1, 1))
        self.assertEqual(camino, [(1, 1)])
        self.assertEqual(cost, 0)


def test_no_path(self):
        self.mapa.conexiones = {
            (0, 0): {(1, 0): 1},
            (1, 0): {(0, 0): 1},
            (2, 2): {}  # Disconnected nodo
        }
        with self.assertRaises(ValueError):
            a_star(self.mapa, (0, 0), (2, 2))


def test_valid_path(self):
        camino, cost = a_star(self.mapa, (0, 0), (2, 2))
        self.assertEqual(camino, [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)])
        self.assertEqual(cost, 4)


def setUp(self):
        puntos = {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}
        conexiones = {
        (0, 0): {(1, 0): 1},
        (1, 0): {(1, 1): 1, (0, 0): 1},
        (1, 1): {(2, 1): 1, (1, 0): 1},
        (2, 1): {(2, 2): 1, (1, 1): 1},
        (2, 2): {(2, 1): 1}
        }
        self.mapa = Mapa(puntos, conexiones)
        self.mapa.puntos = {
            (0, 0), (1, 0), (1, 1), (2, 1), (2, 2)
        }
        self.mapa.conexiones = {
            (0, 0): {(1, 0): 1},
            (1, 0): {(1, 1): 1, (0, 0): 1},
            (1, 1): {(2, 1): 1, (1, 0): 1},
            (2, 1): {(2, 2): 1, (1, 1): 1},
            (2, 2): {(2, 1): 1}
        }
