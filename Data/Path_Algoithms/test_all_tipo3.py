import unittest

from test_dijkstra import TestDijkstra

def cargar_pruebas(cargador, pruebas, patron):
    pruebas.addTests(cargador.loadTestsFromTestCase(TestDijkstra))
    return pruebas

if __name__ == '__main__':
    unittest.main()