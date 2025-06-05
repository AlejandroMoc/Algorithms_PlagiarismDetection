import unittest

from test_dijkstra import TestDijkstra


if __name__ == '__main__':
    unittest.main() 

def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromTestCase(TestDijkstra))
    return tests
