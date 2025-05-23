import unittest
from test_complex_number import TestComplexNumber
from test_operations import TestOperations
from test_utils import TestUtils

def conjunto_de_pruebas():
    conjunto = unittest.TestSuite()
    conjunto.addTest(unittest.makeSuite(TestComplexNumber))
    conjunto.addTest(unittest.makeSuite(TestOperations))
    conjunto.addTest(unittest.makeSuite(TestUtils))
    return conjunto

if __name__ == '__main__':
    corredor = unittest.TextTestRunner()
    corredor.run(conjunto_de_pruebas())