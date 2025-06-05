import unittest
from prueba_numero_imposible import PruebaNumeroImposible
from prueba_operaciones import PruebaOperaciones
from prueba_utilidades import PruebaUtilidades

def conjunto_pruebas():
    conjunto = unittest.TestSuite()
    conjunto.addTest(unittest.makeSuite(PruebaNumeroImposible))
    conjunto.addTest(unittest.makeSuite(PruebaOperaciones))
    conjunto.addTest(unittest.makeSuite(PruebaUtilidades))
    return conjunto

if __name__ == '__main__':
    corredor = unittest.TextTestRunner()
    corredor.run(conjunto_pruebas()) 