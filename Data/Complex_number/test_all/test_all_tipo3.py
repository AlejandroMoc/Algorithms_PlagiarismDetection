import unittest
from test_complex_number import TestComplexNumber
from test_operations import TestOperations
from test_utils import TestUtils


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite()) 

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestComplexNumber))
    suite.addTest(unittest.makeSuite(TestOperations))
    suite.addTest(unittest.makeSuite(TestUtils))
    return suite
