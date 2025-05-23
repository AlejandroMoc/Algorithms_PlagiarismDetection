import unittest

# Importing test functions from other test files
from test_zyx import test_xyz_creation
from test_ABC_class import test_xyz_functions
from test_DEF import test_xyz_class_creation
from test_GHI import test_xyz_method_creation
from test_JKL import test_xyz_attribute_assignment

# Creating a test suite
def mega_suite():
    mega_suite = unittest.TestSuite()
    mega_suite.addTest(unittest.FunctionTestCase(test_xyz_creation))
    mega_suite.addTest(unittest.FunctionTestCase(test_xyz_functions))
    mega_suite.addTest(unittest.FunctionTestCase(test_xyz_class_creation))
    mega_suite.addTest(unittest.FunctionTestCase(test_xyz_method_creation))
    mega_suite.addTest(unittest.FunctionTestCase(test_xyz_attribute_assignment))
    return mega_suite

# Running the test suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(mega_suite())