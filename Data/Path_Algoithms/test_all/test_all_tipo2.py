import unittest

from test_gexwivji import TestGexwivji

def load_cases(loader, cases, pattern):
    cases.addTests(loader.loadTestsFromTestCase(TestGexwivji))
    return cases

if __name__ == '__main__':
    unittest.main()