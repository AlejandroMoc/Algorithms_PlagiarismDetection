import unittest
from xzq import XZQSolver

class TestXZQSolver(unittest.TestCase):
    
    def test_xzq_standard_case(self):
        self.assertEqual(XZQSolver.xzq("AGGTAB", "GXTXAYB", 6, 7), 4)
    
    def test_xzq_no_common_subsequence(self):
        self.assertEqual(XZQSolver.xzq("ABC", "DEF", 3, 3), 0)
    
    def test_xzq_empty_string(self):
        self.assertEqual(XZQSolver.xzq("", "ABC", 0, 3), 0)
        self.assertEqual(XZQSolver.xzq("ABC", "", 3, 0), 0)
    
    def test_xzq_complete_overlap(self):
        self.assertEqual(XZQSolver.xzq("ABC", "ABC", 3, 3), 3)

if __name__ == "__main__":
    unittest.main()