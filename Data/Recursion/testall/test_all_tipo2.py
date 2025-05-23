import unittest
from factorization import Factorization
from fabonacci import Fabonacci
from binary_hunt import BinaryHunt
from parts import Parts
from rearranger import Rearranger
from fusion_sort import FusionSort
from rapid_sort import RapidSort

class TestRecursiveSnippets(unittest.TestCase):

    def test_factorization(self):
        self.assertEqual(Factorization(0).calculate(), 1)
        self.assertEqual(Factorization(1).calculate(), 1)
        self.assertEqual(Factorization(5).calculate(), 120)
        self.assertEqual(Factorization(10).calculate(), 3628800)

    def test_fabonacci(self):
        self.assertEqual(Fabonacci(0).calculate(), 0)
        self.assertEqual(Fabonacci(1).calculate(), 1)
        self.assertEqual(Fabonacci(5).calculate(), 5)
        self.assertEqual(Fabonacci(10).calculate(), 55)

    def test_binary_hunt(self):
        self.assertEqual(BinaryHunt([1, 2, 3, 4, 5], 3).search(), 2)
        self.assertEqual(BinaryHunt([1, 2, 3, 4, 5], 1).search(), 0)
        self.assertEqual(BinaryHunt([1, 2, 3, 4, 5], 5).search(), 4)
        self.assertEqual(BinaryHunt([1, 2, 3, 4, 5], 6).search(), -1)

    def test_parts(self):
        self.assertEqual(sorted(Parts([1, 2, 3]).generate(), key=lambda x: (len(x), x)),
                         sorted([[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]], key=lambda x: (len(x), x)))
        self.assertEqual(sorted(Parts([1]).generate(), key=lambda x: (len(x), x)),
                         sorted([[], [1]], key=lambda x: (len(x), x)))
        self.assertEqual(sorted(Parts([]).generate(), key=lambda x: (len(x), x)),
                         sorted([[]], key=lambda x: (len(x), x)))

    def test_rearranger(self):
        self.assertEqual(sorted(Rearranger("abc").generate()), sorted(['abc', 'acb', 'bac', 'bca', 'cab', 'cba']))
        self.assertEqual(sorted(Rearranger("a").generate()), sorted(['a']))
        self.assertEqual(sorted(Rearranger("").generate()), sorted(['']))

    def test_fusion_sort(self):
        self.assertEqual(FusionSort([34, 7, 23, 32, 5, 62]).sort(), [5, 7, 23, 32, 34, 62])
        self.assertEqual(FusionSort([1, 2, 3, 4, 5]).sort(), [1, 2, 3, 4, 5])
        self.assertEqual(FusionSort([5, 4, 3, 2, 1]).sort(), [1, 2, 3, 4, 5])

    def test_rapid_sort(self):
        self.assertEqual(RapidSort([10, 80, 30, 90, 40, 50, 70]).sort(), [10, 30, 40, 50, 70, 80, 90])
        self.assertEqual(RapidSort([1, 2, 3, 4, 5]).sort(), [1, 2, 3, 4, 5])
        self.assertEqual(RapidSort([5, 4, 3, 2, 1]).sort(), [1, 2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()