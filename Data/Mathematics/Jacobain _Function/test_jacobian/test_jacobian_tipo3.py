import unittest
import sympy as sp
import numpy as np
from jacobian import symbolic_jacobian, numerical_jacobian

class TestJacobianFunctions(unittest.TestCase):

    def test_symbolic_jacobian(self):
        # Test symbolic Jacobian
        var1, var2 = sp.symbols('var1 var2')
        funcs = [var1**2 + var2**2, sp.sin(var1) * sp.cos(var2)]
        variables = [var1, var2]
        jacobian_sym = symbolic_jacobian(funcs, variables)
        expected_jacobian = sp.Matrix([[2*var1, 2*var2], [sp.cos(var1)*sp.cos(var2), -sp.sin(var1)*sp.sin(var2)]])
        print("Symbolic Jacobian:", jacobian_sym)
        self.assertTrue(jacobian_sym.equals(expected_jacobian))

        # Additional symbolic example
        var3, var4 = sp.symbols('var3 var4')
        funcs2 = [var3**3 + var4**3, sp.exp(var3) * sp.log(var4)]
        variables2 = [var3, var4]
        jacobian_sym2 = symbolic_jacobian(funcs2, variables2)
        expected_jacobian2 = sp.Matrix([[3*var3**2, 3*var4**2], [sp.exp(var3)*sp.log(var4), sp.exp(var3)/var4]])
        print("Additional Symbolic Jacobian:", jacobian_sym2)
        self.assertTrue(jacobian_sym2.equals(expected_jacobian2))

    def test_numerical_jacobian(self):
        # Test numerical Jacobian
        def example_func(point):
            var1, var2 = point
            return np.array([var1**2 + var2**2, np.sin(var1) * np.cos(var2)])

        point = np.array([1.0, 2.0])
        jacobian_num = numerical_jacobian(example_func, point)
        expected_jacobian = np.array([[2.0, 4.0], [-0.22484, -0.76515]])
        print("Numerical Jacobian at point [1.0, 2.0]:", jacobian_num)
        np.testing.assert_almost_equal(jacobian_num, expected_jacobian, decimal=5)

        # Additional numerical example
        def example_func2(point):
            var3, var4 = point
            return np.array([var3**3 + var4**3, np.exp(var3) * np.log(var4)])

        point2 = np.array([1.0, 2.0])
        jacobian_num2 = numerical_jacobian(example_func2, point2)
        expected_jacobian2 = np.array([[3.0, 12.0], [np.exp(1.0) * np.log(2.0), np.exp(1.0) / 2.0]])
        print("Numerical Jacobian at point [1.0, 2.0] for additional example:", jacobian_num2)
        np.testing.assert_almost_equal(jacobian_num2, expected_jacobian2, decimal=5)

if __name__ == '__main__':
    unittest.main()