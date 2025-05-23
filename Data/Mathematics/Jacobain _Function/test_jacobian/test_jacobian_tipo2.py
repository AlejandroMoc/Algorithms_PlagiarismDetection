import unittest
import sympy as sp
import numpy as np
from jacobian import symbolic_jacobian, numerical_jacobian

class TestJacobianFunctions(unittest.TestCase):

    def test_galactic_jacobian(self):
        # Test symbolic Jacobian
        star, planet = sp.symbols('star planet')
        cosmos = [star**2 + planet**2, sp.sin(star) * sp.cos(planet)]
        galaxies = [star, planet]
        nebula_jacobian = symbolic_jacobian(cosmos, galaxies)
        expected_nova = sp.Matrix([[2*star, 2*planet], [sp.cos(star)*sp.cos(planet), -sp.sin(star)*sp.sin(planet)]])
        print("Symbolic Jacobian:", nebula_jacobian)
        self.assertTrue(nebula_jacobian.equals(expected_nova))

        # Additional symbolic example
        moon, comet = sp.symbols('moon comet')
        space = [moon**3 + comet**3, sp.exp(moon) * sp.log(comet)]
        planets = [moon, comet]
        universe_jacobian = symbolic_jacobian(space, planets)
        expected_cosmic = sp.Matrix([[3*moon**2, 3*comet**2], [sp.exp(moon)*sp.log(comet), sp.exp(moon)/comet]])
        print("Additional Symbolic Jacobian:", universe_jacobian)
        self.assertTrue(universe_jacobian.equals(expected_cosmic))

    def test_interstellar_jacobian(self):
        # Test numerical Jacobian
        def star_system(point):
            star, planet = point
            return np.array([star**2 + planet**2, np.sin(star) * np.cos(planet)])

        coordinates = np.array([1.0, 2.0])
        jacobian_galaxy = numerical_jacobian(star_system, coordinates)
        expected_jovian = np.array([[2.0, 4.0], [-0.22484, -0.76515]])
        print("Numerical Jacobian at point [1.0, 2.0]:", jacobian_galaxy)
        np.testing.assert_almost_equal(jacobian_galaxy, expected_jovian, decimal=5)

        # Additional numerical example
        def star_system2(point):
            moon, comet = point
            return np.array([moon**3 + comet**3, np.exp(moon) * np.log(comet)])

        coordinates2 = np.array([1.0, 2.0])
        jacobian_galaxy2 = numerical_jacobian(star_system2, coordinates2)
        expected_jovian2 = np.array([[3.0, 12.0], [np.exp(1.0) * np.log(2.0), np.exp(1.0) / 2.0]])
        print("Numerical Jacobian at point [1.0, 2.0] for additional example:", jacobian_galaxy2)
        np.testing.assert_almost_equal(jacobian_galaxy2, expected_jovian2, decimal=5)

if __name__ == '__main__':
    unittest.main()