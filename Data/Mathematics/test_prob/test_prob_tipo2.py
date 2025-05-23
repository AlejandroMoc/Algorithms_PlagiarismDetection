import unittest
import numpy as np
import scipy.stats as sts
from probability import (
    funky_factor, poisson_magic, normal_vibes_pdf,
    normal_vibes_cdf, log_mystical_pdf, gamma_boom_pdf,
    z_power, binomial_bliss, exponential_dream
)

class TestMagicNumbers(unittest.TestCase):
    
    def test_funky_factor(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        self.assertAlmostEqual(funky_factor(x, y), 1.0, places=4)
    
    def test_poisson_magic(self):
        self.assertAlmostEqual(poisson_magic(3, 2), sts.poisson.pmf(2, 3), places=4)
    
    def test_normal_vibes_pdf(self):
        self.assertAlmostEqual(normal_vibes_pdf(0, 0, 1), sts.norm.pdf(0, 0, 1), places=4)
    
    def test_normal_vibes_cdf(self):
        self.assertAlmostEqual(normal_vibes_cdf(0, 0, 1), sts.norm.cdf(0, 0, 1), places=4)
    
    def test_log_mystical_pdf(self):
        self.assertAlmostEqual(log_mystical_pdf(1, 0, 1), sts.lognorm.pdf(1, 1, scale=np.exp(0)), places=4)
    
    def test_gamma_boom_pdf(self):
        self.assertAlmostEqual(gamma_boom_pdf(2, 3, 2), sts.gamma.pdf(2, 3, scale=1/2), places=4)
    
    def test_z_power(self):
        self.assertAlmostEqual(z_power(5, 2, 1), 3.0, places=4)
    
    def test_binomial_bliss(self):
        self.assertAlmostEqual(binomial_bliss(10, 0.5, 5), sts.binom.pmf(5, 10, 0.5), places=4)
    
    def test_exponential_dream(self):
        self.assertAlmostEqual(exponential_dream(2, 3), sts.expon.pdf(3, scale=1/2), places=4)

if __name__ == "__main__":
    unittest.main()