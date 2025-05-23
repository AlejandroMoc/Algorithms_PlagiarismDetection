import unittest
import numpy as np
import scipy.stats as stats
from probability import (
    calculate_correlation_coefficient, calculate_poisson_probability, calculate_normal_distribution_pdf,
    calculate_normal_distribution_cdf, calculate_log_normal_distribution_pdf, calculate_gamma_distribution_pdf,
    calculate_z_value, calculate_binomial_probability, calculate_exponential_probability
)

class TestProbabilityFunctions(unittest.TestCase):
    
    def test_correlation_coefficient(self):
        data_x = np.array([1, 2, 3, 4, 5])
        data_y = np.array([2, 4, 6, 8, 10])
        self.assertAlmostEqual(calculate_correlation_coefficient(data_x, data_y), 1.0, places=4)
    
    def test_poisson_probability(self):
        self.assertAlmostEqual(calculate_poisson_probability(3, 2), stats.poisson.pmf(2, 3), places=4)
    
    def test_normal_distribution_pdf(self):
        self.assertAlmostEqual(calculate_normal_distribution_pdf(0, 0, 1), stats.norm.pdf(0, 0, 1), places=4)
    
    def test_normal_distribution_cdf(self):
        self.assertAlmostEqual(calculate_normal_distribution_cdf(0, 0, 1), stats.norm.cdf(0, 0, 1), places=4)
    
    def test_log_normal_distribution_pdf(self):
        self.assertAlmostEqual(calculate_log_normal_distribution_pdf(1, 0, 1), stats.lognorm.pdf(1, 1, scale=np.exp(0)), places=4)
    
    def test_gamma_distribution_pdf(self):
        self.assertAlmostEqual(calculate_gamma_distribution_pdf(2, 3, 2), stats.gamma.pdf(2, 3, scale=1/2), places=4)
    
    def test_z_value(self):
        self.assertAlmostEqual(calculate_z_value(5, 2, 1), 3.0, places=4)
    
    def test_binomial_probability(self):
        self.assertAlmostEqual(calculate_binomial_probability(10, 0.5, 5), stats.binom.pmf(5, 10, 0.5), places=4)
    
    def test_exponential_probability(self):
        self.assertAlmostEqual(calculate_exponential_probability(2, 3), stats.expon.pdf(3, scale=1/2), places=4)

if __name__ == "__main__":
    unittest.main()